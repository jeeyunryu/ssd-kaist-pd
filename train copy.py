import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from codes_for_kaist.model_fusion import SSD300, MultiBoxLoss
from codes_for_kaist.datasets_readtxt import KaistPDDataset
# from torchvision.utils import save_image
from plotting.plot import plot_box

from codes_for_kaist.utils_fusion import *
from tqdm import tqdm
# from torchvision.utils import draw_bounding_boxes   


# Data parameters
data_folder = '../src/ssd/datasets/kaist'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint =  None # path to model checkpoint, None if none
batch_size = 32  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 20  # print training status every __ batches
save_chkpnt_freq = 20 # save checkpoint every ___ epoch
lr = 5e-4  # learning rate
# decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_at = 70
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True



save_loss = list()

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at


    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = KaistPDDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here


    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // 32) # divide no iterations by no batches 
    epochs = 80
    # decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at] # len(train_dataset) 
    # Epochs
    for epoch in tqdm(range(start_epoch, epochs)):

        # Decay learning rate at particular epochs
        if epoch == decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint every 20 epochs
        # if epoch != 0 and epoch % save_chkpnt_freq == 0:
        #     save_checkpoint(epoch, model, optimizer, filename='./checkpoints/lwir/checkpoint_{}.pth.tar'.format(epoch))
        save_checkpoint(epoch, model, optimizer, filename='./checkpoints/lwir/testing.pth.tar')
    

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    # initializing

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _, image_id) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # plot_box(images, boxes, labels, image_id)
        

    

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
    
        # Forward prop.
        predicted_locs, predicted_scores = model(images, images_lwir)  # (N, 8732, 4), (N, 8732, n_classes)
        

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0)) # Why is loss in avg form? Why is it not handed independently?
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            save_loss.append(losses.val)
            with open(os.path.join('./loss/lwir_testing.json'), 'w') as j: 
                json.dump(save_loss, j) 
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    random_seed()
    main()
