import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model_IA import SSD300, MultiBoxLoss, IALoss
from datasets_IA import KaistPDDataset
# from torchvision.utils import save_image
from plotting.plot import plot_box
# from knockknock import desktop_sender


from utils_fusion import *
from tqdm import tqdm
# from torchvision.utils import draw_bounding_boxes   

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint =  None # path to model checkpoint, None if none
batch_size = 16  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 20  # print training status every __ batches
save_chkpnt_freq = 10 # save checkpoint every ___ epoch
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
        optimizer_gate = torch.optim.SGD([model.alpha, model.beta], lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        optimizer_gate = checkpoint['optimizer_gate']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    criterion_illum = IALoss().to(device)

    

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
            adjust_learning_rate(optimizer_gate, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              criterion_illum=criterion_illum,
              optimizer=optimizer,
              optimizer_gate=optimizer_gate,
              epoch=epoch)

        # Save checkpoint every 20 epochs
        if epoch != 0 and epoch % save_chkpnt_freq == 0:
            save_checkpoint(epoch, model, optimizer, filename='./IA_fusion_80_{}ep.pth.tar'.format(epoch))
        # save_checkpoint(epoch, model, optimizer, optimizer_gate, filename='./')
    save_checkpoint(epoch, model, optimizer, optimizer_gate, filename='./IA_fusion_80_fin.pth.tar')
    

# @desktop_sender(title="Knockknock Desktop Notifier")
# 
def train(train_loader, model, criterion, criterion_illum, optimizer, optimizer_gate, epoch):
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
    for i, (images, images_t, images_lwir, boxes, labels, _, image_id) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - start)
        # plot_box(images, boxes, labels, image_id)   
        
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        images_t = images_t.to(device) 
        images_lwir = images_lwir.to(device)
        boxes = [b.to(device) for b in boxes]
        illum_labels = torch.tensor([1 if id < 5428 else 0 for id in image_id]) # 1: day, 0: night
        
        
        labels = [l.to(device) for l in labels]
    
        # Forward prop.
        # predicted_locs, predicted_scores = model(images, images_lwir)  # (N, 8732, 4), (N, 8732, n_classes)

        pred_locs, pred_scores, pred_locs_lwir, pred_scores_lwir, pred_final, pred_scores_final, illum_value = model(images, images_t, images_lwir)
        

        # Loss
        # loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        loss = criterion(pred_locs, pred_scores, boxes, labels)
        loss += criterion(pred_locs_lwir, pred_scores_lwir, boxes, labels)
        loss += criterion_illum(illum_value, illum_labels)
        
        # Backward prop.
        optimizer.zero_grad()
        loss_clone = loss.clone()
        loss_clone.backward(retain_graph=True)
    

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        for name, param in model.named_parameters():
            if name not in ['alpha', 'beta']:
                param.requires_grad = False
            
    
        loss_fusion = criterion(pred_final, pred_scores_final, boxes, labels)

        # Backward prop.
        optimizer_gate.zero_grad()
        loss_fusion.backward()
        

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_gate, grad_clip)

        # Update model
        optimizer_gate.step()

        for param in model.parameters():
            param.requires_grad = True

            

        
        
        total_loss = loss + loss_fusion

        losses.update(total_loss.item(), images.size(0)) # Why is loss in avg form? Why is it not handed independently?
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
            with open(os.path.join('./loss_IA_fusion_80.json'), 'w') as j: 
                json.dump(save_loss, j) 
    del pred_locs, pred_scores, pred_locs_lwir, pred_scores_lwir, pred_final, pred_scores_final, illum_value, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    random_seed()
    main()
