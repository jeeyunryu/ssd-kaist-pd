from torchvision import transforms
import torch
import json
import os
from PIL import Image, ImageDraw, ImageFont
from datasets_single import KaistPDDataset
from tqdm import tqdm
# from plot_predictedB import plot_box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = './'
keep_difficult = True 
batch_size = 32
workers = 4
width = 640
height = 512

# Load model checkpoint
checkpoint = '/home/urp4/workspace/src/codes_for_pascal/checkpoints_kaist/visible/80_he.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()



test_dataset = KaistPDDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)





res_list = list()
predictions = list()


def detect(img_id, image, min_score, max_overlap, top_k):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    # image = normalize(to_tensor(resize(original_image)))
    # image = normalize(resize(original_image))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image)

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    
    
    
    
    # det_boxes: list of tensors
    original_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)

    



    
    for id, boxes, labels, scores in zip(img_id, det_boxes, det_labels, det_scores):
        
        
        boxes = boxes.to('cpu')
        
        for i in range(boxes.size(0)):
            boxes[i] = boxes[i] * original_dims
        
        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        
        objects = dict()
        objects['image_id'] = id
        objects['boxes'] = boxes
        objects['labels'] = labels
        objects['scores'] = scores
        predictions.append(objects)
    
    # sorted_data = sorted(predictions, key=lambda x: x["image_id"]) # test loader에서 불러와진 이미지는 img_id 순서대로임임
    
    with open(os.path.join('./prediction_files/visible/80_he.json'), 'w') as j: 
        json.dump(predictions, j)

    

    # img_id_new = list()

    # for i in range(len(det_boxes)):
    #     for j in range(len(det_boxes[i])):
    #         img_id_new.append(img_id[i])

    # # Move detections to the CPU
    # # det_boxes = det_boxes.to('cpu')
    # # seperate tensors for each bbox (bboxes for same image is coupled as a tensor)
    # det_boxes_cat = torch.cat(det_boxes, 0) # cover list of tensors to tensor
    # det_labels_cat = torch.cat(det_labels, 0)
    # det_scores_cat = torch.cat(det_scores, 0)
    
    
    
    # det_boxes_cat = torch.cat([det_boxes_cat[:, :2], det_boxes_cat[:, 2:] - det_boxes_cat[:, :2]], 1) # xyxy => xywh

    
    
    # det_boxes_cat = torch.tensor(det_boxes_cat, device =  'cpu')

   

    # # Transform to original image dimensions
    # original_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
    # det_boxes_cat = det_boxes_cat * original_dims

    # # single dict corresponds to one bbox

    

    # for i in range(len(img_id_new)): # iterate number of bboxes
    #     box_dict = dict()
    #     box_dict["image_id"] = img_id_new[i]
    #     box_dict["category_id"] = det_labels_cat[i].item()
    #     box_dict["bbox"] = det_boxes_cat[i].tolist()
    #     box_dict["score"] = det_scores_cat[i].item()
        
    #     res_list.append(box_dict)
        

    return None
    
    
if __name__ == '__main__':

    for i, (image, boxes, labels, difficulties, image_id) in enumerate(tqdm(test_loader)):
        detect(image_id, image, min_score=0.2, max_overlap=0.5, top_k=200)
    
    
    # os.makedirs('./eval_files/lwir', exist_ok=True)
    # with open('./eval_files/lwir/80_base_prev.json', 'w') as j:
    #     json.dump(res_list, j)
    # print('evaluation file created.')



