import os
from PIL import Image, ImageOps
from PIL import ImageFont
from torchvision.utils import draw_bounding_boxes 
import json
from torchvision.utils import save_image
import torchvision.transforms.functional as FT
import torch
import cv2
import numpy as np


data_folder = '../../src/ssd/datasets/kaist'
images = list()
img_type = 'visible'
gt_boxes = list()
gt_labels = list()
p_boxes = list()
p_labels = list()
# p_boxes_lwir = list()
# p_labels_lwir = list()
p_boxes_he = list()
p_labels_he = list()
width = 640
height = 512

kaist_labels = {0: 'others', 1: 'person'}
labels_color = {0: 'yellow', 1: 'green'}


# for line in lines:
#     split_line = line.rsplit("/", 1)
#     folder = split_line[0] # ex) 'set00/v000'
#     image_num = split_line[1]

#     img_path = os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/images', folder, img_type, image_num + '.jpg')
#     images.append(img_path)

#     objects = parse_annotation(os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/annotation_json', line + '.json'), img_id)
#     gt_boxes.append(objects['boxes'])

# with open('../TEST_images_lwir.json') as j:
#     image_paths_lwir = json.load(j)

with open('../TEST_images.json') as j:
    image_paths = json.load(j)

with open('../TEST_objects.json') as j:
    gt_objects = json.load(j)

for objects in gt_objects:
    gt_boxes.append(objects['boxes'])
    gt_labels.append(objects['labels'])

with open(os.path.join('../prediction_files/visible/base_code.json')) as j:
    predictions = json.load(j)

for objects in predictions:
    p_boxes.append(objects['boxes'])
    p_labels.append(objects['labels'])

with open(os.path.join('/home/urp4/workspace/codes_for_kaist/prediction_files/visible/80_he.json')) as j:
    predictions_he = json.load(j)

for objects in predictions_he:
    p_boxes_he.append(objects['boxes'])
    p_labels_he.append(objects['labels'])



with open(os.path.join(data_folder, 'test-all-20.txt')) as f:
    lines = f.read().splitlines()


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # define codec
prev_folder = ''

for i, (line) in enumerate(lines):
    split_line = line.rsplit("/", 1)
    folder = split_line[0] # ex) 'set00/v000'
    image_num = split_line[1]
    

    image = Image.open(image_paths[i], mode='r')
    image_he = ImageOps.equalize(image)
    image = FT.to_tensor(image) 
    image_he = FT.to_tensor(image_he) 

    # image_lwir = Image.open(image_paths_lwir[i], mode='r')
    # image_lwir = FT.to_tensor(image_lwir) 

    if (folder != prev_folder):
        if (i != 0):
            out.release()

        output = '../videos_color_he/{}'.format(folder.split('/')[0])
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, folder.split('/')[1] + '.mp4')
        
        out = cv2.VideoWriter(output, fourcc, 5.0, (width*2, height))
        prev_folder = folder

    p_labels_color = [labels_color[l] for l in p_labels[i]]
    # p_labels_color_lwir = [labels_color[l] for l in p_labels_lwir[i]]
    p_labels_color_he = [labels_color[l] for l in p_labels_he[i]]
    
    gt_labels[i] = [kaist_labels[l] for l in gt_labels[i]]
    p_labels[i] = [kaist_labels[l] for l in p_labels[i]]
    p_labels_he[i] = [kaist_labels[l] for l in p_labels_he[i]]
    
   
    image = draw_bounding_boxes(image, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors='red')
    image = draw_bounding_boxes(image, torch.tensor(p_boxes[i]), labels=p_labels[i], width=1, colors=p_labels_color)

    image_he = draw_bounding_boxes(image_he, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors='red')
    image_he = draw_bounding_boxes(image_he, torch.tensor(p_boxes_he[i]), labels=p_labels_he[i], width=1, colors=p_labels_color_he)


    combined_image = torch.cat([image, image_he], dim=2) 
    
    combined_image = combined_image.permute(1, 2, 0).numpy()
    combined_image = combined_image[:, :, ::-1]
    combined_image = (combined_image*255).astype(np.uint8)
    
    out.write(combined_image)


out.release()
print('Videos saved.')


# for i in range(5):

#     image = Image.open(image_paths[i], mode='r')
#     image = FT.to_tensor(image) 

#     image = draw_bounding_boxes(image, torch.tensor(gt_boxes[i]), width=2, colors='red')
#     image = draw_bounding_boxes(image, torch.tensor(p_boxes[i]), width=2, colors='green')

    

#     # for box in gt_boxes[i]:
#     #     import pdb;pdb.set_trace() # box: ex) gt_boxes[0][0]
#     #     box = torch.tensor(box).unsqueeze(0)
#     #     image = draw_bounding_boxes(image, box, width=5, colors='red') # draw_bounding_boxes excepts box coordinates of type 'tensor'
#     # for box in p_boxes[i]:
#     #     box = torch.tensor(box)
#     #     image = draw_bounding_boxes(image, box, width=5, colors='green')

#     path = './img_w_bbox/gt_and_p/visible'
#     os.makedirs(path, exist_ok=True)

#     save_image(image, os.path.join(path,'img{}.jpg'.format(i)))

# print('Finished.')
