from PIL import ImageOps, Image
import cv2
import json
import os
import numpy as np
import torchvision.transforms.functional as FT
from torchvision.utils import draw_bounding_boxes 
import torch

prev_folder = ''
width = 640
height = 512
gt_boxes = list()
gt_labels = list()

with open('./TRAIN_objects.json') as j:
    gt_objects = json.load(j)

for objects in gt_objects:
    gt_boxes.append(objects['boxes'])
    gt_labels.append(objects['labels'])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

with open(os.path.join('./TRAIN_images.json'), 'r') as j:
    images = json.load(j)

with open(os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/train-all-20.txt')) as f:
    lines = f.read().splitlines()

kaist_labels = {0: 'others', 1: 'person'}

for i, (line) in enumerate(lines):
    img = Image.open(images[i], 'r')
    img = img.convert('RGB')
    img_he = ImageOps.equalize(img)

    img = FT.to_tensor(img) 
    img_he = FT.to_tensor(img_he) 

    split_line = line.rsplit("/", 1)
    folder = split_line[0] # ex) 'set00/v000'
    image_num = split_line[1]

    if (folder != prev_folder):
        if (i != 0):
            out.release()

        output = './videos_he/{}'.format(folder.split('/')[0])
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, folder.split('/')[1] + '.mp4')
        
        out = cv2.VideoWriter(output, fourcc, 5.0, (width*2, height))
        prev_folder = folder

    gt_labels[i] = [kaist_labels[l] for l in gt_labels[i]]

    img = draw_bounding_boxes(img, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors='red')
    img_he = draw_bounding_boxes(img_he, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors='red')

    combined_image = torch.cat([img, img_he], dim=2)
    combined_image = combined_image.permute(1, 2, 0).numpy()
    combined_image = combined_image[:, :, ::-1]
    combined_image = (combined_image*255).astype(np.uint8)

    out.write(combined_image)




