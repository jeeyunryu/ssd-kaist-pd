# plot GT boxes on non-transformed image

import json
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import save_image
import torchvision
import os, os.path

# with open('/home/urp4/workspace/src/a-PyTorch-Tutorial-to-Object-Detection/TRAIN_images.json', 'r') as j:
#     images = json.load(j)
# with open('/home/urp4/workspace/src/a-PyTorch-Tutorial-to-Object-Detection/TRAIN_objects.json', 'r') as j:
#     objects = json.load(j)
with open('/home/urp4/workspace/src/a-PyTorch-Tutorial-to-Object-Detection/TEST_images.json', 'r') as j:
    images = json.load(j)
with open('/home/urp4/workspace/src/a-PyTorch-Tutorial-to-Object-Detection/TEST_objects.json', 'r') as j:
    objects = json.load(j)



label_color_map = {1: '#e6194b', 0: '#3cb44b'}

img_src = 'test'

for img, object in zip(images, objects):

    result = img.rsplit('/', 4)
    
    set_num = result[1]
    v_num = result[2]
    img_type = result[3]
    img_num = result[4]

    image = Image.open(img, mode='r')
    image = image.convert('RGB')

    
    boxes = torch.FloatTensor(object['boxes']) 
    labels = torch.LongTensor(object['labels'])

    annotated_image = image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    box_location = object['boxes']
    box_label = object['labels']

    for loc_single_box, label in zip(box_location, box_label):
        
        draw.rectangle(xy=loc_single_box, outline=label_color_map[label])
        draw.rectangle(xy=[l + 1. for l in loc_single_box], outline=label_color_map[label])

        text_size = font.getbbox(str(label))
        text_location = [loc_single_box[0] + 2., loc_single_box[1] - text_size[3]]
        textbox_location = [loc_single_box[0], loc_single_box[1] - text_size[1], loc_single_box[0] + text_size[0] + 4.,
                            loc_single_box[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[label])
        draw.text(xy=text_location, text=str(label).upper(), fill='white',
                  font=font)

    totensor = torchvision.transforms.ToTensor()
    tensor_img = totensor(annotated_image)

    path = './img_w_bbox/{}/{}/{}/{}'.format(img_src, img_type, set_num, v_num)
    os.makedirs(path, exist_ok=True)

    save_image(tensor_img, os.path.join(path,'{}_{}'.format(object['image_id'], img_num)))
    

print('Successfully saved.')





