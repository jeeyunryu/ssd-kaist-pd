import json
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import save_image
import torchvision
import os, os.path
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes 


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

# width = 300
# height = 300
width = 640
height = 512

def plot_box(boxes, labels, image_id):

    global original_image

    for boxes, labels, image_id in zip(boxes, labels, image_id):
        
        # annotated_image = to_pil_image(img)
        
        # draw = ImageDraw.Draw(annotated_image)
        # font = ImageFont.load_default()

        img_path = images[image_id]
        original_image = Image.open(img_path, mode='r')
        # original_image = original_image.convert('RGB')

        boxes = boxes.to('cpu')


        original_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
        boxes = boxes * original_dims
        
        for box, label in zip(boxes, labels):
            label = label.item()

            
            box = box.unsqueeze(0)
            
            # box = box * [width, height, width, height]
            # box = box.tolist()

          

            

            totensor = torchvision.transforms.ToTensor() 
            original_image = totensor(original_image)     

            original_image = draw_bounding_boxes(original_image, box, width=5, colors="green", fill=False) 
            

            # draw.rectangle(xy=box, outline=label_color_map[label])
            # draw.rectangle(xy=[l + 1. for l in box], outline=label_color_map[label])

            # text_size = font.getbbox(str(label))
            # text_location = [box[0] + 2., box[1] - text_size[3]]
            # textbox_location = [box[0], box[1] - text_size[1], box[0] + text_size[0] + 4.,
            #                     box[1]]
            # draw.rectangle(xy=textbox_location, fill=label_color_map[label])
            # draw.text(xy=text_location, text=str(label).upper(), fill='white',
            #         font=font)
        
        # img = torchvision.transforms.ToPILImage()(img) 


        # totensor = torchvision.transforms.ToTensor()
        # tensor_img = totensor(annotated_image)

        path = './img_w_bbox/detected/color_model'
        os.makedirs(path, exist_ok=True)

        save_image(original_image, os.path.join(path,'img{}.jpg'.format(image_id)))
    print('Images all saved')



    return None

