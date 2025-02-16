import os
from PIL import Image
from PIL import ImageFont
from torchvision.utils import draw_bounding_boxes 
import json
from torchvision.utils import save_image
import torchvision.transforms.functional as FT
import torch
import cv2
import numpy as np
import tqdm
import torchvision

# annotations 가져오기
# fn gt id 순회해서 ...
# gt id 에 해당하는 gt만을 저장하기 딕셔너리로 저장해야겠다
# 각 이미지 위에 gt 그리기 이미지당 gt를 리스트로 저장하고 있어야겠다 근데 모든 gt를 저장할 필요는 없고 fn인 거에 해당하는 것만 

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/remained_FNs_edit.json', 'r') as j:
    remained_FNs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/new_FNs.json', 'r') as j:
    new_FNs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/kaist_annotations_test20.json', 'r') as j:
    kaist_annotation = json.load(j)

gt_info = kaist_annotation['annotations'] # list of dictionaries
empty_dict = dict()

for fnid in remained_FNs:
    image_id = gt_info[fnid]['image_id']
    if image_id not in empty_dict:
        empty_dict[image_id] = []  # Initialize an empty list if key doesn't exist
    empty_dict[image_id].append(gt_info[fnid]['bbox'])

# for fnid in new_FNs:
#     image_id = gt_info[fnid]['image_id']
#     if image_id not in empty_dict:
#         empty_dict[image_id] = []  # Initialize an empty list if key doesn't exist
#     if gt_info[fnid]['bbox'] in empty_dict[image_id] :
        
#         continue
#     else :
#         empty_dict[image_id].append(gt_info[fnid]['bbox'])




data_folder = '../../src/ssd/datasets/kaist'
images = list()
img_type = 'visible'
gt_boxes = list()
gt_labels = list()
p_boxes = list()
p_labels = list()
p_boxes_lwir = list()
p_labels_lwir = list()
p_boxes_fusion = list()
p_labels_fusion = list()
width = 640
height = 512

kaist_labels = {0: 'others', 1: 'person'}

labelsColor = {0: 'yellow', 1: 'green'}
labelsColor_gt = {0: 'purple', 1: 'red'}


# for line in lines:
#     split_line = line.rsplit("/", 1)
#     folder = split_line[0] # ex) 'set00/v000'
#     image_num = split_line[1]

#     img_path = os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/images', folder, img_type, image_num + '.jpg')
#     images.append(img_path)

#     objects = parse_annotation(os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/annotation_json', line + '.json'), img_id)
#     gt_boxes.append(objects['boxes'])

with open('../TEST_images_lwir.json') as j:
    image_paths_lwir = json.load(j)

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

with open(os.path.join('../prediction_files/lwir/base_code.json')) as j:
    predictions_lwir = json.load(j)

for objects in predictions_lwir:
    p_boxes_lwir.append(objects['boxes'])
    p_labels_lwir.append(objects['labels'])

with open(os.path.join('/home/urp4/workspace/codes_for_kaist/prediction_files/fusion/80_halfway.json')) as j:
    predictions_fusion = json.load(j)

for objects in predictions_fusion:
    p_boxes_fusion.append(objects['boxes'])
    p_labels_fusion.append(objects['labels'])

with open(os.path.join(data_folder, 'test-all-20.txt')) as f:
    lines = f.read().splitlines()


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # define codec
prev_folder = ''

# gtId = 0

for i, (line) in enumerate(lines):
    split_line = line.rsplit("/", 1)
    folder = split_line[0] # ex) 'set00/v000'
    image_num = split_line[1]

    image = Image.open(image_paths[i], mode='r')
    image = FT.to_tensor(image) 

    image_lwir = Image.open(image_paths_lwir[i], mode='r')
    image_lwir = FT.to_tensor(image_lwir) 

    # gt_ids = []

    # for j in range(len(gt_boxes[i])):
    #     gt_ids.append(gtId + j)
    
    # gtId += len(gt_boxes[i])


    if (folder != prev_folder):
        if (i != 0):
            out.release()
            

        output = '../plotted_videos/remainFNs6/{}'.format(folder.split('/')[0])
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, folder.split('/')[1] + '.mp4')
        
        out = cv2.VideoWriter(output, fourcc, 5.0, (width*2, height*3))
        prev_folder = folder
    


    gt_labelsColor = [labelsColor_gt[l] for l in gt_labels[i]]
    
    # for j, idx in enumerate(gt_ids):
    #     if (idx in remained_FNs):
    #         gt_labelsColor[j] = 'white'

    p_labelsColor = [labelsColor[l] for l in p_labels[i]]
    p_labelsColor_lwir = [labelsColor[l] for l in p_labels_lwir[i]]
    p_labelsColor_fusion = [labelsColor[l] for l in p_labels_fusion[i]]
    
    gt_labels[i] = [kaist_labels[l] for l in gt_labels[i]]
    p_labels[i] = [kaist_labels[l] for l in p_labels[i]]
    p_labels_lwir[i] = [kaist_labels[l] for l in p_labels_lwir[i]]
    p_labels_fusion[i] = [kaist_labels[l] for l in p_labels_fusion[i]]
    
    image_remainFNs = image
    image_lwir_remainFNs = image_lwir

    if i in empty_dict.keys():
        bbox_ten = torch.tensor(empty_dict[i])
        bbox_ten = torchvision.ops.box_convert(bbox_ten, in_fmt='xywh', out_fmt = 'xyxy')
        image_remainFNs = draw_bounding_boxes(image, bbox_ten, width=1, colors='orange')
        image_lwir_remainFNs = draw_bounding_boxes(image_lwir, bbox_ten, width=1, colors='orange')

    
    img_single = draw_bounding_boxes(image, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors=gt_labelsColor)
    img_single = draw_bounding_boxes(img_single, torch.tensor(p_boxes[i]), labels=p_labels[i], width=1, colors=p_labelsColor)

    img_ir_single = draw_bounding_boxes(image_lwir, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors=gt_labelsColor)
    img_ir_single = draw_bounding_boxes(img_ir_single, torch.tensor(p_boxes_lwir[i]), labels=p_labels_lwir[i], width=1, colors=p_labelsColor_lwir)
    
    image_fusion = draw_bounding_boxes(image, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors=gt_labelsColor)
    image_fusion = draw_bounding_boxes(image_fusion, torch.tensor(p_boxes_fusion[i]), labels=p_labels_fusion[i], width=1, colors=p_labelsColor_fusion)

    image_lwir_fusion = draw_bounding_boxes(image_lwir, torch.tensor(gt_boxes[i]), labels=gt_labels[i], width=1, colors=gt_labelsColor)
    image_lwir_fusion = draw_bounding_boxes(image_lwir_fusion, torch.tensor(p_boxes_fusion[i]), labels=p_labels_fusion[i], width=1, colors=p_labelsColor_fusion)

    


    cimg_single = torch.cat([img_single, img_ir_single], dim=2) 
    cimg_fusion = torch.cat([image_fusion, image_lwir_fusion], dim=2) 
    cimg_remainFNs = torch.cat([image_remainFNs, image_lwir_remainFNs], dim=2) 
    cimg = torch.cat([cimg_single, cimg_fusion], dim=1) 
    cimg = torch.cat([cimg, cimg_remainFNs], dim=1) 


    
    # image = image.permute(1, 2, 0).numpy()
    # image = image[:, :, ::-1]
    # image = (image*255).astype(np.uint8)
    cimg = cimg.permute(1, 2, 0).numpy()
    cimg = cimg[:, :, ::-1]
    cimg = (cimg*255).astype(np.uint8)
    
    out.write(cimg)
    # out.write(image)


out.release()
print('Videos saved.')



