import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from codes_for_kaist.utils_fusion import transform, parse_annotation

class KaistPDDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, img_type, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split
        self.images = list()
        self.objects = list()
        self.img_type = img_type


        assert self.split in {'train', 'test'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult


        with open(os.path.join(data_folder, self.split + '-all-20.txt')) as f:
            lines = f.read().splitlines()

    

        # lines[?] => set11/V001/I00859

        # 이미지 경로 저장
        # /home/urp4/workspace/src/ssd/datasets/kaist/images/set00/V000/visible/I01217.jpg

        img_id = 0
    

        for line in lines:
            split_line = line.rsplit("/", 1)
            folder = split_line[0] # ex) 'set00/v000'
            image_num = split_line[1]

            img_path = os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/images', folder, img_type, image_num + '.jpg')
            self.images.append(img_path)
            
            objects = parse_annotation(os.path.join('/home/urp4/workspace/src/ssd/datasets/kaist/annotation_json', line + '.json'), img_id)

            img_id += 1
            self.objects.append(objects)


        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image

        # if (self.img_type == 'visible'):
        #     image = Image.open(self.images[i], mode='r')
        # elif (self.img_type == 'lwir'):
        #     image = Image.open(self.images[i], mode='r').convert('L')

        image = Image.open(self.images[i], mode='r')
            
        image = image.convert('RGB')
        
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor([0] * len(objects['boxes']))  # (n_objects) object 개수 만큼 0 저장 


        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties, i # i: image index (organized in the sequence listed in '~-all-20.txt' file)

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        image_ids = list()

        for b in batch:
            images.append(b[0])
            
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
            image_ids.append(b[4])
        
        images = torch.stack(images, dim=0)
 

        return images, boxes, labels, difficulties, image_ids  # tensor (N, 3, 300, 300), 3 lists of N tensors each
