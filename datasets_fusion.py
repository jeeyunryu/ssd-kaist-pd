import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageOps
from utils_fusion import transform

from torchvision.utils import save_image
import torchvision


class KaistPDDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """


    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        
        with open(os.path.join(data_folder, self.split + '_images_lwir.json'), 'r') as j:
            self.images_lwir = json.load(j)

        assert len(self.images) == len(self.objects) == len(self.images_lwir)
 
    def __getitem__(self, i):
        # Read image
        
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        
        image_lwir = Image.open(self.images_lwir[i], mode='r')
        image_lwir = image_lwir.convert('RGB')
    
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor([0] * len(objects['boxes']))  # (n_objects)


        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, image_lwir, boxes, labels, difficulties = transform(image, image_lwir, boxes, labels, difficulties, split=self.split)

        return image, image_lwir, boxes, labels, difficulties, i

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
        images_lwir = list()
        boxes = list()
        labels = list()
        difficulties = list()
        image_ids = list()

        for b in batch:
            images.append(b[0])
            images_lwir.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])
            image_ids.append(b[5])
            # for i in range(b[1].size(0)):
            #     image_ids.append(b[4])

        images = torch.stack(images, dim=0)
        images_lwir = torch.stack(images_lwir, dim=0)
 

        return images, images_lwir, boxes, labels, difficulties, image_ids  # tensor (N, 3, 300, 300), 3 lists of N tensors each
