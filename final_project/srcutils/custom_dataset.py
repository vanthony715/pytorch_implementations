# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import os
import numpy as np

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDatasetObjectDetection(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def _xywh_to_xyxy(self, x, y, w, h):
        ##unnormalize coordinates (yolo use normalized coords)
        wu = w * 256
        hu = h * 256
        xu = x * 256 - 0.5 * (wu)
        yu = y * 256 - 0.5 * (hu)
        return xu, yu, xu+wu, yu+hu

    def _parse_txt(self, filepath):
        ##get labels and bboxes
        labels, bboxes = [], []

        ##parse text file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        ##get labels and bboxes seperately
        for line in lines:
            ##check if single digit for class
            if line[1] == '':
                labels.append(int(line[0]))
                bboxes.append(line[2:].split(' '))
            ##assume that the class is two digit such as 10, 11, etc..
            else:
                labels.append(int(line[0:1]))
                bboxes.append(line[3:].split(' '))


        ##remove strings from bbox coordinates
        converted_bboxes = []
        for i, boxes in enumerate(bboxes):
            ##convert bbox strings to float coordinates
            try:
                box = list([i.replace('\n', '') for i in boxes])
            except:
                pass
            box = [float(i) for i in box]

            ##convert to xyxy format
            x, y, w, h = box
            xyxy_coords = self._xywh_to_xyxy(x, y, w, h)
            for i, coord in enumerate(xyxy_coords):
                box[i] = coord
                # print(box)

            converted_bboxes.append(box)
        # print(bboxes)
        return labels, converted_bboxes

    def __getitem__(self, idx):
        ## load image and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = read_image(img_path)

        ##tuple contains class and bbox info
        target_data, bboxes = self._parse_txt(annot_path)
        image_id = idx

        ##Wrap sample and targets into torchvision tv_tensors
        img = tv_tensors.Image(img)

        ##target tensor
        target = {}
        target["area"] = torch.as_tensor(np.zeros(len(bboxes)))
        target["iscrowd"] = torch.as_tensor([False for i in range(len(bboxes))])
        target["labels"] = torch.as_tensor(target_data)
        target["image_id"] = image_id
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes, format="XYXY",
                                                   canvas_size=F.get_size(img))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    # def collate_fn(self, batch):
    #     """
    #     Custom Collate of Object Detection

    #     Credit:
    #         https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py
    #     """

    #     images = list()
    #     boxes = list()
    #     labels = list()

    #     for b in batch:
    #         images.append(b[0])
    #         boxes.append(b[1])
    #         labels.append(b[2])

    #     images = torch.stack(images, dim=0)

    #     print(images.shape)

    #     return images[0], boxes[0], labels[0]  # tensor (N, 3, 300, 300), 3 lists of N tensors each
