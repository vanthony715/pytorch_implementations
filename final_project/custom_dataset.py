# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import tv_tensors
import torchvision.transforms.functional as fn
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class CustomImageDatasetObjectDetection(Dataset):
    def __init__(self, labelspath, imgspath, transform=None, target_transform=None):

        ##define transforms
        self.transform = transform
        self.target_transform = target_transform

        ##use dictionary to keep track of images
        data_dict = {}

        ##get the image file full paths
        imgfiles = sorted(os.listdir(imgspath))
        data_dict['imgpath'] = [imgspath + file for file in imgfiles]

        ##get the label file full paths
        labelfiles = sorted(os.listdir(labelspath))
        data_dict['labelpath'] = [labelspath + file for file in labelfiles]

        ##debug
        # print(data_dict['labelpath'])

        ##get labels and bboxes
        labels, bboxes = [], []
        for labelpath in tqdm(data_dict['labelpath'], 'Reading Labels'):
            with open(labelpath, 'r') as f:
                lines = f.readlines()

            ##a list of labels and bbox coordinates
            labels.append([line[0] for line in lines])
            bboxes.append([line[2:] for line in lines])

        ##remove strings from bbox coordinates
        for i, boxes in enumerate(bboxes):
            ##convert bbox strings to float coordinates
            boxes = [np.array(i.replace('\n', '').split(' ')).astype(float) for i in boxes]
            bboxes[i] = boxes
        # print(boxes)

        ##add to data_dict
        data_dict['labels'] = labels
        data_dict['bboxes'] = bboxes

        ##convert dict to df
        self.df = pd.DataFrame(data_dict)

    def __getitem__(self, idx):
        ##open image
        img_path = self.df.iloc[idx]['imgpath']
        image = fn.resize(read_image(img_path), size=[256, 256])
        # print(image.shape)

        torch.FloatTensor

        ##retrieve labels
        labels = torch.FloatTensor(np.array(self.df.iloc[idx]['labels']).astype(np.float64))
        # print("Len Labels: ", labels.shape)

        ##retrieve bboxes
        bboxes = torch.FloatTensor(np.array(self.df.iloc[idx]['bboxes']).astype(np.float64))
        # print("Len Bboxes: ", bboxes.shape)

        ##transform image if applicable
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, bboxes, labels

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        """
        Custom Collate of Object Detection

        Credit:
            https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        print(images.shape)

        return images[0], boxes[0], labels[0]  # tensor (N, 3, 300, 300), 3 lists of N tensors each
