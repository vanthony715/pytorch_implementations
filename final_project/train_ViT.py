# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

##define libraries
import os, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from custom_dataset import CustomImageDatasetObjectDetection

if __name__ == "__main__":

    ##define paths
    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/numbers_mnist/"
    trainpath = basepath + 'train/'
    valpath = basepath + 'val/'

    ##define transforms
    transforms = v2.Compose([v2.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224]),])
    transforms = None

    ##initialize datasets
    train_data = CustomImageDatasetObjectDetection(trainpath + 'labels/',
                                                   trainpath + 'images/',
                                                   transform=transforms)

    val_data = CustomImageDatasetObjectDetection(valpath + 'labels/',
                                                 valpath + 'images/',
                                                 transform=transforms)

    ##define dataloaders
    train_loader = DataLoader(train_data, batch_size=8, collate_fn=train_data.collate_fn,
                              shuffle=True)

    val_loader = DataLoader(val_data, batch_size=8, collate_fn=val_data.collate_fn,
                            shuffle=True)

    ##sanity check
    # Display image and label.
    data, labels, boxes = next(iter(val_loader))
    print('Data size: ', data.shape)
    print('Labels size: ', labels.shape)
    print('Boxes size: ', boxes.shape)
    img = data[0].squeeze()
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

    from vit_pytorch.nest import NesT
    nest = NesT(image_size = 224, patch_size = 4, dim = 96, heads = 3,
                num_hierarchies = 3,
                block_repeats = (2, 2, 8),
                num_classes = 1000)

    img = torch.randn(1, 3, 224, 224)

    pred = nest(img) # (1, 1000)
