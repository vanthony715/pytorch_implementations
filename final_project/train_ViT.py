# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

##define libraries
import gc, time
gc.collect()

import sys
sys.path.append('srcutils/')

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.rpn import AnchorGenerator

from custom_dataset import CustomImageDatasetObjectDetection

from engine import train_one_epoch, evaluate

##define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":

    ##define paths
    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/numbers_mnist/"
    trainpath = basepath + 'train/'
    valpath = basepath + 'val/'


    ##define transforms
    def get_transform(train):
        transforms = []
        if train:
            ##only do this for training image
            transforms.append(T.RandomHorizontalFlip(0.5))

        ##following transforms for all images
        transforms.append(T.Resize((256, 256), antialias=True)) ##increase training time
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)


    ##initialize datasets
    train_data = CustomImageDatasetObjectDetection(trainpath + 'labels/',
                                                   trainpath + 'images/',
                                                   transform=get_transform(train=True))

    val_data = CustomImageDatasetObjectDetection(valpath + 'labels/',
                                                 valpath + 'images/',
                                                 transform=get_transform(train=False))

    ##define dataloaders
    trainloader = DataLoader(train_data, batch_size=8, collate_fn=train_data.collate_fn,
                              shuffle=True)

    valloader = DataLoader(val_data, batch_size=8, collate_fn=val_data.collate_fn,
                            shuffle=True)

    ##sanity check
    # Display image and label.
    images, targets, boxes = next(iter(valloader))
    print('Data size: ', images.shape)
    print('Labels size: ', targets.shape)
    print('Boxes size: ', boxes.shape)
    img = images[0].squeeze()
    label = targets[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

    from vit_pytorch.nest import NesT
    num_classes = 2
    backbone = NesT(image_size = 224, patch_size = 4, dim = 96, heads = 3,
                num_hierarchies = 3,
                block_repeats = (2, 2, 8),
                num_classes = num_classes)


    ##define anchor boxes
    anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    ##define feature map
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7,
                                                    sampling_ratio=2,)

    ##string model together
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    ##define detection model and load pretrained weights
    weights = "FasterRCNN_MobileNet_V3_Small_FPN_Weights.DEFAULT"
    model = fasterrcnn_mobilenet_v3_large_fpn(backbone,
                                              weights=weights,
                                              box_roi_pool=roi_pooler)

    ##sanity check
    images, targets, boxes = next(iter(trainloader))
    images, targets, boxes = next(iter(valloader))

    ##hyperparameters
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    step_size = 3
    gamma = 0.1
    num_epochs = 1
    print_freq = 5

    ##define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    ##define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                   gamma=gamma)

    ##DEBUG
    # print('\nModel: ', model)
    # print('\nOptimizer: ', optimizer)
    # print('\nTrainloader: ', trainloader)
    # print('\nDevice: ', device)

    ##use train_one_epoch in pycocotools to iterate over train epochs
    for epoch in range(num_epochs):

        ##train and print every X iterations
        train_one_epoch(model, optimizer, trainloader, device, epoch,
                        print_freq=print_freq)
        ##learning-rate update
        lr_scheduler.step()

        ##evaluate using validation set
        evaluate(model, valloader, device=device)

    ##put model on GPU
    ##test model
    # model.eval()
    # pred = model(img) # (1, 1000)
