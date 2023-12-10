# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import os, gc, time, sys
gc.collect()
sys.path.append('srcutils/')


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.models.detection.rpn import AnchorGenerator

from custom_dataset import CustomImageDatasetObjectDetection

from engine import train_one_epoch, evaluate

##define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    t0 = time.time()

    ##define paths
    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/numbers_mnist/vit_dset/"
    weightspath = "./runs/detect/train_vit/weights/vit_10.pt"
    valpath = basepath + "val/"


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

    ##hyperparameters
    batch_size = 1

    ##initialize dataset
    val_data = CustomImageDatasetObjectDetection(valpath,
                                                 transforms=get_transform(train=False))

    ##define dataloader
    valloader = DataLoader(val_data, batch_size=batch_size,
                           collate_fn=val_data.collate_fn, shuffle=True)

    ##define backbone
    from vit_pytorch.nest import NesT
    num_classes = 11
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

    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    model = fasterrcnn_mobilenet_v3_large_320_fpn(backbone,
                                              weights="DEFAULT",
                                              box_roi_pool=roi_pooler).to(device)

    ##report number of parameters
    total_params = sum(param.numel() for param in model.parameters())
    print('\nNum Model Parameters: ', total_params)

    ##sanity check
    images, targets = next(iter(valloader))

    print('Num Images: ', len(images))
    print('Num Targets: ', len(targets))
    print('Shape of Images ', images[0].shape)
    print('Shape of Targets ', targets[0]['labels'].shape)

    ##define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    ##define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                   gamma=0.1)

    ##load model
    model.load_state_dict(torch.load(weightspath))
    model = model.to(device)
    model.eval()

    ##iterate over test set
    eval_transform = get_transform(train=False)
    for i, data in tqdm(enumerate(valloader, 0)):
        inputs, labels = data
        # x = inputs[:3, ...].to(device)
        with torch.no_grad():
            predictions = model(inputs[0].unsqueeze(0).to(device))

    ##stop timer
    tf = time.time()
    print('Code took: ', np.round((tf-t0), 3))
    gc.collect()
