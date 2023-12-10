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
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.rpn import AnchorGenerator

from custom_dataset import CustomImageDatasetObjectDetection

from engine import train_one_epoch, evaluate

##define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    t0 = time.time()

    ##define paths
    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/numbers_mnist/vit_dset/"
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

    ##hyperparameters
    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.0005
    step_size = 3
    gamma = 0.1
    num_epochs = 10
    print_freq = 100
    batch_size = 144

    ##initialize datasets
    train_data = CustomImageDatasetObjectDetection(trainpath,
                                                   transforms=get_transform(train=True))

    val_data = CustomImageDatasetObjectDetection(valpath,
                                                 transforms=get_transform(train=False))

    ##define dataloaders
    trainloader = DataLoader(train_data, batch_size=batch_size,
                             collate_fn=train_data.collate_fn, shuffle=True)

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

    ##string model together
    # from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    ##define detection model and load pretrained weights
    # weights = "FasterRCNN_MobileNet_V3_Small_FPN_Weights.DEFAULT"
    # model = fasterrcnn_mobilenet_v3_large_fpn(backbone,
    #                                           weights=weights,
    #                                           box_roi_pool=roi_pooler).to(device)

    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    model = fasterrcnn_mobilenet_v3_large_320_fpn(backbone,
                                              weights="DEFAULT",
                                              box_roi_pool=roi_pooler).to(device)

    # from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    # model = ssdlite320_mobilenet_v3_large(backbone,
    #                                       weights="COCO_V1",
    #                                       box_roi_pool=roi_pooler).to(device)

    ##report number of parameters
    total_params = sum(param.numel() for param in model.parameters())
    print('\nNum Model Parameters: ', total_params)

    ##sanity check
    images, targets = next(iter(trainloader))
    images, targets = next(iter(valloader))

    print('Num Images: ', len(images))
    print('Num Targets: ', len(targets))
    print('Shape of Images ', images[0].shape)
    print('Shape of Targets ', targets[0]['labels'].shape)

    ##define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
    #                             weight_decay=weight_decay)
    optimizer = torch.optim.Adam(params, lr=lr)
    # optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
    #                             weight_decay=weight_decay)

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

    ##save model
    model_savepath = './vit_' + str(epoch) + '.pt'
    torch.save(model.state_dict(), model_savepath)

    ##load model
    model.load_state_dict(torch.load(model_savepath))

    ##evaluate model


    ##run eval on one image
    image = read_image("./test_imgs/mnist_noise_000010.png")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        ##RGBA -> RGB if is RGBA
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    ##convert image prior to torch tensor
    image = (255.0 * (x - x.min()) / (x.max() - x.min())).to(torch.uint8)
    image = image[:3, ...]

    ##get prediction labels
    pred_labels = [f"Clss: {label} Conf: {score:.3f}" for label,
                   score in zip(pred["labels"], pred["scores"])]

    ##remember that class zero is always background
    idx_to_str_clss_dict = {0: 'bg', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
                            7: '7', 8: '8', 9: '9', 10: '0'}
    print('Labels Key: ', idx_to_str_clss_dict)
    ##get predicted bboxes
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    ##plot image
    fig, axes = plt.subplots(1, 1)
    axes.imshow(output_image.permute(1, 2, 0))
    axes.set_title('Example of Inferenced Image')
    plt.savefig('output_image_' + 'num_epochs_' + str(epoch) + '.png')

    ##stop timer
    tf = time.time()
    print('Code took: ', np.round((tf-t0), 3))
    gc.collect()
