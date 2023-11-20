# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import sys, time

sys.path.append('srcutils/')

import numpy as np
import matplotlib.pyplot as plt
from split_dset import Splitter
import srcutils.utils as utils
from srcutils.custom_dataset import OxfordIIITDataset

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate

##define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_classnames(path) -> dict:
    '''
    Parameters
    ----------
    path : TYPE
        DESCRIPTION. Path to list.txt

    Returns
    -------
    dict
        DESCRIPTION. A dictionary of labels in integer format

    '''
    with open(path, 'r') as f:
        lines = f.readlines()

    ##get classes
    class_samples = []
    for line in lines:
        name = line.split('_')[0]
        if name[0].isupper():
            clss = 'cat'
        else:
            clss = 'dog'
        class_samples.append(clss)
    classes = list(set(class_samples))

    cdict = dict()
    for i, clss in enumerate(classes):
        cdict[clss] = i

    return cdict

if __name__=="__main__":
    ##start timer
    t0 = time.time()

    ##define paths
    datapath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/OxfordIIITDataset/"
    classnamepath = datapath + 'list.txt'
    trainpath = datapath + 'datasplit/train/'
    valpath = datapath + 'datasplit/train/'

    ##retrieve classnames
    class_dict = get_classnames(classnamepath)

    ##create train/val split
    split_obj = Splitter(datapath)
    split_obj.split_dataset()

    ##define detection model and load pretrained weights
    weights = "DEFAULT"
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    ##put model on GPU
    model.to(device)
    num_classes = 2

    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    ##Define datasets and dataloaders
    trainset = OxfordIIITDataset(trainpath, get_transform(train=True), class_dict)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn)

    valset = OxfordIIITDataset(valpath, get_transform(train=False), class_dict)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn)

    ##sanity check
    images, targets = next(iter(trainloader))
    images, targets = next(iter(valloader))

    ##hyperparameters
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    step_size = 3
    gamma = 0.1
    num_epochs = 5

    ##define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    ##define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    ##DEBUG
    # print('\nModel: ', model)
    # print('\nOptimizer: ', optimizer)
    # print('\nTrainloader: ', trainloader)
    # print('\nDevice: ', device)

    ##use train_one_epoch in pycocotools to iterate over train epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valloader, device=device)

    ##save model
    model_savepath = './model_' + str('_epochs.pt')
    torch.save(model.state_dict(), model_savepath)

    ##run eval on one image
    image = read_image("../../data/PennFudanPed/FudanPed00036.png")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    ##convert image prior to torch tensor
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    ##get prediction labels
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]

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
