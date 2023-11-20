# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:02:51 2023

@author: vanth
"""
import os
import torch
import xml.etree.ElementTree as ET

from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class OxfordIIITDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, class_dict):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        self.class_dict = class_dict

    def _parse_xml(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        classname = self.class_dict[list(list(root)[5])[0].text]
        x_min = int(list(list(list(root)[5])[4])[0].text)
        y_min = int(list(list(list(root)[5])[4])[1].text)
        x_max = int(list(list(list(root)[5])[4])[2].text)
        y_max = int(list(list(list(root)[5])[4])[3].text)
        return (classname, x_min, y_min, x_max, y_max)

    def __getitem__(self, idx):
        ## load image and annotations
        ##It is assumed that there is only one pet in each image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = read_image(img_path)

        ##tuple contains class and bbox info
        target_data = self._parse_xml(annot_path)
        xmin, ymin = target_data[1], target_data[2]
        xmax, ymax = target_data[3], target_data[4]
        bboxes = list([xmin, ymin, xmax, ymax])
        labels = list(target_data)
        image_id = idx


        ##Wrap sample and targets into torchvision tv_tensors
        img = tv_tensors.Image(img)

        ##target tensor
        target = {}
        target["area"] = torch.as_tensor([0])
        target["iscrowd"] = torch.as_tensor([False])
        target["labels"] = torch.as_tensor(labels)
        target["image_id"] = image_id
        target["boxes"] = tv_tensors.BoundingBoxes(bboxes, format="XYXY",
                                                   canvas_size=F.get_size(img))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
