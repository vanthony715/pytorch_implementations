# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:14:25 2023

@author: vanth
"""
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
img = Image.open("../../data/OxfordIIITDataset/images/Abyssinian_65.jpg")
convert_tensor = transforms.ToTensor()
img = convert_tensor(img)
img = img.reshape(1, 3, 400, 267)
out = vits16(img)

out = out.detach().cpu().numpy()
out = out.reshape(24, 16)
plt.imshow(out)
