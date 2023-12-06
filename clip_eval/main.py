# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import time
import gc
import numpy as np

import clip
import torch
from PIL import Image
from torch.utils.data import DataLoader
from utils.custom_dataset import CustomDataset
from torchvision.transforms import v2 as T

device = "cuda" if torch.cuda.is_available() else "cpu"

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

if __name__ == "__main__":
    t0 = time.time()
    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/intel_dataset/"
    trainpath = basepath + 'seg_train/'
    valpath = basepath + 'seg_test/'
    train_data = CustomDataset(trainpath, 'train set')
    val_data = CustomDataset(valpath, 'valid set')


    ##define dataloaders
    trainloader = DataLoader(train_data, batch_size=1, collate_fn=train_data.collate_fn,
                              shuffle=True)

    valloader = DataLoader(val_data, batch_size=1, collate_fn=val_data.collate_fn,
                            shuffle=True)

    ##sanity check
    images, targets = next(iter(trainloader))
    images, targets = next(iter(valloader))

    print('Num Images: ', len(images))
    print('Num Targets: ', len(targets))
    print('Shape of Image ', images[0].shape)

    ##inference
    model, preprocess = clip.load('ViT-B/32', device)

    ##put image on device
    image = Image.fromarray(images[0])
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.classes]).to(device)

    '''
    - why are the image features  of shape 1, 512

    - why are are the text features of shape 6, 512

    ##my understanding:
        the pretrained weights contain features that can be used to run similiarity
        of the network generated encodings with pre-existing head encodings.
    '''
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 10 most similar labels for the image
    # num_top_picks = 5
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # values, indices = similarity[0].topk(num_top_picks)

    # # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{train_data.classes[index]:>16s}: {100 * value.item():.2f}%")


    tf = time.time()
    print('Runtime: ', np.round((tf - t0), 4))
    gc.collect()
