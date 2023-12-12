# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import os
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
from skimage import io, transform
from torchvision.io import read_image, ImageReadMode
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, tag, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        ##define paths and classes named after folders
        self.tag = tag
        self.root = root
        self.classes = sorted(os.listdir(self.root))

        ##get string name to numerical conversion
        self.string_to_num = {}
        for i, clss in enumerate(self.classes):
            self.string_to_num[clss] = i

        #store transforms
        self.transforms = transforms

        ##get all images fromfolder
        self._get_images_from_folder()

        print('\nDataset Initialized with Tag: ', self.tag)
        print('Unique number of classes: ', len(self.classes))
        print('Classes: ', self.classes)
        print('Dataset Size: ', len(self.df))
        print('\n')

    def _get_images_from_folder(self):
        obj_dict = {'class': [], 'path': []}
        for i, clss in tqdm(enumerate(self.classes), desc = "Getting Files from Disk."):
            for j, file in enumerate(sorted(os.listdir(self.root + '/' + clss))):
                obj_dict['class'].append(self.string_to_num[clss])
                obj_dict['path'].append(self.root + '/' + clss + '/' + file)
        self.df = pd.DataFrame(obj_dict)

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.df.iloc[idx, 1])
        # image = resize(image, (128, 128), anti_aliasing=True)

        ##transform data
        if self.transforms:
            image = Image.fromarray(image.astype('uint8'))
            image = self.transforms(image)

        # image = read_image(self.df.iloc[idx, 1], mode=ImageReadMode.RGB)
        label = self.df.iloc[idx, 0]
        return image, label

    def collate_fn(self, batch):
        return tuple(zip(*batch))

'''
##Example Use-Case
if __name__ == "__main__":

    basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/intel_dataset/"
    trainpath = basepath + 'seg_train/'
    valpath = basepath + 'seg_test/'
    train_data = CustomDataset(trainpath, 'train set')
    val_data = CustomDataset(valpath, 'valid set')

    ##define dataloaders
    trainloader = DataLoader(train_data, batch_size=32, collate_fn=train_data.collate_fn,
                              shuffle=True)

    valloader = DataLoader(val_data, batch_size=32, collate_fn=val_data.collate_fn,
                            shuffle=True)

    ##sanity check
    images, targets = next(iter(trainloader))
    images, targets = next(iter(valloader))
'''
