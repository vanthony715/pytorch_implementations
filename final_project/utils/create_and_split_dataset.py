# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

##library imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

if __name__ == "__main__":
    ##define paths
    # datapath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/numbers_mnist/"
    datapath = "../../data/numbers_mnist/"
    trainwrite = "../../../data/numbers_mnist/train/images/"
    valwrite =  "../../../data/numbers_mnist/val/images/"


    ##if path exists then remove then make a new path
    remove_path(trainwrite)
    remove_path(trainwrite.replace('images', 'annotations'))
    remove_path(valwrite)
    remove_path(valwrite.replace('images', 'annotations'))

    ##read in trainset
    train_df = pd.read_csv(datapath + 'mnist_train.csv')
    y_train = train_df['label'].values
    train_df = train_df.drop('label', axis=1, inplace=False)

    ##read in testset
    val_df = pd.read_csv(datapath + 'mnist_test.csv')
    y_val = val_df['label'].values
    val_df = val_df.drop('label', axis=1, inplace=False)

    ##show an example of large noisy image
    noise = np.random.normal(0, 50, 65536).reshape(256, -1)
    plt.imshow(noise)

    ##train set preproc hyperparametes
    gauss_sigma = 3
    num_train_imgs = 60000
    ##create train dataset
    train_dict = create_dataset(gauss_sigma, num_train_imgs, trainwrite, train_df, y_train, is_train=True)
    write_img_paths(trainwrite, is_train=True)

    ##val set preproc hyperparameters
    gauss_sigma = 3
    num_test_imgs = 9000

    ##create test dataset
    val_dict = create_dataset(gauss_sigma, num_test_imgs, valwrite, val_df, y_val, is_train=False)
    write_img_paths(valwrite, is_train=False)

    ##check bboxes make sense
    num_to_write = 200
    imgpath = '../../../data/numbers_mnist/val/images/'
    qawritepath = '../../../data/numbers_mnist/QA/images/'
    remove_path(qawritepath)
    for i, file in tqdm(enumerate(os.listdir(imgpath)), desc="Writing QA Files"):
        if i <= num_to_write:
            fullpath = imgpath + file
            check_bboxes(fullpath, savepath=qawritepath)

    ##check annotations for validation set
    check_annotations(valwrite.replace('images', 'annotations'))

    ##check annotations for validation set
    check_annotations(trainwrite.replace('images', 'annotations'))
