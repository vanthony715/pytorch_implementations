# -*- coding: utf-8 -*-
"""
@author: vanth
"""

import os, gc
import time
import numpy as np
gc.collect()

from ultralytics import YOLO

if __name__ == "__main__":
    t0 = time.time()

    ##define paths
    obj_yaml_path = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/deep_learning_dev_w_pytorch/final_project/obj.yaml"
    model_weights_path = 'runs/detect/train2/weights/best.pt'

    ## Load a model
    model = YOLO(model_weights_path)  # load a pretrained model for fine-tuning


    metrics = model.val()  # evaluate model performance on the validation set

    tf = time.time()
    print('Code Took: ', np.round((tf-t0), 4))
