# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""

import os, gc, time

import numpy as np
from utils.utils import *

gc.collect()

if __name__ == "__main__":
    t0 = time.time()

    c = np.zeros([3, 3])
    a = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    z = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]])

    for i in range(5):
        c = c + a + b
    c = c/np.max(c)
    c = c - np.mean(c, axis = 1)

    for i in range(5):
        c = c + z
    c = c/np.max(c)
    c = c - np.mean(c, axis = 1)

    for i in range(5):
        c = c + b
    c = c/np.max(c)
    c = c - np.mean(c, axis = 1)

    for i in range(5):
        c = c + a
    c = c/np.max(c)
    c = c - np.mean(c, axis = 1)


    gc.collect()
    tf = time.time()
    print('Total Runtime: ', np.round(tf - t0, 4))