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

    gc.collect()
    tf = time.time()
    print('Total Runtime: ', np.round(tf - t0, 4))