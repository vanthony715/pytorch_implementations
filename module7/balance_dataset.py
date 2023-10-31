#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import os, shutil
import numpy as np
file_dict = {'folder': [], 'cnt': []}

# exception handler 
def handler(func, path, exc_info): 
    print("Inside handler") 
    print(exc_info) 

def get_directory_counts(path):
    ##get directory counts
    cdict = {'filepath': [], 'count': []}
    for _, directory in enumerate(os.listdir(path)):
        if os.path.isdir(path + directory + '/'):
            cdict['filepath'].append(path + directory + '/')
            cdict['count'].append(int(len(os.listdir(path + directory + '/'))/2))
    print('\nSample Directory Counts: \n', cdict['count'])
    print('\nMin Count: ', np.min(cdict['count']))
    return cdict

def remove_directories_with_too_many_sample_folders(data_dict):
    ##remove directories that are over the minimum value
    min_count = np.min(counts_dict['count'])
    for i, (filepath, counts) in enumerate(zip(data_dict['filepath'], data_dict['count'])):
        if counts >= min_count:
            remove_num = counts - min_count
            folders = [i for i in os.listdir(filepath) if '.avi' not in i]
            rand_files = np.random.choice(folders, remove_num)
            for _, rand_file in enumerate(rand_files):
                path_to_remove = os.path.join(filepath, rand_file)
                # path_to_remove = filepath + rand_file + '/' 
                shutil.rmtree(path_to_remove, onerror = handler)
               
            print(f"Removed: {len(rand_files)} files from: {filepath}")

def remove_directories_wo_adequate_samples_in_folder(path):
    for _, directory in enumerate(os.listdir(path)):
        for _, folder in enumerate(os.listdir(path + directory + '/')):
            if os.path.isdir(path + directory + '/' + folder + '/'):
                if os.path.isdir(path + directory + '/' + folder + '/'):
                    if len(os.listdir(path + directory + '/' + folder + '/')) != 64:
                        shutil.rmtree(path + directory + '/' + folder + '/')
    
if __name__ == '__main__':
    datapath = '/home/tony/deep_learning_dev_w_pytorch/data/UCF50/'
    
    ##get data counts
    counts_dict = get_directory_counts(datapath)
    
    ##remove directories without enough samples
    remove_directories_wo_adequate_samples_in_folder
    
    ##remove folders if there are too many samples
    remove_directories_with_too_many_sample_folders(counts_dict)

    ##recount
    get_directory_counts(datapath)
