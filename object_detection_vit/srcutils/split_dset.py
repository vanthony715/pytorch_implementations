# -*- coding: utf-8 -*-
"""
@author: vanthony715@gmail.com
"""
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Splitter:
    def __init__(self, basepath):
        self.basepath = basepath
        self.annot_path = self.basepath + 'annotations/'
        self.img_path = self.basepath + 'images/'
        self.annots = sorted(os.listdir(self.annot_path))
        self.imgs = sorted(os.listdir(self.img_path))

    def _make_folders(self):
        ##remove dataset folder if it exists
        if os.path.isdir(self.basepath + 'datasplit/'):
            shutil.rmtree(self.basepath + 'datasplit/')

        ##make train folders
        os.makedirs(self.basepath + 'datasplit/train/images/')
        os.makedirs(self.basepath + 'datasplit/train/annotations/')

        ##make valid folders
        os.makedirs(self.basepath + 'datasplit/val/images/')
        os.makedirs(self.basepath + 'datasplit/val/annotations/')

    def _copy_files(self, src, dst):
        shutil.copy(src, dst)

    def split_dataset(self):
        ##make folders
        self._make_folders()

        ##basically going to use sklearn to get filenames for test and train sets
        X_train, X_test, _, _ = train_test_split(self.annots,
                                                 self.annots, test_size=0.20,
                                                 random_state=42)

        ##write to train and test sets
        for i, annot in tqdm(enumerate(X_train), desc='Copying Train Set'):
            ##write images
            # print(self.img_path)
            source = self.img_path + annot.replace('xml', 'jpg')
            dest = self.basepath + 'datasplit/train/images/' + annot.replace('xml', 'jpg')
            self._copy_files(source, dest)

            ##write annots
            source = self.annot_path + annot
            dest = self.basepath + 'datasplit/train/annotations/' + annot
            self._copy_files(source, dest)

        ##write to train and test sets
        for i, annot in tqdm(enumerate(X_test), desc='Copying Test Set'):
            ##write images
            source = self.img_path + annot.replace('xml', 'jpg')
            dest = self.basepath + 'datasplit/val/images/' + annot.replace('xml', 'jpg')
            self._copy_files(source, dest)

            ##write annots
            source = self.annot_path + annot
            dest = self.basepath + 'datasplit/val/annotations/' + annot
            self._copy_files(source, dest)

# if __name__=="__main__":
#     basepath = "C:/Users/vanth/OneDrive/Desktop/JHUClasses/data/OxfordIIITDataset/"
#     split_obj = Splitter(basepath)
#     split_obj.split_dataset()
