This work studies common performance metrics differences between YOLOV8 and ViT, by training the two object detection 
models, inferencing on a test set, and reporting the results.

docs/ contains the proposal, a performance metrics Excel spreadsheet, a Power Point used to for brainstorming, and the project report.

literature/ contains most of the literature used for the report, to include published work from journals and online sources.

runs/ contains the trained weights of both YV8 and ViT, as well as performance plots, part of which were used in the report.

srcutils/ contains the custom dataset class used to load data for ViT, along with coco utils used Pytorch.

test_imgs/ contains a few images that are inferenced on as a sanity check.

utils/ contains a common utilities and also contains the code that was used to create the dataset for ViT, which is mostly the same as the code used to create the YV8 dataset, except that the coordinates are not normalized.

create_dataset.ipynb can be used to create the dataset that was used in this study for YV8.

obj.yaml is the file that tells YV8 where the data lives.

train_ViT.py trains the ViT network. Simply change the savepath and data paths and run.

train_yolov8.ipynb trains a YV8 network.

yolov8n.pt are the pretrained weights for a YV8 nano sized model.


