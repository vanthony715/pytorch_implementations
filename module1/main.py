# -*- coding: utf-8 -*-
"""
@author: avasque1@jh.edu
"""
import os, gc, time
gc.collect()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show_image(img, plot=False):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if plot:
        plt.show()
    plt.savefig('./test_image_samples_s.jpg')

if __name__ == "__main__":
    t0 = time.time()
    ##define datapath
    os.chdir('C:/Users/vanth/OneDrive/Desktop/JHUClasses/deep_learning_dev_w_pytorch/module1/')
    basepath = './'
    datapath = basepath + '../../data/seg/'
    trainpath = datapath + 'seg_train/'
    testpath = datapath + 'seg_test/'
    classnames = os.listdir(testpath)

    ##hyperparameters
    epochs = 10
    opt = 'Adam'
    lr = 1e-3
    momentum = 0.9
    freeze_weights = False

    ##define transforms
    import torchvision.transforms as transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    test_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    ##create dataset from imagefoder
    train_dataset = torchvision.datasets.ImageFolder(root=trainpath,
                                                     transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=testpath,
                                                    transform=test_transforms)

    ##create dataloaders
    from torch.utils.data import DataLoader
    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    ##get resnet model with pretrained weights
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
    param_cnt = 0
    for i, param in enumerate(model.parameters()):
        param_cnt+=1
    print('\nModel Parameter Cnt: ', param_cnt)

    from torchsummary import summary
    print(summary(model, (3, 224, 224)))

    ##freeze layers
    nofreeze_layers = ['conv1.weight', 'bn1.weight', 'bn1.bias',
                       'fc.weight', 'fc.bias']
    if freeze_weights:
        for name, param in model.named_parameters():
            if name not in nofreeze_layers:
                param.requires_grad = False

    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('\n\nOptimizer: \n', optimizer)

    ##define loss
    criterion = torch.nn.CrossEntropyLoss()
    print('\n\nLoss function: \n', criterion)


    print('\nTraining Loop:')
    train_accuracy = []
    val_accuracy = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
                running_loss = 0.0

    print('\nFinished Training')

    ##save weights
    weights_name = './resnet18.pt'
    print('\nSaving Weights: ', )
    torch.save(model.state_dict(), weights_name)

    # =============================================================================
    #     Test Model
    # =============================================================================
    ##test the model
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    show_image(torchvision.utils.make_grid(images))

    class_map = {0: "buildings", 1:"forest", 2:"glacier", 3:"mountain", 4:"sea",
                  5:"street"}

    print('\nTesting Loop')
    pred_dict = {'truth':[], 'pred':[]}
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            ##book keeping for truth and predictions
            for label, prediction in zip(labels, predictions):
                pred_dict['truth'].append(label.item())
                pred_dict['pred'].append(prediction.item())

    ##turn data into string labels
    pred_dict_labels = {'truth': [], 'pred': []}
    for t, p in zip(pred_dict['truth'], pred_dict['pred']):
        ##check that there aren't any strange or large number predicted
        if p > len(classnames):
            p = 0
        else:
            pred_dict_labels['truth'].append(class_map[t])
            pred_dict_labels['pred'].append(class_map[p])

    from sklearn.metrics import accuracy_score
    print('\nAccuracy Score: ', accuracy_score(pred_dict_labels['truth'],
                                               pred_dict_labels['pred'],
                                               normalize=True))
    ##get confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred_dict_labels['truth'], pred_dict_labels['pred'], labels=list(class_map.values()))
    print('\nConfusion Matrix: \n', cm)

    ##plot confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, 1, figsize=(10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_map.values()))
    disp.plot()

    if freeze_weights:
        plt.savefig('./fixed_confusion_matrix.jpg')
    else:
        plt.savefig('./finetune_confusion_matrix.jpg')

    print('\nTime elapsed: ', np.round(time.time() - t0), 5)