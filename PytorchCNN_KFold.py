import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import csv

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from modules import preProcess

import glob
from tqdm import tqdm


import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
SKLearnMLP.py
1. Import CSV (column 0 image path, column 1 class label) and Convert to Numpy Array
2. Suffle Numpy Array
3. Write Each Image to a Column in array


"""


#Read In Healthy Data
hel_path = "/home/tdawg/Desktop/ML_Project/data/hel_filenames_labels.csv"
with open(hel_path, newline='') as csvfile:
    hel_data = list(csv.reader(csvfile))

#Read In Diseased Data
dis_path = "/home/tdawg/Desktop/ML_Project/data/dis_filenames_labels.csv"
with open(dis_path, newline='') as csvfile:
    dis_data = list(csv.reader(csvfile))


#Creating Training Data Array-----------------------------------
num_train_instances = 250
train_data = []
for i in range(num_train_instances):
    train_data.append(dis_data[i])
for i in range(num_train_instances):
    train_data.append(hel_data[i])

# print(train_data)

#Converting to Np_array and shuffling
#Converting to NP Array
train_data_np = np.array(train_data)
#Suffling Data
np.random.shuffle(train_data_np)

#Creating Test Data Array ----------------------------------------
num_test_instances = 50
test_data = []
for i in range(num_test_instances):
    test_data.append(dis_data[i+num_train_instances])
for i in range(num_test_instances):
    test_data.append(hel_data[i+num_train_instances])

#Converting to Np_array and shuffling
#Converting to NP Array
test_data_np = np.array(test_data)
#Suffling Data
np.random.shuffle(test_data_np)

# Set Number of K-Fold CV partitions
K_fold_num = 5
# Set sizes of train test and validation sets
Train_Val_Size = len(train_data_np)
Test_Size = len(test_data_np)
Val_Size = 1/K_fold_num*Train_Val_Size
Train_Size = (K_fold_num-1)/K_fold_num*Train_Val_Size


for k in range(K_fold_num):

    Y_train = []
    X_train = []

    Y_valid = []
    X_valid = []

    Y_test = []
    X_test = []

    # Parse Train and Validation Data
    for i in range(Train_Val_Size):
        img_array = train_data_np[i][0] #Iterate
        label = int(train_data_np[i][1])  #Iterate

        if k == 0:
            if i <400:
                Y_train.append(label)
                X_train.append(img_array)
            else:
                Y_valid.append(label)
                X_valid.append(img_array)
        elif k == 1:
            if i > 299 and i < 400:
                Y_valid.append(label)
                X_valid.append(img_array)
            else:
                Y_train.append(label)
                X_train.append(img_array)
        elif k == 2:
            if i > 199 and i < 300:
                Y_valid.append(label)
                X_valid.append(img_array)
            else:
                Y_train.append(label)
                X_train.append(img_array)
        elif k == 3:
            if i > 99 and i < 200:
                Y_valid.append(label)
                X_valid.append(img_array)
            else:
                Y_train.append(label)
                X_train.append(img_array)
        elif k == 4:
            if i < 100:
                Y_valid.append(label)
                X_valid.append(img_array)
            else:
                Y_train.append(label)
                X_train.append(img_array)
        
    # Parse test data
    for i in range(len(test_data_np)):
        img_array = test_data_np[i][0] #Iterate
        label = int(test_data_np[i][1])  #Iterate

        Y_test.append(label)
        X_test.append(img_array)

    class LeafDataset(Dataset):
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_filepath = self.image_paths[idx]
            image = cv2.imread(image_filepath)
            image = preProcess.fix_image(image, 200,200,1)/255 - 0.5
            image = image.astype(np.float32)
            label = self.labels[idx]   
            return image, label


    train_dataset = LeafDataset(X_train,Y_train)
    valid_dataset = LeafDataset(X_valid, Y_valid)
    test_dataset = LeafDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)



    net = models.resnet50(pretrained=True)
    net = net.to(device=device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    counter = 1
    batch_size = 8
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs= inputs.permute(0, 3, 1, 2)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if counter == batch_size:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}')
                running_loss = 0.0
                counter = 1
            counter = counter + 1

    print('Finished Training')


    PATH = './test.pth'
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0

    # # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        train_guess = np.zeros(len(X_train))
        train_labels = np.zeros(len(X_train))
        count = 0
        for data in train_loader:
            images, labels = data
            images = images.permute(0, 3, 1, 2)
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            _, train_predictions = torch.max(outputs, 1)
            train_predictions = train_predictions.cpu()
            train_predictions = train_predictions.numpy()
            for i in range(len(train_predictions)):
                train_guess[count] = train_predictions[i]
                train_labels[count] = labels[i]
                count = count+1
        print("Train APR")
        print(accuracy_score(train_labels, train_guess))
        print(precision_score(train_labels, train_guess))
        print(recall_score(train_labels, train_guess))

        val_guess = np.zeros(len(X_valid))
        val_labels = np.zeros(len(X_valid))
        count = 0
        for data in valid_loader:
            images, labels = data
            images = images.permute(0, 3, 1, 2)
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            _, val_predictions = torch.max(outputs, 1)
            val_predictions = val_predictions.cpu()
            val_predictions = val_predictions.numpy()
            for i in range(len(val_predictions)):
                val_guess[count] = val_predictions[i]
                val_labels[count] = labels[i]
                count = count+1
        print("Validation APR")
        print(accuracy_score(val_labels, val_guess))
        print(precision_score(val_labels, val_guess))
        print(recall_score(val_labels, val_guess))
            
        test_guess = np.zeros(len(X_test))
        test_labels = np.zeros(len(X_test))
        count = 0
        for data in test_loader:
            images, labels = data
            images = images.permute(0, 3, 1, 2)
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            _, test_predictions = torch.max(outputs, 1)
            test_predictions = test_predictions.cpu()
            test_predictions = test_predictions.numpy()
            for i in range(len(test_predictions)):
                test_guess[count] = test_predictions[i]
                test_labels[count] = labels[i]
                count = count+1
        print("Test APR")
        print(accuracy_score(test_labels, test_guess))
        print(precision_score(test_labels, test_guess))
        print(recall_score(test_labels, test_guess))
        
