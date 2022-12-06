import numpy as np
import matplotlib as mp
import csv
import cv2
from modules import preProcess
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

   
    #Read In Healthy Data
    hel_path = "/home/tdawg/Desktop/ML_Project/data/hel_filenames_labels.csv"
    with open(hel_path, newline='') as csvfile:
       hel_data = list(csv.reader(csvfile))
   
    #Read In Diseased Data
    dis_path = "/home/tdawg/Desktop/ML_Project/data/dis_filenames_labels.csv"
    with open(dis_path, newline='') as csvfile:
       dis_data = list(csv.reader(csvfile))
    

    #Creating Training Data Array-----------------------------------
    num_train_val_instances = 250
    train_val_data = []
    for i in range(num_train_val_instances):
        train_val_data.append(dis_data[i])
    for i in range(num_train_val_instances):
        train_val_data.append(hel_data[i])

    train_data_np = np.array(train_val_data)
    np.random.shuffle(train_data_np)

    #Creating Test Data Array ----------------------------------------
    num_test_instances = 50
    test_data = []
    for i in range(num_test_instances):
        test_data.append(dis_data[i+num_train_val_instances])
    for i in range(num_test_instances):
        test_data.append(hel_data[i+num_train_val_instances])
    
    #Converting to Np_array and shuffling
    #Converting to NP Array
    test_data_np = np.array(test_data)
    #Suffling Data
    np.random.shuffle(test_data_np)


    #Get Instance Will Look Something Like This 
    # fix_image(img, width, height, mode):   
    """
    Mode 1 - Just Resized
    Mode 2 - grayscale w Clahe 
    Mode 3 - Canny Edge Detection
    Mode 4 - Sobel Edge Detection 
    """

    # Set Number of K-Fold CV partitions
    K_fold_num = 5
    # Set sizes of train test and validation sets
    Train_Val_Size = len(train_data_np)
    Test_Size = len(test_data_np)
    Val_Size = 1/K_fold_num*Train_Val_Size
    Train_Size = (K_fold_num-1)/K_fold_num*Train_Val_Size

    # Set the image height and width
    imgSize = 200
    imgMode = 1

    for k in range(K_fold_num):
        
        # Create input and ouput holders for data
        Y_train = []
        X_train = []

        Y_val = []
        X_val = []

        Y_test = []
        X_test = []

        counter = 0

        # Parse Train and Validation Data
        for i in range(Train_Val_Size):
            img = cv2.imread(train_data_np[i][0]) #Iterate
            label = int(train_data_np[i][1])  #Iterate

            img = preProcess.fix_image(img, imgSize, imgSize, imgMode)
            img_array =  preProcess.get_array(img)

            # Partition Data for K-Fold CV
            if k == 0:
                if i <400:
                    Y_train.append(label)
                    X_train.append(img_array)
                else:
                    Y_val.append(label)
                    X_val.append(img_array)
            elif k == 1:
                if i > 299 and i < 400:
                    Y_val.append(label)
                    X_val.append(img_array)
                else:
                    Y_train.append(label)
                    X_train.append(img_array)
            elif k == 2:
                if i > 199 and i < 300:
                    Y_val.append(label)
                    X_val.append(img_array)
                else:
                    Y_train.append(label)
                    X_train.append(img_array)
            elif k == 3:
                if i > 99 and i < 200:
                    Y_val.append(label)
                    X_val.append(img_array)
                else:
                    Y_train.append(label)
                    X_train.append(img_array)
            elif k == 4:
                if i < 100:
                    Y_val.append(label)
                    X_val.append(img_array)
                else:
                    Y_train.append(label)
                    X_train.append(img_array)
            
        # Parse Test Data
        for i in range(len(test_data_np)):
            img = cv2.imread(test_data_np[i][0]) #Iterate
            label = int(test_data_np[i][1])  #Iterate

            img = preProcess.fix_image(img, imgSize, imgSize, imgMode)
            img_array =  preProcess.get_array(img)

            Y_test.append(label)
            X_test.append(img_array)

        PredTrain = np.zeros(Train_Size)
        PredVal = np.zeros(Val_Size)
        PredTest = np.zeros(Test_Size)

        # Train SKLearns MLP Classifier
        clf = MLPClassifier(solver='lbfgs', learning_rate='constant', learning_rate_init=5e-1,  max_iter = 50, activation = 'logistic', alpha=1e-5, hidden_layer_sizes=(10,1))
        clf.fit(X_train, Y_train)  

        # Print training results
        print('Train APR')
        PredTrain = clf.predict(X_train)
        print(accuracy_score(Y_train, PredTrain))
        print(precision_score(Y_train, PredTrain))
        print(recall_score(Y_train, PredTrain))

        # Print Validation Results
        print('Val APR')
        PredVal = clf.predict(X_val)
        print(accuracy_score(Y_val, PredVal))
        print(precision_score(Y_val, PredVal))
        print(recall_score(Y_val, PredVal))

        # Print Testing Results
        print('Test APR')
        PredTest = clf.predict(X_test)
        print(accuracy_score(Y_test, PredTest))
        print(precision_score(Y_test, PredTest))
        print(recall_score(Y_test, PredTest))
        del clf
        



 