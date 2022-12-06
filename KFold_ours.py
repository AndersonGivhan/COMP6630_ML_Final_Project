import numpy as np
import matplotlib as mp
import csv
import cv2
from modules import preProcess
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score




class StructType:
    # Create Struct Type class to hold all layer information
    def __init__(self):
        # Hold weight matrix for the layer
        self.w = np.array(0)
        # Hold output of the activation function
        self.out = np.array(0)
        # Hold error term at node for back propagation
        self.delta = np.array(0)


class MLP:
    # Class contains all the functionality to train and run a Multi Layer Perceptron Neural Network
    def __init__(self, num_input_features, num_layers, num_nodes, learning_rate):
        # Number of hidden layers in the MLP. Must be an integer > 0
        self.numLayers = num_layers
        # Number of nodes in each layer. Must be an array of length numLayers. Exception: Needs at minimum 2 elements.
        # If there is only one layer, make the second number 0. It won't get used, but it is needed for indexing.
        self.numNodes = num_nodes
        # Learning rate of network. Hyperparamter that effects how much a given error gradient is allowed to change the
        # weights.
        self.lr = learning_rate
        # Number of outputs. This is a binary classification problem.
        self.numOut = 1
        # Number of input features. The length of the input array.
        self.numIn = num_input_features
        # Holders for class input and label to be provided in training.
        self.input = np.zeros(num_input_features)
        self.label = 0

        # Initialize the layers of the network using the StructType() class
        self.layer = [StructType() for i in range(1+num_layers)]

        # Set matrix to random values of correct size for inputs and outputs.
        for i in range(1+num_layers):
            if i == 0:
                self.layer[i].w = np.random.randn(num_input_features, num_nodes[i])/50000 #50000
                self.layer[i].out = np.zeros(num_nodes[i])
                self.layer[i].delta = np.zeros(num_nodes[i])
            elif i == num_layers:
                self.layer[i].w = np.random.randn(num_nodes[i-1], 1)/100 #100
                self.layer[i].out = np.zeros(1)
                self.layer[i].delta = np.zeros(1)
            else:
                self.layer[i].w = np.random.randn(num_nodes[i-1], num_nodes[i])/100
                self.layer[i].out = np.zeros(num_nodes[i])
                self.layer[i].delta = np.zeros(num_nodes[i])

    # Function that takes numpy arrays of states and weights and provides the sigmoid of the dot product.
    def sigmoid(self, states, weights):
        z = np.dot(states, weights)
        x = 1/(1 + np.exp(-z))
        return x

    # Function that propagates the network forward a layer and records the outputs at the nodes. Takes in a numpy array
    # of inputs and an integer that determines what layer to store the output in.
    def forward_prop(self, input_vec, layer_idx):
        self.layer[layer_idx].out = self.sigmoid(input_vec, self.layer[layer_idx].w)

    # Function that calculates the error at a node given the error at the nodes in front of it and the weights
    # connecting those nodes
    def get_delta(self, layer_idx):
        if layer_idx == self.numLayers:
            self.layer[layer_idx].delta = (self.label - self.layer[layer_idx].out) * self.layer[layer_idx].out*(1 - self.layer[layer_idx].out)
        else:
            x = self.layer[layer_idx].out
            for i in range(len(x)):
                self.layer[layer_idx].delta[i] = x[i]*(1 - x[i])*np.dot(self.layer[layer_idx+1].w[i], self.layer[layer_idx+1].delta)


    # The weights of every layer are corrected using the errors at the forward nodes and the inputs into the node.
    def update_weights(self, layer_idx):
        if layer_idx == 0:
            self.layer[layer_idx].w += self.lr * np.outer(self.input, self.layer[layer_idx].delta)
        else:
            self.layer[layer_idx].w += self.lr * np.outer(self.layer[layer_idx-1].out, self.layer[layer_idx].delta)

    # Function that trains the network one input vector at a time.
    def train_nn(self, input_vec, label):
        self.input = input_vec
        self.input = self.input/256 #Normalize RGB data to 0-1 range
        self.label = label

        # Forward Propagation
        for i in range(self.numLayers+1):
            if i == 0:
                self.forward_prop(self.input, i)
            else:
                self.forward_prop(self.layer[i-1].out, i)
        
        # Back Propagation
        for i in range(self.numLayers, -1, -1):
            self.get_delta(i)

        for i in range(self.numLayers + 1):
            self.update_weights(i)

    # Function that calculates the output of the trained network.
    def predict_nn(self, input_vec):
        self.input = input_vec/256 #Normalize RGB data to 0-1 range

        # Forward Propagate the input
        for i in range(self.numLayers + 1):
            if i == 0:
                self.forward_prop(self.input, i)
            else:
                self.forward_prop(self.layer[i-1].out, i)

        x = self.layer[self.numLayers].out

        # Descsion Boundary of the output
        if x > 0.5:
            y = 1
        else:
            y = 0
        
        return y

    # Function returns the square error loss value for the last prediction
    def getLoss(self, label):
        return (label - self.layer[self.numLayers].out)**2




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

    #Converting to Np_array and shuffling
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
    test_data_np = np.array(test_data)
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
    imgMode = 4

    # Set inputs to the ANN MLP
    input_size = 3*imgSize*imgSize
    num_nodes = np.array([100, 0])
    num_layers = 1
    lr = 1e-2

    # Set number of training Epochs
    num_episodes = 20

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

        # Create Instance of Our MLP
        nn = MLP(input_size, num_layers, num_nodes, lr)

        # Preallocate arrays for prediction values
        PredTrain = np.zeros(Train_Size)
        PredVal = np.zeros(Val_Size)
        PredTest = np.zeros(Test_Size)

        # Train the Network
        for i in range(num_episodes):
            print('Num Episodes: ', i)
            for j in range(Train_Size):
                nn.train_nn(X_train[j], Y_train[j])

        # Predict Train Outputs
        for j in range(Train_Size):
            PredTrain[j] = nn.predict_nn(X_train[j])

        # Print Train Results
        print('Train APR')
        print(accuracy_score(Y_train, PredTrain))
        print(precision_score(Y_train, PredTrain))
        print(recall_score(Y_train, PredTrain))

        # Predict Validation Outputs
        for j in range(Val_Size):
            PredVal[j] = nn.predict_nn(X_val[j])

        # Print Validation Results
        print('Val APR')
        print(accuracy_score(Y_val, PredVal))
        print(precision_score(Y_val, PredVal))
        print(recall_score(Y_val, PredVal))

        # Predict Test Outputs
        for j in range(Test_Size):
            PredTest[j] = nn.predict_nn(X_test[j])

        # Print Test Results
        print('Test APR')
        print(accuracy_score(Y_test, PredTest))
        print(precision_score(Y_test, PredTest))
        print(recall_score(Y_test, PredTest))
            
        



 