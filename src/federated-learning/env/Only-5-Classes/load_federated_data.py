#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#

import numpy as np

from skimage import transform
from pickle import load
from sklearn.utils import shuffle
from ova_processing import binary_labels

def load_data_federated_2_classes(dataset_name,clientID,numClients,basicNN,modelType,trPer):
    X = np.array([],dtype=np.float32)
    Y = np.array([],dtype=np.float32)

    for i in range(clientID%5,(clientID%5)+2):
        X_train_current = np.array([],dtype=np.float32)
        Y_train_current = np.array([],dtype=np.float32)
        X_test_current = np.array([],dtype=np.float32)
        Y_test_current = np.array([],dtype=np.float32)

        X_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'Train','rb')), dtype=np.float32)
        
        X_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'Test','rb')), dtype=np.float32)

        Y_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'TrainLabel','rb')), dtype=np.float32)

        Y_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'TestLabel','rb')),dtype=np.float32)

        begin_slice_train = int(len(Y_train_current)/(numClients/5)*(clientID-1))
        end_slice_train = int(len(Y_train_current)/(numClients/5)*clientID)
        begin_slice_test = int(len(Y_test_current)/(numClients/5)*(clientID-1))
        end_slice_test = int(len(Y_test_current)/(numClients/5)*clientID)

        X_train_current = X_train_current[begin_slice_train:end_slice_train]
        Y_train_current = Y_train_current[begin_slice_train:end_slice_train]
        X_test_current = X_test_current[begin_slice_test:end_slice_test]
        Y_test_current = Y_test_current[begin_slice_test:end_slice_test]
   
        if len(X) == 0:
            X = np.concatenate((X_train_current,X_test_current))
            Y = np.concatenate((Y_train_current,Y_test_current))
        else:
            X = np.concatenate((X,X_train_current,X_test_current))
            Y = np.concatenate((Y,Y_train_current,Y_test_current))
   
    # normalize the data
    X /= 255 

    # reshape MNIST and FMNIST
    if dataset_name == "MNIST" or dataset_name == "FMNIST":
        X = transform.resize(X, (len(X), 32, 32, 1))

    # If it is a basic NN we train a One-versus-All models
    if basicNN:
        Y = binary_labels(Y,modelType)
    
    X, Y = shuffle(X, Y, random_state=47527)
    
    trSize = int(len(X)*trPer)
    
    return X[:trSize], Y[:trSize], X[trSize:], Y[trSize:]

def load_data_federated_5_classes(dataset_name,clientID,numClients,basicNN,modelType,trPer):
    X = np.array([],dtype=np.float32)
    Y = np.array([],dtype=np.float32)

    for i in range(clientID%2,clientID%2+5):
        X_train_current = np.array([],dtype=np.float32)
        Y_train_current = np.array([],dtype=np.float32)
        X_test_current = np.array([],dtype=np.float32)
        Y_test_current = np.array([],dtype=np.float32)

        X_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'Train','rb')), dtype=np.float32)
        
        X_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'Test','rb')), dtype=np.float32)

        Y_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'TrainLabel','rb')), dtype=np.float32)

        Y_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'TestLabel','rb')),dtype=np.float32)

        begin_slice_train = int(len(Y_train_current)/(numClients/2)*(clientID-1))
        end_slice_train = int(len(Y_train_current)/(numClients/2)*clientID)
        begin_slice_test = int(len(Y_test_current)/(numClients/2)*(clientID-1))
        end_slice_test = int(len(Y_test_current)/(numClients/2)*clientID)

        X_train_current = X_train_current[begin_slice_train:end_slice_train]
        Y_train_current = Y_train_current[begin_slice_train:end_slice_train]
        X_test_current = X_test_current[begin_slice_test:end_slice_test]
        Y_test_current = Y_test_current[begin_slice_test:end_slice_test]
   
        if len(X) == 0:
            X = np.concatenate((X_train_current,X_test_current))
            Y = np.concatenate((Y_train_current,Y_test_current))
        else:
            X = np.concatenate((X,X_train_current,X_test_current))
            Y = np.concatenate((Y,Y_train_current,Y_test_current))
   
    # normalize the data
    X /= 255 

    # reshape MNIST and FMNIST
    if dataset_name == "MNIST" or dataset_name == "FMNIST":
        X = transform.resize(X, (len(X), 32, 32, 1))

    # If it is a basic NN we train a One-versus-All models
    if basicNN:
        Y = binary_labels(Y,modelType)
    
    X, Y = shuffle(X, Y, random_state=47527)
    
    trSize = int(len(X)*trPer)
    
    return X[:trSize], Y[:trSize], X[trSize:], Y[trSize:]


def load_data_federated_IID(dataset_name,clientID,numClients,basicNN,modelType,trPer):
    X = np.array([],dtype=np.float32)
    Y = np.array([],dtype=np.float32)

    for i in range(0,10):
        X_train_current = np.array([],dtype=np.float32)
        Y_train_current = np.array([],dtype=np.float32)
        X_test_current = np.array([],dtype=np.float32)
        Y_test_current = np.array([],dtype=np.float32)

        X_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'Train','rb')), dtype=np.float32)
        
        X_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'Test','rb')), dtype=np.float32)

        Y_train_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/train/class'+str(i)+'TrainLabel','rb')), dtype=np.float32)


        Y_test_current = np.asarray(load(open('../../../../datasets/'+
            dataset_name+'/Non-IID-distribution/test/class'+str(i)+'TestLabel','rb')),dtype=np.float32)

        begin_slice_train = int(len(Y_train_current)/numClients*(clientID-1))
        end_slice_train = int(len(Y_train_current)/numClients*clientID)
        begin_slice_test = int(len(Y_test_current)/numClients*(clientID-1))
        end_slice_test = int(len(Y_test_current)/numClients*clientID)

        X_train_current = X_train_current[begin_slice_train:end_slice_train]
        Y_train_current = Y_train_current[begin_slice_train:end_slice_train]
        X_test_current = X_test_current[begin_slice_test:end_slice_test]
        Y_test_current = Y_test_current[begin_slice_test:end_slice_test]
   
        if len(X) == 0:
            X = np.concatenate((X_train_current,X_test_current))
            Y = np.concatenate((Y_train_current,Y_test_current))
        else:
            X = np.concatenate((X,X_train_current,X_test_current))
            Y = np.concatenate((Y,Y_train_current,Y_test_current))
   
    # normalize the data
    X /= 255 

    # reshape MNIST and FMNIST
    if dataset_name == "MNIST" or dataset_name == "FMNIST":
        X = transform.resize(X, (len(X), 32, 32, 1))

    # If it is a basic NN we train a One-versus-All models
    if basicNN:
        Y = binary_labels(Y,modelType)
    
    X, Y = shuffle(X, Y, random_state=47527)
    
    #print("clientID: ",clientID," Y: ",len(Y))
    
    trSize = int(len(X)*trPer)
    
    return X[:trSize], Y[:trSize], X[trSize:], Y[trSize:]
