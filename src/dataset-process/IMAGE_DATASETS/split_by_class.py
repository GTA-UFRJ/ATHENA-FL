# Authors: Gustavo Franco Camilo and Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python split_by_class.py 


import tensorflow as tf
import pickle
import numpy as np

from skimage import transform
from augment_data import augment_dataset

def split_data(dataset_name):
    # classes in the dataset
    dictClassTrain = {}
    dictLabelTrain = {}
    dictClassTest = {}
    dictLabelTest = {}

    
    # Load dataset
    if dataset_name == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    elif dataset_name == "FMNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        #x_train = transform.resize(x_train, (len(x_train), 32, 32, 1))
        #x_test = transform.resize(x_test, (len(x_test), 32, 32, 1))
        
    elif dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #x_train = transform.resize(x_train, (len(x_train), 32, 32, 1))
        #x_test = transform.resize(x_test, (len(x_test), 32, 32, 1))

    x_train_list = x_train.tolist() 
    y_train_list = y_train.tolist() 
    x_test_list = x_test.tolist()
    y_test_list = y_test.tolist()
    
    # Initialize the dictionaries
    for index in np.unique(y_train):
        dictClassTrain['class'+str(index)+'Train'] = []

    for index in np.unique(y_test):
        dictClassTest['class'+str(index)+'Test'] = []
    
    for index in np.unique(y_test):
        dictLabelTest['class'+str(index)+'TestLabel'] = []

    for index in np.unique(y_train):
        dictLabelTrain['class'+str(index)+'TrainLabel'] = []

    index = 0
    for sample in y_train_list:
        if type(sample) == list:
            dictClassTrain['class'+str(sample[0])+'Train'].append(np.asarray(x_train_list[index]))
            dictLabelTrain['class'+str(sample[0])+'TrainLabel'].append(np.asarray(sample))
        else:
            dictClassTrain['class'+str(sample)+'Train'].append(np.asarray(x_train_list[index]))
            dictLabelTrain['class'+str(sample)+'TrainLabel'].append(np.asarray(sample))
        index += 1

    index = 0
    for sample in y_test_list:
        if type(sample) == list:
            dictClassTest['class'+str(sample[0])+'Test'].append(np.asarray(x_test_list[index]))
            dictLabelTest['class'+str(sample[0])+'TestLabel'].append(np.asarray(sample))
        else:
            dictClassTest['class'+str(sample)+'Test'].append(np.asarray(x_test_list[index]))
            dictLabelTest['class'+str(sample)+'TestLabel'].append(np.asarray(sample))
        index += 1
    
    # Augment the dataset
    for index in np.unique(y_train):
        dictClassTrain['class'+str(index)+'Train'], dictLabelTrain['class'+str(index)+'TrainLabel'] = augment_dataset(dictClassTrain['class'+str(index)+'Train'], dictLabelTrain['class'+str(index)+'TrainLabel'])
    
    for index in np.unique(y_test):
        dictClassTest['class'+str(index)+'Test'], dictLabelTest['class'+str(index)+'TestLabel'] =  augment_dataset(dictClassTest['class'+str(index)+'Test'], dictLabelTest['class'+str(index)+'TestLabel'])


    # Save the files
    for index in np.unique(y_train):
        pickle.dump(dictClassTrain['class'+str(index)+'Train'],open('class'+str(index)+'Train',"wb"))
    
    for index in np.unique(y_test):
        pickle.dump(dictClassTest['class'+str(index)+'Test'],open('class'+str(index)+'Test',"wb"))

    for index in np.unique(y_test):
        pickle.dump(dictLabelTest['class'+str(index)+'TestLabel'],open('class'+str(index)+'TestLabel',"wb"))

    for index in np.unique(y_train):
        pickle.dump(dictLabelTrain['class'+str(index)+'TrainLabel'],open('class'+str(index)+'TrainLabel',"wb"))


if __name__ == "__main__":
    split_data('CIFAR-10')
