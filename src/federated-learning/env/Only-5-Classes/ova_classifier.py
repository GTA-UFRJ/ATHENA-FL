# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python ova-classifier.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>

import flwr as fl
import tensorflow as tf
import numpy as np

from pickle import load, dump
from sys import argv
from sklearn.metrics import top_k_accuracy_score

# client configuration
serverPort = '8080'
modelType = 1
clientID = 1
numClients = 10
basicNN = True

model_1 = tf.keras.models.load_model('models/model_class_0_simple_True')
model_2 = tf.keras.models.load_model('models/model_class_1_simple_True')
model_3 = tf.keras.models.load_model('models/model_class_2_simple_True')
model_4 = tf.keras.models.load_model('models/model_class_3_simple_True')
model_5 = tf.keras.models.load_model('models/model_class_4_simple_True')


# first class
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')),dtype=np.float32)

x_test1 = x_test1[int(len(x_test1)/numClients*(clientID-1)):int(len(x_test1)/numClients*clientID)]
y_test1 = y_test1[int(len(y_test1)/numClients*(clientID-1)):int(len(y_test1)/numClients*clientID)]


# second class
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')),dtype=np.float32)


x_test2 = x_test2[int(len(x_test2)/numClients*(clientID-1)):int(len(x_test2)/numClients*clientID)]
y_test2 = y_test2[int(len(y_test2)/numClients*(clientID-1)):int(len(y_test2)/numClients*clientID)]



# third class

x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')),dtype=np.float32)
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')),dtype=np.float32)

x_test3 = x_test3[int(len(x_test3)/numClients*(clientID-1)):int(len(x_test3)/numClients*clientID)]
y_test3 = y_test3[int(len(y_test3)/numClients*(clientID-1)):int(len(y_test3)/numClients*clientID)]


# fourth class

x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')),dtype=np.float32)
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')),dtype=np.float32)

x_test4 = x_test4[int(len(x_test4)/numClients*(clientID-1)):int(len(x_test4)/numClients*clientID)]
y_test4 = y_test4[int(len(y_test4)/numClients*(clientID-1)):int(len(y_test4)/numClients*clientID)]
    
# fifth class

x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')),dtype=np.float32)
y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')),dtype=np.float32)


x_test5 = x_test5[int(len(x_test5)/numClients*(clientID-1)):int(len(x_test5)/numClients*clientID)]
y_test5 = y_test5[int(len(y_test5)/numClients*(clientID-1)):int(len(y_test5)/numClients*clientID)]

x_test = np.concatenate((x_test1,x_test2,x_test3,x_test4,x_test5))
y_test = np.concatenate((y_test1,y_test2,y_test3,y_test4,y_test5))


top_k_accuracy_score(y_test,tf.reshape(load(open('lista_predicao','rb')),[len(y_test),1]),k=5)

result = []

for sample in x_test:
    predicted = [model_1.predict(tf.reshape(sample,[1,32,32,3])),
            model_2.predict(tf.reshape(sample,[1,32,32,3])),
            model_3.predict(tf.reshape(sample,[1,32,32,3])),
            model_4.predict(tf.reshape(sample,[1,32,32,3])),
            model_5.predict(tf.reshape(sample,[1,32,32,3]))]
    tmp = max(predicted)
    index = predicted.index(tmp)
    result.append(index)


