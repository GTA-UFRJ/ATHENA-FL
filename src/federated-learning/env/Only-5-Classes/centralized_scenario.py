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
# usage: python centralized-scenario.py 


from sys import argv

from load_federated_data import *
from generate_neural_network import build_model

# scenario configuration
modelType = 1
clientID = 1
numClients = 1
basicNN = False
dataset_name = "CIFAR-10"
trPer = 0.8
numEpochs = 100

if len(argv) > 1:
    dataset_name = argv[1]

# Load the dataset
x_train, y_train, x_test, y_test = load_data_federated_IID(dataset_name, clientID, numClients, basicNN, modelType, trPer)

# Build neural network
model = build_model(basicNN,dataset_name)

# Fit the model
model.fit(x_train, y_train, validation_split=0.2, epochs=numEpochs,batch_size=128,steps_per_epoch=int(0.8*len(x_train)//128))

# Save the model for later evaluations
model.save('models/centralized_model_'+str(len(x_train))+'_training_samples_'+str(numEpochs)+'_epochs_'+dataset_name+'_dataset')

# Evaluate the final model
loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)

	



