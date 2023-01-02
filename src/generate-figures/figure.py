import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std, arange
from math import sqrt
from os import listdir

# figureType = 1 -> print loss
# figureType = otherwise -> print accuracy

# language = 1 -> print in portuguese-br
# language = otherwise -> print in english

def plot_image(figureType=0,language=1,numFiles=10,epochs=100,dataset_name="MNIST",complex_model=True):

    
    file_names = {}
    result_files = {}
    file_lines = {}
    
    results_path = '../federated-learning/env/Only-5-Classes/results/'
    
    # numFiles variable maybe deprecated
    total_files = len(listdir(results_path))

    if complex_model:
        #for i in range(1,numFiles+1):
        #   file_names["filename"+str(i)] = results_path+'result-'+dataset_name+'-complex-model-epochs-'+str(epochs)+'-clients-'+str(numFiles)+'-client'+str(i)

        # need to test this new loop
        for name, index in enumerate(listdir(results_path)):
            file_names["filename"+str(index+1)] = results_path+name
            
        for i in range(1,total_files+1):
            result_files["result"+str(i)] = open(file_names['filename'+str(i)], 'r')


        for i in range(1,total_files+1):
            file_lines["Lines"+str(i)] = result_files['result'+str(i)].readlines()
    
        
        for i in range(1,total_files+1):
            result_files['result'+str(i)].close()
    else:
        
        for index, name in enumerate(listdir(results_path)):
            file_names["filename"+str(index+1)] = results_path+name

        for i in range(1,total_files+1):
            result_files["result"+str(i)] = open(file_names['filename'+str(i)], 'r')


        for i in range(1,total_files+1):
            file_lines["Lines"+str(i)] = result_files['result'+str(i)].readlines()
    
        
        for i in range(1,total_files+1):
            result_files['result'+str(i)].close()
        

    accuracies = []
    ac = []
    

    for i in range(total_files):
        accuracies.append([])
        ac.append([])

    if figureType == 1:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[1][1:].split(' ')[0]))        
    else:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[2].split(' ')[1]))
       
    for i in range(total_files):
        ac[i] = accuracies[i][:75]

    ac = [ele for ele in ac if ele != []]
        
#    x1Mean = mean(accuracies,axis=0);
#    x1Interval = std(accuracies,axis=0)*1.96/sqrt(10);
    x1Mean = mean(ac,axis=0);
    x1Interval = std(ac,axis=0)*1.96/sqrt(10);
    x = arange(len(x1Mean))


    if language == 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)    
   #     plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
   #     plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Perda no Teste', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

    elif language == 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Acurácia', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

        
    elif language != 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Test Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()


    elif language != 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

if __name__ == "__main__":
    plot_image(0,1)



