import seaborn as sn
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from pickle import dump,load
from os import listdir
from time import time
from scipy.stats import pearsonr

def evaluate_execution_time(file_name,function):
    if function  == "list":
        initial_time = time()
        get_dataset_itens(file_name)
        final_time =  time()
    else:
        initial_time = time()
        load_dataset_to_dataframe(file_name)
        final_time =  time() 
    return final_time - initial_time

def load_dataset_to_dataframe(file_name):
    header_length = get_header_length(file_name)
    dataset = pd.read_csv(file_name,low_memory=False)
    dataset_identifiers = dataset.iloc[:,range(8)]
    dataset_features = dataset.iloc[:,range(8,header_length-2)]
    dataset_labels = dataset.iloc[:,[header_length-1]]
    return dataset_identifiers,dataset_features,dataset_labels

def get_header(file_name):
    with open(file_name,"r") as dataset:
        header = dataset.readline().split(',')
    return header

def get_header_length(file_name):
    return len(get_header(file_name))


def get_dataset_itens(file_name):
    dataset_identifiers = []
    dataset_features = []
    dataset_labels = []
    with open(file_name,"r") as dataset:
        header = dataset.readline()
        for line in dataset:
            line_list = line.split(',')
            dataset_identifiers.append(line_list[:8])
            dataset_features.append(line_list[8:-1])
            dataset_labels.append(line_list[-1])
    return [dataset_identifiers,dataset_features,dataset_labels]


def load_all_dataset_csvs_to_dataframe(path):
    names = []
    files = listdir(path)
    for name in files:
        names.append(path+name) 
    dataset_identifiers, dataset_features, dataset_labels = load_dataset_to_dataframe(names[0])
    for file_name in names[1:]:
        new_dataset_identifiers, new_dataset_features, new_dataset_labels = load_dataset_to_dataframe(file_name)
        dataset_identifiers = pd.concat([dataset_identifiers,new_dataset_identifiers])
        dataset_features = pd.concat([dataset_features,new_dataset_features])
        dataset_labels = pd.concat([dataset_labels,new_dataset_labels])
    return dataset_identifiers,dataset_features,dataset_labels
     

def load_all_dataset_csvs_to_list(path):
    names = []
    files = listdir(path)
    dataset = [[],[],[]]
    for name in files:
        names.append(path+name)
    for file_name in names:
        data = get_dataset_itens(file_name)
        dataset = [dataset[0]+data[0],dataset[1]+data[1],dataset[2]+data[2]] 

def verify_nan_features(dfDataset):
    print('Number of missing features: ', dfDataset.isna().sum().sum(),'\n\n')
    for column_name in dfDataset.keys():
        if dfDataset[column_name].isna().sum():
            print('Number of missing features in column %s:' % str(column_name), dfDataset[column_name].isna().sum())

def binarize_label_dataframe(labels):
    return labels.apply(lambda label: 1 if (label.item() == "BENIGN") else -1, axis=1)
    
def multiclass_label_dataframe(labels):
    labels_dictionary = get_dataframe_labels_dictionary(labels)
    return labels.apply(lambda label: labels_dictionary[label.item()], axis=1)

def get_dataframe_labels_dictionary(dataframe):
    unique_labels =  list(dataframe[" Label"].unique())
    labels_dictionary = {}
    for label,name in enumerate(unique_labels):
        labels_dictionary[name] = label
    return labels_dictionary


def get_binary_labels_list(labels):
    binary_labels = []
    for item in labels:
        if item == "BENIGN\n":
            binary_labels.append(-1)
        else:
            binary_labels.append(1)
    return binary_labels

def get_labels_dictionary(labels):
    unique_labels = list(set(labels))
    labels_dictionary = {}
    for label,name in enumerate(unique_labels):
        labels_dictionary[name] = label 
    return labels_dictionary

def get_multiclass_labels(labels):
    multiclass_labels = []
    labels_dictionary = get_labels_dictionary(labels)
    for item in labels:
        multiclass_labels.append(labels_dictionary[item])
    return multiclass_labels


def write_dataset(dataset,file_name):
    with open(file_name,"wb") as dataset_file:
        dump(dataset,dataset_file)

def read_dataset(file_name):
    with open(file_name,'rb') as dataset_file:
        dataset = load(dataset_file)
    return dataset

def print_column_names(file_name):
    with open(file_name,"r") as dataset:
        names = dataset.readline()
    names_list = names.split(',')
    for item in names_list:
        print(item)


def remove_high_correlation_features(corr_matrix,features):
    high_correlation_columns = {}
    for column_name in corr_matrix.keys():
        for index in range(len(corr_matrix[column_name])):
            if (corr_matrix[column_name][index] >= 0.8) and (corr_matrix.index[index] != column_name):
                if corr_matrix.index[index] in high_correlation_columns.keys():
                    high_correlation_columns[corr_matrix.index[index]].append(str(column_name))
                else:
                    high_correlation_columns[corr_matrix.index[index]] = [str(column_name)]
    
    # separe the features 
    features_to_not_delete = []
    features_to_delete = []

    # sort features according to the number of correlated features
    for key in sorted(high_correlation_columns, key=lambda key: len(high_correlation_columns[key]), reverse=True):
        if key not in features_to_delete:
            features_to_not_delete.append(key)
            for item in high_correlation_columns[key]:
                if item not in features_to_delete:
                    features_to_delete.append(item)
    
    # delete the features
    for name in features_to_delete:
        corr_matrix.drop(name, axis='columns', inplace=True)
        corr_matrix.drop(name, axis='index', inplace=True)
        features.drop(name, axis='columns', inplace=True)



def remove_nan_features(data,corr_matrix):
    columns_to_delete = []
    for column_name in corr_matrix.keys():
        if corr_matrix[column_name].isna().sum() == 77:
            columns_to_delete.append(str(column_name))

    for name in columns_to_delete:
        corr_matrix.drop(name, axis='columns', inplace=True)
        corr_matrix.drop(name, axis='index', inplace=True)
        data.drop(name, axis='columns', inplace=True)
  
    return columns_to_delete



if __name__ == "__main__":
    
    path = '../../../datasets/CICDataset/processed_data/all_features_dataframes/'
    VERBOSE = True
    
    if VERBOSE:
        print('Checking pre-process files')

    if 'features_1' in listdir(path):
        identifiers_1 = read_dataset(path+'identifiers_1')
        features_1 = read_dataset(path+'features_1')
        labels_1 = read_dataset(path+'labels_1')
        
        identifiers_2 = read_dataset(path+'identifiers_2')
        features_2 = read_dataset(path+'features_2')
        labels_2 = read_dataset(path+'labels_2')

    else:
        identifiers_1, features_1, labels_1 = load_all_dataset_csvs_to_dataframe('/root/ATHENA-FL/datasets/CICDataset/CSVs/01-12/')
        identifiers_2, features_2, labels_2 = load_all_dataset_csvs_to_dataframe('/root/ATHENA-FL/datasets/CICDataset/CSVs/03-11/')

    if VERBOSE:
        print('calcute the correlation between features')
    corr_matrix_1 = features_1.corr()
    corr_matrix_2 = features_2.corr()

    if VERBOSE:
        print('remove NaN features')
    
    remove_nan_features(features_1,corr_matrix_1)
    remove_nan_features(features_2,corr_matrix_2)

    if VERBOSE:
        print('remove the high correlated features')
    
    remove_high_correlation_features(corr_matrix_1,features_1)
    remove_high_correlation_features(corr_matrix_2,features_2)

    if VERBOSE:
        print('categorize labels')
    
    label_1_binary = binarize_label_dataframe(labels_1)
    label_2_binary = binarize_label_dataframe(labels_2)
    
    label_1_multiclass = multiclass_label_dataframe(labels_1)
    label_2_multiclass = multiclass_label_dataframe(labels_2)


    if VERBOSE:
        print('save the processed data')
    
    write_dataset(identifiers_1,path+'/../pre_processed_dataframes/identifiers_1')
    write_dataset(features_1,path+'/../pre_processed_dataframes/features_1')
    write_dataset(label_1_binary,path+'/../pre_processed_dataframes/label_1_binary')
    write_dataset(label_1_multiclass,path+'/../pre_processed_dataframes/label_1_multiclass')


    write_dataset(identifiers_2,path+'/../pre_processed_dataframes/identifiers_2')
    write_dataset(features_2,path+'/../pre_processed_dataframes/features_2')
    write_dataset(label_2_binary,path+'/../pre_processed_dataframes/label_2_binary')
    write_dataset(label_2_multiclass,path+'/../pre_processed_dataframes/label_2_multiclass')


