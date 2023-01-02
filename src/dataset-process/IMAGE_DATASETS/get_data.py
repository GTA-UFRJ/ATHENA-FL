# Authors: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python get_data.py <dataset_name>

from sys import argv

from split_by_class import split_data


# check the dataset to download
if len(argv) > 1:
    dataset_name = argv[1]

else:
    dataset_name = "MNIST"


if dataset_name in ['CIFAR-10','FMNIST','MNIST']:
    split_data(dataset_name)

else:
    print("We do not implement yet the processing of this dataset.")
    



