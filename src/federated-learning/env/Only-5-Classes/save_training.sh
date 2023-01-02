#!/bin/bash
#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#

data=$1
serverPort=$2
epochs=$3
numClients=$4
ova=$5
scenario=$6


# usage of client.py: python3.9 client.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>


# print results on the screen for 5 clients and a robust model
#for i in $(seq 5)
#do
#	python3.9 client.py 1 8081 $i 0 5 & 
#done

# save results for a robust model
if [ $ova -eq 0 ];
then

	# initialize the server
	python3.9 server.py $epochs $serverPort &

	sleep 2
	
	# initialize the clients
	for i in $(seq $numClients)
	do
		python3.9 client.py 1 $serverPort $i 0 $numClients $data $scenario >> results/result-$data-complex-model-epochs-$epochs-clients-$numClients-client$i &
	done
fi	


# save results a simple model
if [ $ova -eq 1 ];
then
	# initialize the servers
	for j in $(seq 0 9)
	do
		python3.9 server.py $epochs "$(($serverPort+$j))" &
	done

	sleep 2

	# intialize the clients
	for i in $(seq $numClients)
	do
		for j in $(seq 0 9)
		do
			python3.9 client.py $j "$(($serverPort+$j))" $i 1 $numClients $data >> results/result-$data-simple-model-$j-epochs-$epochs-clients-$numClients-client$i &
		done
	done
fi

