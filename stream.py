#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
from tqdm.auto import tqdm

# Run using python3 stream.py to use CIFAR dataset and default batch_size as 100
# Run using python3 stream.py -f <input_file> -b <batch_size> to use a custom file/dataset and batch size
# Run using python3 stream.py -e True to stream endlessly in a loop
parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context')
parser.add_argument('--file', '-f', help='File to stream', required=False,
                    type=str, default="cifar")    # path to file for streaming
parser.add_argument('--batch-size', '-b', help='Batch size',
                    required=False, type=int, default=100)  # default batch_size is 100
parser.add_argument('--endless', '-e', help='Enable endless stream',
                    required=False, type=bool, default=False)  # looping disabled by default
parser.add_argument('--split','-s', help="training or test split", required=False, type=str, default='train')

TCP_IP = "localhost"
TCP_PORT = 6100


class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.labels = []
        self.epoch = 0

    def data_generator(self,file,batch_size):
        batch = []
        with open(f"cifar/{file}","rb") as batch_file:
            batch_data = pickle.load(batch_file, encoding='bytes')
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])

        data = np.vstack(self.data)
        self.data = list(map(np.ndarray.tolist, data))
        for ix in range(0,(len(self.data)//batch_size)*batch_size,batch_size):
            # print(ix)
            image = self.data[ix:ix+batch_size]
            label = self.labels[ix:ix+batch_size]
            batch.append([image,label])
        
        self.data = self.data[ix+batch_size:]
        self.labels = self.labels[ix+batch_size:]
        # print(f"Remaining Images : {len(self.data)}")
        return batch

    # separate function to stream CIFAR batches since the format is different
    def sendCIFARBatchFileToSpark(self,tcp_connection, input_batch_file):
        # load the entire dataset
        pbar = tqdm(total=int((5e4//batch_size)+1)) if train_test_split=='train' else tqdm(total=int((1e4//batch_size)+1))
        data_received = 0
        for file in input_batch_file:
            batches = self.data_generator(file,batch_size)
            for ix,batch in enumerate(batches):
                image,labels = batch
                image = np.array(image)
                received_shape = image.shape
                image = list(map(np.ndarray.tolist, image))

                feature_size = len(image[0])

                payload = dict()
                for mini_batch_index in range(len(image)):
                    payload[mini_batch_index] = dict()
                    for feature_index in range(feature_size):  # iterate over features
                        payload[mini_batch_index][f'feature{feature_index}'] = image[mini_batch_index][feature_index]
                    payload[mini_batch_index]['label'] = labels[mini_batch_index]

                send_batch = (json.dumps(payload) + '\n').encode()
                try:
                    tcp_connection.send(send_batch)
                except BrokenPipeError:
                    print("Either batch size is too big for the dataset or the connection was closed")
                except Exception as error_message:
                    print(f"Exception thrown but was handled: {error_message}")
                data_received+=1
                pbar.update(1)
                pbar.set_description(f"epoch: {self.epoch} it: {data_received} | received : {received_shape} images")
                time.sleep(3)
        for batch in [[self.data,self.labels]]:
                image,labels = batch
                image = np.array(image)
                received_shape = image.shape
                image = list(map(np.ndarray.tolist, image))

                feature_size = len(image[0])

                payload = dict()
                for mini_batch_index in range(len(image)):
                    payload[mini_batch_index] = dict()
                    for feature_index in range(feature_size):  # iterate over features
                        payload[mini_batch_index][f'feature{feature_index}'] = image[mini_batch_index][feature_index]
                    payload[mini_batch_index]['label'] = labels[mini_batch_index]

                send_batch = (json.dumps(payload) + '\n').encode()
                try:
                    tcp_connection.send(send_batch)
                except BrokenPipeError:
                    print("Either batch size is too big for the dataset or the connection was closed")
                except Exception as error_message:
                    print(f"Exception thrown but was handled: {error_message}")
                data_received+=1
                pbar.update(1)
                pbar.set_description(f"epoch: {self.epoch} it: {data_received} | received : {received_shape} images")
                self.data = []
                self.labels = []
                time.sleep(3)    
        pbar.pos=0
        self.epoch+=1

    def connectTCP(self):   # connect to the TCP server -- there is no need to modify this function
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamCIFARDataset(self,tcp_connection, dataset_type='cifar'):
        CIFAR_BATCHES = [
            'data_batch_1',
            'data_batch_2',   # uncomment to stream the second training dataset
            'data_batch_3',   # uncomment to stream the third training dataset
            'data_batch_4',   # uncomment to stream the fourth training dataset
            'data_batch_5',    # uncomment to stream the fifth training dataset
            'test_batch'      # uncomment to stream the test dataset
        ]
        CIFAR_BATCHES = CIFAR_BATCHES[:-1] if train_test_split=='train' else [CIFAR_BATCHES[-1]]
        self.sendCIFARBatchFileToSpark(tcp_connection,CIFAR_BATCHES)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    input_file = args.file
    batch_size = args.batch_size
    endless = args.endless
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    # to stream a custom dataset, uncomment the elif block and create your own dataset streamer function (or modify the existing one)
    if input_file == "cifar":
        _function = dataset.streamCIFARDataset
    if endless:
        while True:
            _function(tcp_connection, input_file)
    else:
        _function(tcp_connection, input_file)

    tcp_connection.close()

# Setup your own dataset streamer by following the examples above.
# If you wish to stream a single newline delimited file, use streamFile()
# If you wish to stream a CSV file, use streamCSVFile()
# If you wish to stream any other type of file(JSON, XML, etc.), write an appropriate function to load and stream the file
