#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context')
parser.add_argument('--file', '-f', help='File to stream', required=False,type=str, default="cifar")
parser.add_argument('--batch-size', '-b', help='Batch size',required=False, type=int, default=100)
parser.add_argument('--endless', '-e', help='Enable endless stream',required=False, type=bool, default=False)
parser.add_argument('--split','-s', help="training or test split", required=False, type=str, default='train')
parser.add_argument('--sleep','-t', help="streaming interval", required=False, type=int, default=3)

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
            image = self.data[ix:ix+batch_size]
            label = self.labels[ix:ix+batch_size]
            batch.append([image,label])
        
        self.data = self.data[ix+batch_size:]
        self.labels = self.labels[ix+batch_size:]
        return batch

    def sendCIFARBatchFileToSpark(self,tcp_connection, input_batch_file):
        
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
                    for feature_index in range(feature_size):
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
                time.sleep(sleep_time)
        for batch in [[self.data,self.labels]]:
                image,labels = batch
                image = np.array(image)
                received_shape = image.shape
                image = list(map(np.ndarray.tolist, image))

                feature_size = len(image[0])

                payload = dict()
                for mini_batch_index in range(len(image)):
                    payload[mini_batch_index] = dict()
                    for feature_index in range(feature_size):
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
                time.sleep(sleep_time)    
        pbar.pos=0
        self.epoch+=1

    def connectTCP(self):   
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
            'data_batch_2',   
            'data_batch_3',   
            'data_batch_4',   
            'data_batch_5',    
            'test_batch'
        ]
        CIFAR_BATCHES = CIFAR_BATCHES[:-1] if train_test_split=='train' else [CIFAR_BATCHES[-1]]
        self.sendCIFARBatchFileToSpark(tcp_connection,CIFAR_BATCHES)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    input_file = args.file
    batch_size = args.batch_size
    endless = args.endless
    sleep_time = args.sleep
    train_test_split = args.split
    dataset = Dataset()
    tcp_connection, _ = dataset.connectTCP()

    if input_file == "cifar":
        _function = dataset.streamCIFARDataset
    if endless:
        while True:
            _function(tcp_connection, input_file)
    else:
        _function(tcp_connection, input_file)

    tcp_connection.close()