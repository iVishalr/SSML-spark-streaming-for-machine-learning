import numpy as np
import pyspark
import json
import matplotlib.pyplot as plt
from pyspark import conf
import os
import pickle

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType,StringType,BinaryType
from pyspark.ml.linalg import VectorUDT, DenseVector
from sklearn.neural_network import MLPClassifier

from models.MLP import MLP

class TrainingConfig:
    num_samples = 5e4
    max_epochs = 100
    learning_rate = 3e-4
    batch_size = 128
    alpha = 5e-4
    ckpt_interval = 1
    ckpt_interval_batch = 195
    ckpt_dir = "./checkpoints/"
    model_name = "MLPvtest"
    verbose = True

    def __init__(self, **kwargs) -> None:
        for key,value in kwargs.items():
            setattr(self,key,value)

class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 3

    def __init__(self, **kwargs) -> None:
        for key,value in kwargs.items():
            setattr(self,key,value)

from dataloader import DataLoader

class Trainer:
    def __init__(self, model : MLP, split:str, training_config:TrainingConfig, spark_config:SparkConfig, transforms) -> None:
        self.model = model
        self.split = split
        self.configs = training_config
        self.sparkConf = spark_config
        self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc,self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc,self.ssc,self.sqlContext,self.sparkConf,self.transforms)
        
        self.accuracy = []
        self.smooth_accuracy = []
        self.loss = []
        self.smooth_loss = []
        self.precision = []
        self.smooth_precision = []
        self.recall = []
        self.smooth_recall = []
        self.f1 = []
        self.smooth_f1 = []
        self.epoch = 0
        self.batch_count = 0

    def save_checkpoint(self, message):

        path = os.path.join(self.configs.ckpt_dir,self.configs.model_name)
        print(f"Saving Model under {path}/...{message}")
        if not os.path.exists(self.configs.ckpt_dir):
            os.mkdir(self.configs.ckpt_dir)
        
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        np.save(f"{path}/accuracy-{message}.npy",self.accuracy)
        np.save(f"{path}/loss-{message}.npy",self.loss)
        np.save(f"{path}/precision-{message}.npy",self.precision)
        np.save(f"{path}/recall-{message}.npy",self.recall)
        np.save(f"{path}/f1-{message}.npy",self.f1)

        np.save(f"{path}/smooth_accuracy-{message}.npy",self.smooth_accuracy)
        np.save(f"{path}/smooth_loss-{message}.npy",self.smooth_loss)
        np.save(f"{path}/smooth_precision-{message}.npy",self.smooth_precision)
        np.save(f"{path}/smooth_recall-{message}.npy",self.smooth_recall)
        np.save(f"{path}/smooth_f1-{message}.npy",self.smooth_f1)

        self.model.model = self.raw_model
        with open(f"{path}/model-{message}.pkl", 'wb') as f:
            pickle.dump(self.model,f)

        with open(f"{path}/model-raw-{message}.pkl", 'wb') as f:
            pickle.dump(self.raw_model,f)

    def load_checkpoint(self, message):
        print("Loading Model ...")
        path = os.path.join(self.configs.ckpt_dir,self.configs.model_name)
        self.accuracy = np.load(f"{path}/accuracy-{message}.npy")
        self.loss = np.load(f"{path}/loss-{message}.npy")
        self.precision = np.load(f"{path}/precision-{message}.npy")
        self.recall = np.load(f"{path}/recall-{message}.npy")
        self.f1 = np.load(f"{path}/f1-{message}.npy")

        self.smooth_accuracy = np.load(f"{path}/smooth_accuracy-{message}.npy")
        self.smooth_loss = np.load(f"{path}/smooth_loss-{message}.npy")
        self.smooth_precision = np.load(f"{path}/smooth_precision-{message}.npy")
        self.smooth_recall = np.load(f"{path}/smooth_recall-{message}.npy")
        self.smooth_f1 = np.load(f"{path}/smooth_f1-{message}.npy")

        with open(f"{path}/model-raw-{message}.pkl", 'rb') as f:
            self.raw_model = pickle.load(f)

        with open(f"{path}/model-{message}.pkl", 'rb') as f:
            self.model = pickle.load(f)

        self.model.model = self.raw_model
        print("Model Loaded.")

    def plot(self, timestamp, df: pyspark.RDD) -> None:
        for i,ele in enumerate(df.collect()):
            image = ele[0].astype(np.uint8)
            image = image.reshape(32,32,3)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.savefig(f"images/image{i}.png")

    def configure_model(self):
        return self.model.configure_model(self.configs)

    def train(self):
        stream = self.dataloader.parse_stream()
        self.raw_model = self.configure_model()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            self.batch_count += 1
            schema = StructType([StructField("image",VectorUDT(),True),StructField("label",IntegerType(),True)])
            df = self.sqlContext.createDataFrame(rdd, schema)

            predictions, accuracy, loss, precision, recall, f1 = self.model.train(df,self.raw_model)

            if self.configs.verbose:
                print(f"Predictions = {predictions}")
                print(f"Accuracy = {accuracy}")
                print(f"Loss = {loss}")
                print(f"Precision = {precision}")
                print(f"Recall = {recall}")
                print(f"F1 Score = {f1}")

            self.accuracy.append(accuracy)
            self.loss.append(loss)
            self.precision.append(precision)
            self.recall.append(recall)
            self.f1.append(f1)

            self.smooth_accuracy.append(np.mean(self.accuracy))
            self.smooth_loss.append(np.mean(self.loss))
            self.smooth_precision.append(np.mean(self.precision))
            self.smooth_recall.append(np.mean(self.recall))
            self.smooth_f1.append(np.mean(self.f1))

            if self.split is 'train':
                if self.batch_count!=0 and self.batch_count%(self.configs.num_samples//self.configs.batch_size) == 0:
                    self.epoch+=1

                if (isinstance(self.configs.ckpt_interval, int) and self.epoch!=0 and self.batch_count==(self.configs.num_samples//self.configs.batch_size) and self.epoch%self.configs.ckpt_interval == 0):
                    self.save_checkpoint(f"epoch-{self.epoch}")
                    self.batch_count = 0
                elif self.configs.ckpt_interval_batch is not None and self.batch_count!=0 and self.batch_count%self.configs.ckpt_interval_batch == 0:
                    self.save_checkpoint(f"epoch-{self.epoch}-batch-{self.batch_count}")
        if self.split is 'train':
            print(f"epoch: {self.epoch} | batch: {self.batch_count}")
        print("Total Batch Size of RDD Received :",len(rdd.collect()))
        print("---------------------------------------")         