import numpy as np
import pyspark
import json

from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType,StringType,BinaryType
from pyspark.ml.linalg import VectorUDT, DenseVector

class TrainingConfig:
    max_epochs = 100
    learning_rate = 3e-4
    batch_size = 128
    ckpt_path = "./checkpoints/"

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
    def __init__(self, model, split:str, training_config:TrainingConfig, spark_config:SparkConfig, transforms) -> None:
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

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        if not rdd.isEmpty():
            schema = StructType([StructField("image",VectorUDT(),True),StructField("label",IntegerType(),True)])
            df = self.sqlContext.createDataFrame(rdd, schema)
            df.show()
            df.printSchema()
        print("Total Batch Size of RDD Received :",len(rdd.collect()))
        print("---------------------------------------")



# class DataLoader:
#     def __init__(self, sparkContext:SparkContext, sparkStreamingContext: StreamingContext, sqlContext: SQLContext,sparkConf: SparkConfig, transforms) -> None:
#         self.sc = sparkContext
#         self.ssc = sparkStreamingContext
#         self.sparkConf = sparkConf
#         self.sql_context = sqlContext
#         self.stream = self.ssc.socketTextStream(self.sparkConf.stream_host,self.sparkConf.port)
#         self.transforms = transforms

#     def parse_stream(self)->DStream:
#         json_stream = self.stream.map(lambda line: json.loads(line))        
#         json_stream_exploded = json_stream.flatMap(lambda x: x.values())
#         json_stream_exploded = json_stream_exploded.map(lambda x : list(x.values()))
#         pixels = json_stream_exploded.map(lambda x: [np.array(x[:-1]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),x[-1]])
#         pixels = DataLoader.preprocess(pixels,self.transforms)
#         return pixels

#     @staticmethod
#     def preprocess(stream: DStream, transforms) -> DStream:
#         stream = stream.map(lambda x: [transforms.transform(x[0]).reshape(32,32,3).reshape(-1).tolist(),x[1]])
#         stream = stream.map(lambda x: [DenseVector(x[0]),x[1]])
#         return stream