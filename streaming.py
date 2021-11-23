"""
-----------------------------------

DEPRECATED

-----------------------------------
"""


from array import ArrayType
from ctypes import Array
import pyspark
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.dataframe import DataFrame
from pyspark.streaming import StreamingContext
from pyspark.sql.types import IntegerType, StructField, StructType,StringType,BinaryType
from pyspark.sql import SQLContext
from sparkdl.image import imageIO
from sparkdl import DeepImageFeaturizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from transforms import Transforms, Normalize, RandomHorizontalFlip,RandomVerticalFlip, Resize, ColorShift
from models.ANN import ANN
from models.MLP import MLP
from models.SVM import SVM
from sklearn.neural_network import MLPClassifier
import json
import numpy as np
import matplotlib.pyplot as plt

import os
os.system("taskset -p 0xff %d" % os.getpid())
os.environ["OMP_NUM_THREADS"] = "4"
def test():
    def plot(df: pyspark.RDD) -> None:
        for i,ele in enumerate(df.collect()):
            image = ele[0].astype(np.uint8)
            image = image.reshape(32,32,3)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.savefig(f"images/image{i}.png")

    def standardize(df: DataFrame) -> DataFrame:
        
        rdds = df.rdd.map(lambda x: [imageIO.imageStructToArray(x[0]),x[1]])
        rdds = rdds.map(lambda x: [t.transform(x[0]),x[1]])
        rdds = rdds.map(lambda x: [imageIO.imageArrayToStruct(x[0]),x[1]])
        df = rdds.toDF(["image","label"])
        return df

    def map_rdd_to_df(rdd: pyspark.RDD) -> DataFrame: #Returns None as of now
        if not rdd.isEmpty():

            image_schema = StructType([StructField("data",BinaryType(),True),StructField("height",IntegerType(),True),StructField("mode",StringType(),True),StructField("nChannels",IntegerType(),True),StructField("width",IntegerType(),True)])
            schema = StructType([StructField("image",image_schema,True),StructField("label",IntegerType(),True)])
            df = sql_context.createDataFrame(rdd, schema)
            # feature = DeepImageFeaturizer(inputCol="image",outputCol="feature",modelName="Xception")
            # model = Pipeline(stages=[feature]).fit(df)
            # transformed_df = model.transform(df).show()
            # predictionAndLabels = transformed_df.select("prediction")
            # predictionAndLabels.show()
            # df.show()
        #     df.printSchema()
            # return transformed_df
        print("Total Batch Size of RDD Received :",len(rdd.collect()))
        print("---------------------------------------")

    def train_ann(rdd: pyspark.RDD):
        if not rdd.isEmpty():
            schema = StructType([StructField("image",VectorUDT(),True),StructField("label",IntegerType(),True)])
            df = sql_context.createDataFrame(rdd,schema)
            Ann.train(df,mlp)
            # pred = model.transform(df).show()
        print("Total Batch Size of RDD Received :",len(rdd.collect()))
        print("---------------------------------------")

    sc = SparkContext("local[2]","CIFAR")
    ssc = StreamingContext(sc,3)
    sql_context = SQLContext(sc)

    stream = ssc.socketTextStream("localhost",6100)

    json_stream = stream.map(lambda line: json.loads(line))        
    json_stream_exploded = json_stream.flatMap(lambda x: x.values())
    json_stream_exploded = json_stream_exploded.map(lambda x : list(x.values()))
    pixels = json_stream_exploded.map(lambda x: [np.array(x[:-1]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),x[-1]])

    t = Transforms([
            # # Resize((512,512)),
            # # ColorShift(2,0,1),
            RandomHorizontalFlip(p=0.45),
            # RandomVerticalFlip(p=1),
            Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
                    std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
        ])

    Ann = MLP(layers=[3072,10], transforms=t, batch_size=256)
    mlp = MLPClassifier(hidden_layer_sizes=[3072,512,64,10], batch_size=256, learning_rate_init=3e-4, random_state=0, warm_start=False, alpha=5e-4,max_iter=100)
    pixels = Ann.preprocess(pixels)
    pixels.foreachRDD(train_ann)


    # ann_model = ANN(
    #     layers=[10,10,10], transforms=t, sql_context=sql_context
    # )

    # ann_model.train(pixels)
    # model = ann_model.model

    # pixels = pixels.map(lambda x: [t.transform(x[0]),x[1]])
    # pixels = pixels.map(lambda x : [imageIO.imageArrayToStruct(x[0],sparkMode='RGB'),x[1]])
    # pixels.foreachRDD(map_rdd_to_df)
    # pixels.map(lambda x : map_rdd_to_df(x))
    # pixels = json_stream_exploded.map(lambda x: [np.array(list(x.values())[:-1]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),list(x.values())[-1]])
    # pixels = json_stream_exploded.map(lambda x: [imageIO.imageArrayToStruct(np.array([[v for k,v in x.items() if 'feature' in k]]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8)),[v for k,v in x.items() if 'label' in k][0]])

    ssc.start()
    ssc.awaitTermination()