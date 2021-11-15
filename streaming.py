import pyspark
from pyspark import SparkContext
from pyspark.sql.dataframe import DataFrame
from pyspark.streaming import StreamingContext
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.sql import SQLContext
from sparkdl.image import imageIO

from transforms import Transforms, Normalize, RandomHorizontalFlip,RandomVerticalFlip, Resize, ColorShift

import json
import numpy as np
import matplotlib.pyplot as plt


def plot(df: DataFrame) -> None:
    
    df = df.rdd.map(lambda x: [imageIO.imageStructToArray(x[0])])
    for ele in df.collect():
        image = ele[0].astype(int)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.show()

def standardize(df: DataFrame) -> DataFrame:
    t = Transforms([
        Resize((224,224)),
        ColorShift(2,0,1),
        RandomHorizontalFlip(p=0.45),
        RandomVerticalFlip(p=0.8),
        Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)),
    ])
    rdds = df.rdd.map(lambda x: [imageIO.imageStructToArray(x[0]),x[1]])
    rdds = rdds.map(lambda x: [t.transform(x[0]),x[1]])
    rdds = rdds.map(lambda x: [imageIO.imageArrayToStruct(x[0]),x[1]])
    df = rdds.toDF(["image","label"])
    # plot(df.select("image"))
    return df

def map_rdd_to_df(rdd: pyspark.RDD) -> None: #Returns None as of now
    if not rdd.isEmpty():
        schema = StructType([StructField("image",imageIO.imageSchema,True), StructField("label", IntegerType(), True)])
        df = sql_context.createDataFrame(rdd, schema)
        df = standardize(df)
        df.show()
        df.printSchema()
    print("Total Batch Size of RDD Received :",len(rdd.collect()))
    print("---------------------------------------")

sc = SparkContext("local[2]","CIFAR")
ssc = StreamingContext(sc,4)
sql_context = SQLContext(sc)

stream = ssc.socketTextStream("localhost",6100)

json_stream = stream.map(lambda line: json.loads(line))   
keys = json_stream.flatMap(lambda x:list(map(int,x.keys()))) 
                                                               
json_stream_exploded = json_stream.flatMap(lambda x: x.values())

pixels = json_stream_exploded.map(lambda x: [imageIO.imageArrayToStruct(np.array([[v for k,v in x.items() if 'feature' in k]]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),'RGB'),[v for k,v in x.items() if 'label' in k][0]])
pixels.foreachRDD(map_rdd_to_df)

ssc.start()
ssc.awaitTermination()