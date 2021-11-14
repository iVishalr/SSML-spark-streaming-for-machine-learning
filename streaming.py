from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.sql import SQLContext
from pyspark.ml.feature import StandardScaler
from sparkdl.image import imageIO

import json
import numpy as np

def Standardize(df):
    scaler = StandardScaler(inputCol="image", outputCol="scaled_image",withStd=True, withMean=False)
    scalerModel = scaler.fit(df)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(df)
    return scaledData

def map_rdd_to_df(rdd):
    if not rdd.isEmpty():
        schema = StructType([StructField("image",imageIO.imageSchema,True), StructField("label", IntegerType(), True)])
        df = sql_context.createDataFrame(rdd, schema)
        df.show()

    print("Total Batch Size of RDD :",len(rdd.collect()))
    print("----------------------------")

sc = SparkContext("local[2]","CIFAR")
ssc = StreamingContext(sc,1)
sql_context = SQLContext(sc)

stream = ssc.socketTextStream("localhost",6100)

json_stream = stream.map(lambda line: json.loads(line))   
keys = json_stream.flatMap(lambda x:list(map(int,x.keys()))) 
                                                               
json_stream_exploded = json_stream.flatMap(lambda x: x.values())

pixels = json_stream_exploded.map(lambda x: [imageIO.imageArrayToStruct(np.array([[v for k,v in x.items() if 'feature' in k]]).reshape(32,32,3)/255.0),[v for k,v in x.items() if 'label' in k][0]])
pixels.foreachRDD(map_rdd_to_df)


ssc.start()
ssc.awaitTermination()