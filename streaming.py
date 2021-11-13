from pyspark.sql import SparkSession as spark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode,split,from_json
from pyspark.sql.types import IntegerType, StringType, StructField, StructType,ArrayType
from pyspark.sql import SQLContext
import json
import numpy as np
def print_rdd(rdd):
    if not rdd.isEmpty():
        schema = StructType([StructField("image",ArrayType(ArrayType(ArrayType(IntegerType()))),True), StructField("label", ArrayType(IntegerType()), True)])
        df = sql_context.createDataFrame(rdd, schema)
        # for i,ele in enumerate(rdd.collect()):
        #     print(ele)
        df.show()
    print("Total Batch Size of RDD :",len(rdd.collect()))
    print("----------------------------")

sc = SparkContext("local[2]","CIFAR")
ssc = StreamingContext(sc,1)
sql_context = SQLContext(sc)

stream = ssc.socketTextStream("localhost",6100)

json_stream = stream.map(lambda line: json.loads(line))   
keys = json_stream.flatMap(lambda x:list(map(int,x.keys()))) # [0 8,1 8,2,3,4,5..31]
                                                               
json_stream_exploded = json_stream.flatMap(lambda x: x.values())

pixels = json_stream_exploded.map(lambda x: list((np.array([[v for k,v in x.items() if 'feature' in k]]).reshape(3,32,32).tolist(),[v for k,v in x.items() if 'label' in k])))

pixels.foreachRDD(print_rdd)


ssc.start()
ssc.awaitTermination()



