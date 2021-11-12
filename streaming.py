
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode,split,from_json
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

import json

def parse_json(line):
    line = json.loads(line)
    return list(line.keys())


# spark = SparkSession.builder.appName("CIFAR").getOrCreate()
spark = SparkContext("local[2]","CIFAR")
ssc = StreamingContext(spark,10)

batch = ssc.socketTextStream("localhost",6100)

batch = batch.map(lambda line: parse_json(line))

# batch.foreachRDD()
batch.pprint()
# batch.reduceByKey(lambda x: json.loads()
# batch.saveAsTextFiles("hello")

ssc.start()
ssc.awaitTermination()

# lines = spark.readStream.format("socket").option("host","localhost").option("port",6100).option().load()

# lines = explode(split(lines.value,":"))
# lines = explode(lines)

# lines = lines.select().alias("words")
# lines = lines.select(from_json(
#     lines.value,StructType(
#         [StructField("batch",StringType(),False),
#          StructField("feature",StringType(),False),
#          StructField("value",IntegerType(),False)
#         ])).alias("json")).collect()

# data = [{"0": {"a": 1}}]
# schema = StructType([StructField("feature", StringType(),False),StructField("pixel",IntegerType(),False)])
# df = spark.createDataFrame(data)

# df = df.select(from_json(df.value,schema))
# df = df.select(from_json(df.value, schema).alias("json")).collect()
# df.show()
# query = df.writeStream.outputMode("append").format("console").start()
# query.awaitTermination()