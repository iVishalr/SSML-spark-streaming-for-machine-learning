import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector

from transforms import Transforms
from trainer import SparkConfig

import json

class DataLoader:
    def __init__(self, sparkContext:SparkContext, sparkStreamingContext: StreamingContext, sqlContext: SQLContext,sparkConf: SparkConfig, transforms: Transforms) -> None:
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(self.sparkConf.stream_host,self.sparkConf.port)
        self.transforms = transforms

    def parse_stream(self)->DStream:
        json_stream = self.stream.map(lambda line: json.loads(line))        
        json_stream_exploded = json_stream.flatMap(lambda x: x.values())
        json_stream_exploded = json_stream_exploded.map(lambda x : list(x.values()))
        pixels = json_stream_exploded.map(lambda x: [np.array(x[:-1]).reshape(3,32,32).transpose(1,2,0).astype(np.uint8),x[-1]])
        pixels = DataLoader.preprocess(pixels,self.transforms)
        return pixels

    @staticmethod
    def preprocess(stream: DStream, transforms) -> DStream:
        stream = stream.map(lambda x: [transforms.transform(x[0]).reshape(32,32,3).reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]),x[1]])
        return stream