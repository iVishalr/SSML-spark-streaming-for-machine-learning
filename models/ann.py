import numpy as np

from pyspark.ml.linalg import DenseVector
from pyspark.sql.dataframe import DataFrame
from pyspark.streaming.dstream import DStream 
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

class ANN:
    def __init__(self, layers, transforms):
        self.layers = layers
        self.transforms = transforms
        self.params = None

    def preprocess(self, stream: DStream):
        stream = stream.map(lambda x: [self.transforms.transform(x[0]).reshape(3,32,32).reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]),x[1]])
        return stream

    def train(self, df: DataFrame):
        ann = MultilayerPerceptronClassifier(layers=self.layers,maxIter=100,blockSize=128, seed=0, featuresCol="image")
       
        if self.params == None:
            bias = []
            for layer in self.layers[1:]:
                bias += [0]*layer
            weights = []
            for layer in range(1,len(self.layers)):
                weight = np.random.randn(self.layers[layer-1], self.layers[layer]) * 0.01
                weights += weight.reshape(-1).tolist()
            self.params = weights + bias
        
        ann = ann.setInitialWeights(self.params)
        model = ann.fit(df)
        prediction_df = model.transform(df).select("label", "prediction")
        evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label',metricName='accuracy')
        print(evaluator.evaluate(prediction_df))
        self.params = model.weights
