import numpy as np
from numpy.lib.function_base import select
import pyspark
from sklearn.neural_network import MLPClassifier
from pyspark.ml.linalg import DenseVector
from pyspark.streaming.dstream import DStream 
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator
from joblibspark import register_spark
from sklearn.utils import parallel_backend
import matplotlib.pyplot as plt

register_spark()

class MLP:
    def __init__(self, layers, transforms, batch_size):
        self.layers = layers
        self.transforms = transforms
        self.batch_size = batch_size
        # self.mlp = MLPClassifier(hidden_layer_sizes=self.layers, batch_size=batch_size, learning_rate_init=3e-4, random_state=0, warm_start=False, alpha=5e-4,max_iter=100)
        
    def preprocess(self, stream: DStream):
        stream = stream.map(lambda x: [self.transforms.transform(x[0]).reshape(32,32,3).reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]),x[1]])
        return stream

    def train(self, df: DataFrame, mlp):
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        print(X.shape)
        print(y)
        # self.plot(X)
        # mlp = MLPClassifier(hidden_layer_sizes=self.layers, batch_size=self.batch_size, learning_rate_init=3e-4, random_state=0, warm_start=False, alpha=5e-4,max_iter=100)
        # mlp.classes_ = np.arange(0,10).tolist()
        with parallel_backend("spark", n_jobs=4):
            mlp.partial_fit(X,y,np.arange(0,10).tolist())
        print("Score on training set: %0.8f" % mlp.score(X, y))

    def plot(self, images):
        image = images[0].reshape(32,32,3).astype(np.uint8)
        # image = image.reshape(32,32,3)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.savefig(f"images/model_image.png")
       