from sklearn.linear_model import SGDClassifier
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

register_spark()

class SVM:
    def __init__(self, layers, transforms):
        self.layers = layers
        self.transforms = transforms
        self.params = None

    def preprocess(self, stream: DStream):
        stream = stream.map(lambda x: [self.transforms.transform(x[0]).reshape(3,32,32).reshape(-1).tolist(),x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]),x[1]])
        return stream

    def train(self, df: DataFrame):
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        print(X.shape)
        print(y)

        svm = SGDClassifier(random_state=0)
        with parallel_backend("spark", n_jobs=8):
            svm.partial_fit(X,y,np.arange(0,10).tolist())
        print("Score on training set: %0.8f" % svm.score(X, y))