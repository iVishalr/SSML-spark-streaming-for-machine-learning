from typing import List
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
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

register_spark()

class MLP:
    def __init__(self, layers=[3072,512,64,10], activation="relu"):
        self.model = MLPClassifier(hidden_layer_sizes=layers, activation=activation)

    def train(self, df: DataFrame, mlp : MLPClassifier) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        print(X.shape)
        print(y)

        with parallel_backend("spark", n_jobs=8):
            mlp.partial_fit(X,y,np.arange(0,10).tolist())
        # print("Score on training set: %0.8f" % mlp.score(X, y))
        predictions = mlp.predict(X)
        predictions_prob = mlp.predict_proba(X)
        accuracy = mlp.score(X,y)
        loss = log_loss(y,predictions_prob,labels=np.arange(0,10))
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [mlp,predictions, accuracy, loss, precision, recall, f1]

    def configure_model(self, configs):
        model = self.model
        model.alpha = configs.alpha
        model.learning_rate_init = configs.learning_rate
        model.batch_size = configs.batch_size
        model.warm_start = False
        model.max_iter = configs.max_epochs
        return model

    def predict(self, df: DataFrame, mlp : MLPClassifier) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        
        predictions = mlp.predict(X)
        predictions_prob = mlp.predict_proba(X)
        accuracy = mlp.score(X,y)
        loss = log_loss(y,predictions_prob,labels=np.arange(0,10))
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [predictions, accuracy, loss, precision, recall, f1]