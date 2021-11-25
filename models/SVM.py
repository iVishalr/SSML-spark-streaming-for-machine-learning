from typing import List
import numpy as np
from numpy.lib.function_base import select
import pyspark
from pyspark.ml.linalg import DenseVector
from sklearn.linear_model import SGDClassifier
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

class SVM:
    def __init__(self, loss='log', penalty='l2'):
        self.model = SGDClassifier(loss=loss, penalty=penalty, random_state=0)
    
    def configure_model(self, configs):
        model = self.model
        model.alpha = configs.alpha
        model.warm_start = False
        model.n_iter_ = configs.max_epochs
        return model

    def train(self, df: DataFrame, svm : SGDClassifier) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        print(X.shape)
        print(y)

        with parallel_backend("spark", n_jobs=4):
            svm.partial_fit(X,y,np.arange(0,10).tolist())
        # print("Score on training set: %0.8f" % svm.score(X, y))
        predictions = svm.predict(X)
        predictions_prob = svm.predict_proba(X)
        accuracy = svm.score(X,y)
        # loss = log_loss(y,predictions_prob,labels=np.arange(0,10))
        loss = 0
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [predictions, accuracy, loss, precision, recall, f1]