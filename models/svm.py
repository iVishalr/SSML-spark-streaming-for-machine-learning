from typing import List

import warnings
import numpy as np
import matplotlib.pyplot as plt

from joblibspark import register_spark

from sklearn.linear_model import SGDClassifier
from sklearn.utils import parallel_backend
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class SVM:
    def __init__(self, loss='log', penalty='l2'):
        self.model = SGDClassifier(loss=loss, penalty=penalty, random_state=0)
    
    def configure_model(self, configs):
        model = self.model
        model.alpha = configs.learning_rate
        model.warm_start = False
        model.n_iter_ = configs.max_epochs
        return model

    def train(self, df: DataFrame, svm : SGDClassifier) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            svm.partial_fit(X,y,np.arange(0,10).tolist())
        predictions = svm.predict(X)
        predictions = np.array(predictions)
        predictions_prob = svm.predict_proba(X)
        predictions_prob = np.array(predictions_prob)
        predictions_prob[np.isnan(predictions_prob)] = 0 

        accuracy = svm.score(X,y)
        loss = log_loss(y,predictions_prob,labels=np.arange(0,10), eps=1e-1)
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [svm,predictions, accuracy, loss, precision, recall, f1]

    def predict(self, df: DataFrame, svm : SGDClassifier, path) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        
        predictions = svm.predict(X)
        predictions = np.array(predictions)
        predictions_prob = svm.predict_proba(X)
        accuracy = svm.score(X,y)
        predictions_prob = np.array(predictions_prob)
        predictions_prob[np.isnan(predictions_prob)] = 0 
        loss = log_loss(y,predictions_prob,labels=np.arange(0,10),eps=1e-1)
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)
        cm = confusion_matrix(y, predictions)
        return [predictions, accuracy, loss, precision, recall, f1, cm]