from typing import List
import numpy as np
import warnings

from joblibspark import register_spark

from sklearn.utils import parallel_backend
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')

register_spark()

class DeepImageMLP:
    def __init__(self, layers=[2048,128,64,10], activation="relu"):
        self.model = MLPClassifier(hidden_layer_sizes=layers, activation=activation)

    def train(self, df: DataFrame, mlp : MLPClassifier, path) -> List:
        with open(path, "rb") as f:
            X = np.load(f)
        y = np.array(df.select("label").collect()).reshape(-1)

        with parallel_backend("spark", n_jobs=4):
            mlp.partial_fit(X,y,np.arange(0,10).tolist())

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

    def predict(self, df: DataFrame, mlp : MLPClassifier, path) -> List:
        with open(path, "rb") as f:
            X = np.load(f)
        y = np.array(df.select("label").collect()).reshape(-1)
        
        predictions = mlp.predict(X)
        predictions_prob = mlp.predict_proba(X)
        accuracy = mlp.score(X,y)
        loss = log_loss(y,predictions_prob,labels=np.arange(0,10))
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)
        cm = confusion_matrix(y, predictions)
        return [predictions, accuracy, loss, precision, recall, f1, cm]