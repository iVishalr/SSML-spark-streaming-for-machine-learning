from typing import List
import numpy as np
from numpy.lib.function_base import select
import pyspark
from pyspark.ml.linalg import DenseVector
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
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

class Kmeans:
    def __init__(self, n_clusters=10):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    
    def configure_model(self, configs):
        model = self.model
        model.batch_size = configs.batch_size
        model.max_iter = configs.max_epochs
        return model

    def train(self, df: DataFrame, km : MiniBatchKMeans) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        print(X.shape)
        print(y)

        with parallel_backend("spark", n_jobs=4):
            pca = PCA(n_components=10)
            X = pca.fit_transform(X)
            km.partial_fit(X)
        # print("Score on training set: %0.8f" % svm.score(X, y))
        predictions = km.predict(X)
        # predictions_prob = km.predict_proba(X)
        accuracy = 0
        # loss = log_loss(y,predictions_prob,labels=np.arange(0,10))
        loss = 0
        # precision = 0
        # recall = 0
        # f1 = 0    
        precision = precision_score(y,predictions, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predictions, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        stats = self.clusters_stats(predictions, y)
        purity = self.clusters_purity(stats)

        print("Plotting an extract of the 10 clusters, overall purity: %f" % purity)
        self.plot_clusters(predictions, y, stats, X)

        return [predictions, accuracy, loss, precision, recall, f1]

    def clust_stats(self, cluster):
        class_freq = np.zeros(10)
        for i in range(10):
            class_freq[i] = np.count_nonzero(cluster == i)
        most_freq = np.argmax(class_freq)
        n_majority = np.max(class_freq)
        n_all = np.sum(class_freq)
        return (n_majority, n_all, most_freq)
  
    def clusters_stats(self,predict, y):
        stats = np.zeros((10,3))
        for i in range(10):
            indices = np.where(predict == i)
            cluster = y[indices]
            stats[i,:] = self.clust_stats(cluster)
        return stats
  
    def clusters_purity(self,clusters_stats):
        majority_sum  = clusters_stats[:,0].sum()
        n = clusters_stats[:,1].sum()
        return majority_sum / n

    def plot_d(self, digit, label):
        plt.axis('off')
        plt.imshow(digit.reshape((32,32,3)), cmap=plt.cm.gray)
        plt.title(label)

    def plot_ds(self, digits, title, labels):
        n=digits.shape[0]
        print("n", n)
        print("digits", digits)
        n_rows=n//25+1
        n_cols=25
        plt.figure(figsize=(20,20))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(title)
        for i in range(n):
            plt.subplot(n_rows, n_cols, i + 1)
            self.plot_d(digits[i,:], "%d" % labels[i])
        
    def plot_clusters(self, predict, y, stats, X):
        for i in range(10):
            indices = np.where(predict == i)
            label_encoding = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            title = "Most freq item %s, cluster size %d, majority %d, Label %d " % (label_encoding[int(stats[i,2])], stats[i,1], stats[i,0],stats[i,2])
            self.plot_ds(X[indices], title, y[indices])