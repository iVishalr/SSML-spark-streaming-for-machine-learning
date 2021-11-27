from typing import List
import numpy as np
from numpy.lib.function_base import select
import pyspark
from pyspark.ml.linalg import DenseVector
from sklearn import cluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from pyspark.streaming.dstream import DStream 
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator
from joblibspark import register_spark
from sklearn.utils import parallel_backend
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from scipy import stats
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import StandardScaler

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
        # print(X.shape)
        # print(y)

        with parallel_backend("spark", n_jobs=4):
            pca = PCA(n_components=10)
            sc = StandardScaler() 
            X_pca = sc.fit_transform(X)
            X_pca = pca.fit_transform(X)
            km.partial_fit(X_pca)

        cluster_label_dict = {}
        for cluster_num in range(10):
            # print("cluster_num + km labels",cluster_num, km.labels_)
            idx = self.cluster_indices(cluster_num, km.labels_)
            if len(idx) == 0:
                # print("idx is empty so appending 0")
                idx = np.array([0]) 
                # print("idx", idx)
            original_labels = np.take(y, idx)
            # print("original labels", original_labels)
            mode = stats.mode(original_labels)[0][0]
            # print(mode)
            cluster_label_dict.update({cluster_num: mode})

        # prediction
        predicted_cluster = km.predict(X_pca)
        predicted_labels = np.vectorize(cluster_label_dict.get)(predicted_cluster)

        accuracy = self.classification_accuracy(predicted_labels, y)
        # print(" K means clustering accuracy for cifar 10 = {}".format(accuracy))

        # visualise clusters
        cluster_dict = {key: [] for key in np.arange(0,10)}
        for i,value in enumerate(predicted_cluster):
            cluster_dict[value].append(X[i])
        cluster_centroids = km.cluster_centers_
        # print("cluster_centroids", cluster_centroids)
        self.visualize(cluster_centroids.astype(np.uint8), cluster_dict, y)

        
        # predictions = km.predict(X)
       
        # accuracy = km.score(X, y)
        
        # loss = mean_squared_error(y,predicted_cluster)

        loss = km.inertia_
        precision = precision_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [predicted_cluster, accuracy, loss, precision, recall, f1]

    def classification_accuracy(self, prediction, ground_truth):
        ground_truth = ground_truth[:prediction.shape[0]]
        n_images = prediction.shape[0]
        x = prediction - ground_truth
        n_wrong_predictions = np.count_nonzero(x)
        accuracy = (n_images - n_wrong_predictions) / n_images

        return accuracy*100

    def cluster_indices(self,clust_num, labels_array):
        # print("cluster_indices_function",np.where(labels_array == clust_num)[0])
        return np.where(labels_array == clust_num)[0]


    def visualize(self,cluster_centroid, cluster_dict, y):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # cluster_centroid = np.reshape(cluster_centroid, [10, 32, 32, 3])
        # for i in range(10):
        #     centroid = cluster_centroid[i, :, :, :]
        #     # centroid = centroid / centroid.max()
        #     plt.subplot(2, 5, i + 1)
        #     plt.axis('off')
        #     plt.title(f"{i + 1}")
        #     plt.imshow(centroid)
        # plt.savefig('images/centroid.png')

        for i in cluster_dict:
            if len(cluster_dict[i]) == 0:
                continue
            print("length of cluster_dict[i]",len(cluster_dict[i]))
            data = np.vstack(cluster_dict[i]).reshape(-1, 32,32,3).astype(np.uint8)
            n = min(25, data.shape[0]) 
            random = False
            self.plot_img(data,n,random, classes[i])


    def plot_img(self,data,n,random, true_label):
        n_rows = min(5, data.shape[0])
        # if data.shape[0] % n_rows == 0:
        #     y = data.shape[0]//n_rows
        # else:
        #     y = (data.shape[0]//n_rows) + 1
        # fig, ax = plt.subplots(n_rows,y,figsize=(5,5))
        # i=0
        # print("ax and len(ax)",ax, len(ax))
        # for j in range(n_rows-1):
        #     for k in range(y):
        #         if random:
        #             i = np.random.choice(range(len(data)))
        #         ax[j][k].set_axis_off()
        #         ax[j][k].imshow(data[i:i+1][0])
        #         i+=1
        # j = n_rows-1
        # for k in range(abs(i-data.shape[0])):
        #     ax[j][k].set_axis_off()
        #     ax[j][k].imshow(data[i:i+1][0])

        fig = plt.figure(figsize=(5, 5))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(n_rows, (data.shape[0]//n_rows)+1),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(grid, [i for i in data]):
            # Iterating over the grid returns the Axes.
            ax.set_axis_off()
            ax.imshow(im)
        plt.savefig(f'images/cluster-image-{true_label}.png')


    

    