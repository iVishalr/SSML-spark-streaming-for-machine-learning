from typing import List

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from joblibspark import register_spark

from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import parallel_backend
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA, KernelPCA, TruncatedSVD, PCA

from sklearn.metrics import confusion_matrix

from pyspark.sql.dataframe import DataFrame

from torchvision.utils import make_grid
from torch import tensor


register_spark()

class Kmeans:
    def __init__(self, n_clusters=10):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init_size=1024, reassignment_ratio=0.01, batch_size=512)
        self.pca = IncrementalPCA(n_components=10,whiten=False,batch_size=512)
        self.kpca = KernelPCA(n_components=336,kernel="rbf",alpha=2.67,gamma=5e-4,n_jobs=8)
        self.tsvd = TruncatedSVD(n_components=336)
        self.tsne = TSNE(n_components=2,perplexity=30,init="pca")

    def configure_model(self, configs):
        model = self.model
        model.batch_size = configs.batch_size
        model.max_iter = configs.batch_size * 20
        return model

    def train(self, df: DataFrame, km : MiniBatchKMeans) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        print(X.shape)
        print(y)

        with parallel_backend("spark", n_jobs=8):
            
            km.partial_fit(X)

        predicted_cluster = km.predict(X)
        reference_labels = self.get_reference_dict(predicted_cluster,y)
        predicted_labels = self.get_labels(predicted_cluster,reference_labels)
        accuracy = accuracy_score(y,predicted_labels)
        loss = km.inertia_
        precision = precision_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        return [km,predicted_cluster, accuracy, loss, precision, recall, f1]

    def predict(self, df: DataFrame, km: MiniBatchKMeans) -> List:
        X = np.array(df.select("image").collect()).reshape(-1,3072)
        y = np.array(df.select("label").collect()).reshape(-1)
        # X_pca = self.pca.transform(X)
        predicted_cluster = km.predict(X)
        reference_labels = self.get_reference_dict(predicted_cluster,y)
        predicted_labels = self.get_labels(predicted_cluster,reference_labels)
        cm = confusion_matrix(y,predicted_labels)
        accuracy = accuracy_score(y,predicted_labels)
        loss = km.inertia_
        precision = precision_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        recall = recall_score(y,predicted_cluster, labels=np.arange(0,10),average="macro")
        f1 = 2*precision*recall/(precision+recall)

        X = self.inverse_transform(X,mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
        cluster_dict = {i:0 for i in np.unique(predicted_labels).astype(int)}
        for i in np.unique(predicted_labels):
            cluster_dict[i] = X[predicted_labels==i]
        
        print(np.unique(predicted_labels))
        for i in cluster_dict:
            print(cluster_dict[i].shape)

        self.visualize(None,cluster_dict,y)
        cluster_dict = {}

        return [predicted_labels,accuracy, loss, precision, recall, f1, cm]

    def inverse_transform(self,X:np.ndarray, mean: List, std: List) -> np.ndarray:
        shape = X.shape
        X = X.reshape(-1,32,32,3)
        r, g, b = X[:,:,:,0], X[:,:,:,1], X[:,:,:,2]
        r = r*std[0] + mean[0]
        g = g*std[1] + mean[1]
        b = b*std[2] + mean[2]
        X[:,:,:,0] = r
        X[:,:,:,1] = g
        X[:,:,:,2] = b
        X = X*255.0
        X = X.reshape(-1,shape[1]).astype(int)
        return X


    def get_reference_dict(self,predictions,y):
        reference_label = {}
        # For loop to run through each label of cluster label
        for i in range(len(np.unique(predictions))):
            print(np.unique(predictions))
            index = np.where(predictions == i,1,0)
            num = np.bincount(y[index==1]).argmax()
            reference_label[i] = num
        return reference_label
    def get_labels(self,clusters,reference_labels):
        # Mapping predictions to original labels
        temp_labels = np.random.rand(len(clusters))
        for i in range(len(clusters)):
            temp_labels[i] = reference_labels[clusters[i]]
        return temp_labels

    # def cluster_indices(self,clust_num, labels_array):
    #     return np.where(labels_array == clust_num)[0]


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
            grid = make_grid(tensor(cluster_dict[i].reshape(-1,32,32,3).transpose(0,3,1,2)),nrow=10)
            fig = plt.figure(figsize=(10, 10))
            plt.title(f"Class : {classes[i]}")
            plt.imshow(grid.permute(1,2,0))
            plt.savefig(f"images/{classes[i]}.png")