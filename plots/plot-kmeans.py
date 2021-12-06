import matplotlib.pyplot as plt
# from torchvision.utils import make_grid,save_image
import seaborn as sns 
import numpy as np
import pandas as pd
import time

avg_accuracy_DeepKmeans = np.load('../checkpoints/DEEPKMEANS-Batch=256/smooth_accuracy-epoch-18.npy')
loss_smooth_DeepKmeans =  np.load('../checkpoints/DEEPKMEANS-Batch=256/smooth_loss-epoch-18.npy')
avg_recall_DeepKmeans = np.load('../checkpoints/DEEPKMEANS-Batch=256/smooth_recall-epoch-18.npy')
avg_precision_DeepKmeans = np.load('../checkpoints/DEEPKMEANS-Batch=256/smooth_precision-epoch-18.npy')
avg_f1_DeepKmeans = np.load('../checkpoints/DEEPKMEANS-Batch=256/smooth_f1-epoch-18.npy')

avg_accuracy_Kmeans = np.load('../checkpoints/KMEANS-Batch:512/smooth_accuracy-epoch-16.npy')
loss_smooth_Kmeans =  np.load('../checkpoints/KMEANS-Batch:512/smooth_loss-epoch-16.npy')
avg_recall_Kmeans = np.load('../checkpoints/KMEANS-Batch:512/smooth_recall-epoch-16.npy')
avg_precision_Kmeans = np.load('../checkpoints/KMEANS-Batch:512/smooth_precision-epoch-16.npy')
avg_f1_Kmeans = np.load('../checkpoints/KMEANS-Batch:512/smooth_f1-epoch-16.npy')

def plot_train_loss(loss_smooth_Kmeans, loss_smooth_DeepKmeans):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    

    plt.plot(loss_smooth_Kmeans,label="Kmeans",alpha=0.9,color="red")
    plt.plot(loss_smooth_DeepKmeans,label="DeepImage + Kmeans",alpha=0.9,color="blue")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)

    labels = np.arange(1,len(loss_smooth_Kmeans)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs =  np.arange(1,len(loss_smooth_Kmeans)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train Loss Kmeans vs DeepKmeans ",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Loss",fontsize=15,labelpad=15)
    plt.savefig('./loss-pic.png')

def plot_train_accuracies(avg_accuracy_Kmeans, avg_accuracy_DeepKmeans):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_accuracy_Kmeans,label="Kmeans",alpha=0.9,color="red")
    plt.plot(avg_accuracy_DeepKmeans,label="DeepImage + Kmeans",alpha=0.9,color="blue")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_accuracy_Kmeans)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_accuracy_Kmeans)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train accuracy Kmeans vs DeepKmeans ",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Accuracy",fontsize=15,labelpad=15)
    plt.savefig('./acc-pic.png')

def plot_train_recall(avg_recall_Kmeans, avg_recall_DeepKmeans):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_recall_Kmeans,label="Kmeans",alpha=0.9,color="red")
    plt.plot(avg_recall_DeepKmeans,label="DeepImage + Kmeans",alpha=0.9,color="blue")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_recall_Kmeans)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_recall_Kmeans)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train recall Kmeans vs DeepKmeans ",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./recall-pic.png')

def plot_train_precision(avg_precision_Kmeans, avg_precision_DeepKmeans):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    

    plt.plot(avg_precision_Kmeans,label="Kmeans",alpha=0.9,color="red")
    plt.plot(avg_precision_DeepKmeans,label="DeepImage + Kmeans",alpha=0.9,color="blue")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_precision_Kmeans)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_precision_Kmeans)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train precision Kmeans vs DeepKmeans ",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./precision-pic.png')

def plot_train_f1(avg_f1_Kmeans, avg_f1_DeepKmeans):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_f1_Kmeans,label="Kmeans",alpha=0.9,color="red")
    plt.plot(avg_f1_DeepKmeans,label="DeepImage + Kmeans",alpha=0.9,color="blue")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_f1_Kmeans)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_f1_Kmeans)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train f1 Kmeans vs DeepKmeans ",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train f1",fontsize=15,labelpad=15)
    plt.savefig('./f1-pic.png')


plot_train_f1(avg_f1_Kmeans, avg_f1_DeepKmeans)
plot_train_precision(avg_precision_Kmeans, avg_precision_DeepKmeans)
plot_train_recall(avg_recall_Kmeans, avg_recall_DeepKmeans)
plot_train_accuracies(avg_accuracy_Kmeans, avg_accuracy_DeepKmeans)
plot_train_loss(loss_smooth_Kmeans, loss_smooth_DeepKmeans)

