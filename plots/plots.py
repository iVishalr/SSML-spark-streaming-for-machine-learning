import matplotlib.pyplot as plt
# from torchvision.utils import make_grid,save_image
import seaborn as sns 
import numpy as np
import pandas as pd
import time

accuracy = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/accuracy-epoch-5.npy')
avg_accuracy = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_accuracy-epoch-5.npy')

acc = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/accuracy-epoch-5.npy')
accsmooth = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_accuracy-epoch-5.npy')

loss = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/loss-epoch-5.npy')
loss_smooth =  np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_loss-epoch-5.npy')

recall = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/recall-epoch-5.npy')
avg_recall = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_recall-epoch-5.npy')

precision = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/precision-epoch-5.npy')
avg_precision = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_precision-epoch-5.npy')

f1 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/f1-epoch-5.npy')
avg_f1 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_f1-epoch-5.npy')



def plot_train_loss(loss, loss_smooth):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    

    plt.plot(loss,label="Train Loss",alpha=0.3,color="orange",marker="o")
    plt.plot(loss_smooth,label="Avg Train Loss",alpha=0.9,color="red")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)

    labels = np.arange(1,len(loss)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs =  np.arange(1,len(loss)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4 (Train Loss)",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Loss",fontsize=15,labelpad=15)
    plt.savefig('./loss-pic.png')

def plot_train_accuracies(accuracy, avg_accuracy):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(accuracy,label="Train Accuracy",alpha=0.2,color="orange",marker="o")
    plt.plot(avg_accuracy,label="Avg Train Accuracy",alpha=0.9,color="red")
    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(accuracy)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(accuracy)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4 (Train Accuracy)",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Accuracy",fontsize=15,labelpad=15)
    plt.savefig('./acc-pic.png')

def plot_train_recall(recall, avg_recall):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(recall,label="Train Recall",alpha=0.2,color="orange",marker="o")
    plt.plot(avg_recall,label="Avg Train Recall",alpha=0.9,color="red")
    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(recall)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(recall)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4 (Train Recall)",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./recall-pic.png')

def plot_train_precision(precision, avg_precision):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(precision,label="Train Precision",alpha=0.2,color="orange",marker="o")
    plt.plot(avg_precision,label="Avg Train Precision",alpha=0.9,color="red")
    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(precision)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(precision)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4 (Train Precision)",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./precision-pic.png')

def plot_train_f1(f1, avg_f1):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.plot(f1,label="Train f1",alpha=0.2,color="orange",marker="o")
    plt.plot(avg_f1,label="Avg Train f1",alpha=0.9,color="red")
    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(f1)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(f1)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4 (Train f1)",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train f1",fontsize=15,labelpad=15)
    plt.savefig('./f1-pic.png')


def all_plots(accuracy, loss, recall, precision, f1):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.plot(accuracy,label="Train Accuracy",alpha=1,color="orange",linestyle='-')
    # plt.plot(loss,label="Train Loss",alpha=0.3,color="cyan",marker="_")
    plt.plot(recall,label="Train Recall",alpha=1,color="red",linestyle='-')
    plt.plot(precision,label="Train Precision",alpha=1,color="blue",linestyle='-')
    plt.plot(f1,label="Train f1",alpha=1,color="green",linestyle='-')

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(f1)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(f1)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train metrics",fontsize=15,labelpad=15)
    plt.savefig('./all-pic.png')


plot_train_accuracies(acc, accsmooth)
plot_train_loss(loss, loss_smooth)
plot_train_recall(recall, avg_recall)
plot_train_precision(precision, avg_precision)
plot_train_f1(f1, avg_f1)
all_plots(accsmooth, loss_smooth, avg_recall, avg_precision, avg_f1)

