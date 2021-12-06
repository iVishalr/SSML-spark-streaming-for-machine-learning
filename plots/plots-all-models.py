import matplotlib.pyplot as plt
# from torchvision.utils import make_grid,save_image
import seaborn as sns 
import numpy as np
import pandas as pd
import time

avg_accuracy_DeepMLP128 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_accuracy-epoch-5.npy')
loss_smooth_DeepMLP128 =  np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_loss-epoch-5.npy')
avg_recall_DeepMLP128 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_recall-epoch-5.npy')
avg_precision_DeepMLP128 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_precision-epoch-5.npy')
avg_f1_DeepMLP128 = np.load('../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/smooth_f1-epoch-5.npy')

avg_accuracy_DIMSVM128 = np.load('../checkpoints/DIMSVM-Batch:64-LR:2e-4-alpha:6e-4/smooth_accuracy-epoch-7.npy')
loss_smooth_DIMSVM128 =  np.load('../checkpoints/DIMSVM-Batch:64-LR:2e-4-alpha:6e-4/smooth_loss-epoch-7.npy')
avg_recall_DIMSVM128 = np.load('../checkpoints/DIMSVM-Batch:64-LR:2e-4-alpha:6e-4/smooth_recall-epoch-7.npy')
avg_precision_DIMSVM128 = np.load('../checkpoints/DIMSVM-Batch:64-LR:2e-4-alpha:6e-4/smooth_precision-epoch-7.npy')
avg_f1_DIMSVM128 = np.load('../checkpoints/DIMSVM-Batch:64-LR:2e-4-alpha:6e-4/smooth_f1-epoch-7.npy')

avg_accuracy_MLP128 = np.load('../checkpoints/MLP-Batch:64-LR:2e-4-alpha:6e-4/smooth_accuracy-epoch-16.npy')
loss_smooth_MLP128 =  np.load('../checkpoints/MLP-Batch:64-LR:2e-4-alpha:6e-4/smooth_loss-epoch-16.npy')
avg_recall_MLP128 = np.load('../checkpoints/MLP-Batch:64-LR:2e-4-alpha:6e-4/smooth_recall-epoch-16.npy')
avg_precision_MLP128 = np.load('../checkpoints/MLP-Batch:64-LR:2e-4-alpha:6e-4/smooth_precision-epoch-16.npy')
avg_f1_MLP128 = np.load('../checkpoints/MLP-Batch:64-LR:2e-4-alpha:6e-4/smooth_f1-epoch-16.npy')






def plot_train_loss(loss_smooth_DeepMLP128,loss_smooth_DIMSVM128,loss_smooth_MLP128):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(loss_smooth_DeepMLP128,label="Deep Image MLP",alpha=0.9,color="red")
    plt.plot(loss_smooth_DIMSVM128,label="Deep Image SVM",alpha=0.9,color="blue")
    plt.plot(loss_smooth_MLP128,label="MLP",alpha=0.9,color="green")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)

    labels = np.arange(1,len(loss_smooth_MLP128)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs =  np.arange(1,len(loss_smooth_MLP128)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title(" Avg Train Loss Deep Image MLP vs Deep Image SVM vs MLP Batch=64",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Loss",fontsize=15,labelpad=15)
    plt.savefig('./loss-pic.png')

def plot_train_accuracies(avg_accuracy_DeepMLP128, avg_accuracy_DIMSVM128,avg_accuracy_MLP128):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_accuracy_DeepMLP128,label="Deep Image MLP",alpha=0.9,color="red")
    plt.plot(avg_accuracy_DIMSVM128,label="Deep Image SVM",alpha=0.9,color="blue")
    plt.plot(avg_accuracy_MLP128,label="MLP",alpha=0.9,color="green")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_accuracy_MLP128)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_accuracy_MLP128)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train Accuracy Deep Image MLP vs Deep Image SVM vs MLP Batch=64",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Accuracy",fontsize=15,labelpad=15)
    plt.savefig('./acc-pic.png')

def plot_train_recall(avg_recall_DeepMLP128, avg_recall_DIMSVM128, avg_recall_MLP128):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_recall_DeepMLP128,label="Deep Image MLP",alpha=0.9,color="red")
    plt.plot(avg_recall_DIMSVM128,label="Deep Image SVM",alpha=0.9,color="blue")
    plt.plot(avg_recall_MLP128,label="MLP",alpha=0.9,color="green")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_recall_MLP128)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_recall_MLP128)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train Recall Deep Image MLP vs Deep Image SVM vs MLP Batch=64",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./recall-pic.png')

def plot_train_precision(avg_precision_DeepMLP128, avg_precision_DIMSVM128, avg_precision_MLP128):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_precision_DeepMLP128,label="Deep Image MLP",alpha=0.9,color="red")
    plt.plot(avg_precision_DIMSVM128,label="Deep Image SVM",alpha=0.9,color="blue")
    plt.plot(avg_precision_MLP128,label="MLP",alpha=0.9,color="green")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_precision_MLP128)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_precision_MLP128)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train Precision Deep Image MLP vs Deep Image SVM vs MLP Batch=64",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train Recall",fontsize=15,labelpad=15)
    plt.savefig('./precision-pic.png')

def plot_train_f1(avg_f1_DeepMLP128, avg_f1_DIMSVM128, avg_f1_MLP128):
    sns.set()
    plt.style.use('seaborn-ticks')
    plt.figure(figsize=(15,20))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.plot(avg_f1_DeepMLP128,label="Deep Image MLP",alpha=0.9,color="red")
    plt.plot(avg_f1_DIMSVM128,label="Deep Image SVM",alpha=0.9,color="blue")
    plt.plot(avg_f1_MLP128,label="MLP",alpha=0.9,color="green")

    yticks = plt.yticks()
    for y_locs in yticks[0][1:]:
        plt.axhline(y=y_locs,color='lightgrey',linestyle='--',lw=1,alpha=1)
    labels = np.arange(1,len(avg_f1_MLP128)+1,1e3)
    xlabels = ['{:,.0f}'.format(x) + 'k' for x in labels/1000]
    xlabels[0] = '0'
    locs = np.arange(1,len(avg_f1_MLP128)+1,1e3).astype(int)
    plt.xticks(ticks=locs,labels=xlabels)
    plt.legend(loc=0,prop={'size':10})
    plt.title("Avg Train f1 Deep Image MLP vs Deep Image SVM vs MLP Batch=64",pad=20,fontsize=15)
    plt.xlabel("Iterations",fontsize=15,labelpad=15)
    plt.ylabel("Train f1",fontsize=15,labelpad=15)
    plt.savefig('./f1-pic.png')



plot_train_loss(loss_smooth_DeepMLP128,loss_smooth_DIMSVM128,loss_smooth_MLP128)
plot_train_f1(avg_f1_DeepMLP128, avg_f1_DIMSVM128, avg_f1_MLP128)
plot_train_precision(avg_precision_DeepMLP128, avg_precision_DIMSVM128, avg_precision_MLP128)
plot_train_recall(avg_recall_DeepMLP128, avg_recall_DIMSVM128, avg_recall_MLP128)
plot_train_accuracies(avg_accuracy_DeepMLP128, avg_accuracy_DIMSVM128,avg_accuracy_MLP128)





