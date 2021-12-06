from os import name
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



labels = ["DeepKmeans batch:256","KMEANS-Batch:512","DeepImageMLP-Batch:64","DeepImageMLP-Batch:128","DeepImageMLP-Batch:256","DeepImageSVM-Batch:64","DeepImageSVM-Batch:128","DeepImageSVM-Batch:256","MLP-Batch:64","MLP-Batch:128","MLP-Batch:256"]
accuracy = [0.33573717948717957, 0.2664473684210526, 0.6414262820512823, 0.7854567307692307, 0.7733373397435896, 0.7233573717948723, 0.7423878205128207, 0.7653245192307692, 0.5012019230769225,0.5097155448717948,0.5096153846153846]
loss = [8936.685546875,1056481.1663118165,1.187242896819353,0.6286743360556681,0.6709908371962158,1.2823130166313963,1.2078510695907128,1.1959309790819002,2.4967555351070434,2.2069786782801506,1.9778537957324351]
precision = [0.06074304862622983,0.11495603215736669,0.6188115961320473, 0.7906647579602967,0.7778159687455183,0.7755774806328507,0.7816642041362158,0.79474608155932,0.5170526678261407,0.5074741301575812,0.5053458307185814]
recall = [0.07160462700985748,0.11631429464213336,0.6450460063440833,0.7884757049121947,0.7743285653382254,0.7285519893212201,0.7454667970126363,0.7673750911289803,0.49694476997361603,0.5090171474055188,0.5080146739147432]
f1 = [0.06553183844230531,0.11560675495940824,0.6311544388770829,0.7895135064027454,0.7760527279390593,0.7507892488762303,0.7630046371425083,0.7807834412378677,0.5061590552395465,0.5080732658687651,0.5066468548136006]
index = np.arange(0,10).tolist()
X_axis = np.arange(len(labels[2:]))

sns.set()
plt.style.use('seaborn-ticks')

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[2:], accuracy[2:], color="red", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Accuracy")
plt.title("Comparison of Accuracy for different models")
plt.savefig("./bargraph-Accuracy.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[2:], loss[2:], color="blue", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Loss")
plt.title("Comparison of Loss for different models")
plt.savefig("./bargraph-Loss.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[2:], precision[2:], color="purple", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Precision")
plt.title("Comparison of Precision for different models")
plt.savefig("./bargraph-Precision.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[2:], recall[2:], color="orange", width=0.7, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Recall")
plt.title("Comparison of Recall for different models")
plt.savefig("./bargraph-Recall.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[2:], f1[2:], color="red", width=0.2, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("F1 SCORE")
plt.title("Comparison of F1 Score for different models")
plt.savefig("./bargraph-F1.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[:2], accuracy[:2], color="red", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Accuracy")
plt.title("Comparison of Accuracy for different Kmeans models")
plt.savefig("./bargraph-Kmeans-Accuracy.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[:2], loss[:2], color="blue", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Loss")
plt.title("Comparison of Loss for different Kmeans models")
plt.savefig("./bargraph-Kmeans-Loss.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[:2], precision[:2], color="purple", width=0.5, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Precision")
plt.title("Comparison of Precision for different Kmeans models")
plt.savefig("./bargraph-Kmeans-Precision.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[:2], recall[:2], color="orange", width=0.7, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("Recall")
plt.title("Comparison of Recall for different Kmeans models")
plt.savefig("./bargraph-Kmeans-Recall.png")

fig = plt.figure(figsize=(15,15))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.bar(labels[:2], f1[:2], color="red", width=0.2, alpha=0.5)
plt.xticks(rotation = 45)
plt.ylabel("F1 SCORE")
plt.title("Comparison of F1 Score for different Kmeans models")
plt.savefig("./bargraph-Kmeans-F1.png")

# DeepKmeans batch 256
# Test Accuracy : 0.33573717948717957 
# Test Loss : 8936.685546875 
# Test Precision : 0.06074304862622983 
# Test Recall : 0.07160462700985748 
# Test F1 Score: 0.06553183844230531

# KMEANS-Batch:512
# Test Accuracy : 0.2664473684210526 
# Test Loss : 1056481.1663118165 
# Test Precision : 0.11495603215736669 
# Test Recall : 0.11631429464213336 
# Test F1 Score: 0.11560675495940824

# DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4
# Test Accuracy : 0.6414262820512823 
# Test Loss : 1.187242896819353 
# Test Precision : 0.6188115961320473 
# Test Recall : 0.6450460063440833 
# Test F1 Score: 0.6311544388770829

# DeepMLP-Layers:2048-128-64-10-Batch:128-lr:1e-4-alpha:8e-4
# Test Accuracy : 0.7854567307692307 
# Test Loss : 0.6286743360556681 
# Test Precision : 0.7906647579602967 
# Test Recall : 0.7884757049121947 
# Test F1 Score: 0.7895135064027454

# DeepMLP-Layers:2048-128-64-10-Batch:256-lr:3e-4-alpha:5e-4
# Test Accuracy : 0.7733373397435896 
# Test Loss : 0.6709908371962158 
# Test Precision : 0.7778159687455183 
# Test Recall : 0.7743285653382254 
# Test F1 Score: 0.7760527279390593


# DIMSVM-Batch:64-LR:2e-4-alpha:6e-4
# Test Accuracy : 0.7233573717948723 
# Test Loss : 1.2823130166313963 
# Test Precision : 0.7755774806328507 
# Test Recall : 0.7285519893212201 
# Test F1 Score: 0.7507892488762303

# DIMSVM-Batch:128-LR:3e-4-alpha:5e-4
# Test Accuracy : 0.7423878205128207 
# Test Loss : 1.2078510695907128 
# Test Precision : 0.7816642041362158 
# Test Recall : 0.7454667970126363 
# Test F1 Score: 0.7630046371425083

# DIMSVM-Batch:256-LR:3e-4-alpha:5e-4
# Test Accuracy : 0.7653245192307692 
# Test Loss : 1.1959309790819002 
# Test Precision : 0.79474608155932 
# Test Recall : 0.7673750911289803 
# Test F1 Score: 0.7807834412378677 

# MLP-Batch:64-LR:2e-4-alpha:6e-4
# Test Accuracy : 0.5012019230769225 
# Test Loss : 2.4967555351070434 
# Test Precision : 0.5170526678261407 
# Test Recall : 0.49694476997361603 
# Test F1 Score: 0.5061590552395465

# MLP-Batch:128-LR:3e-4-alpha:5e-4
# Test Accuracy : 0.5097155448717948 
# Test Loss : 2.2069786782801506 
# Test Precision : 0.5074741301575812 
# Test Recall : 0.5090171474055188 
# Test F1 Score: 0.5080732658687651 

# MLP-Batch:256-LR:3e-4-alpha:5e-4
# Test Accuracy : 0.5096153846153846 
# Test Loss : 1.9778537957324351 
# Test Precision : 0.5053458307185814 
# Test Recall : 0.5080146739147432 
# Test F1 Score: 0.5066468548136006


