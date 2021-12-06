from os import name
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load(filename):
    file = open(filename)
    data = file.readlines()

    for i in range(len(data)):
        data[i] = data[i][:-2].split(' : ')
    data = data[:5]

    di = {}

    for i in data[:]:
        if ":" in i[0]:
            i[0] = i[0].split(":")
            i[0][1] = i[0][1][1:]
            i = i[0]
            # print(i)

        di[i[0]] = float(i[1])

    return di

Deep_MLP_128 = load("../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:128-lr:1e-4-alpha:8e-4/test-scores-128.txt")
Deep_MLP_256 = load("../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:256-lr:3e-4-alpha:5e-4/test-scores-256.txt")
# MLP_128 = load("../checkpoints/DeepMLP-2048-128-64-10-bs=128-lr=1e-4-alpha=8e-4/test-scores-128.txt")
# MLP_256 = load("../checkpoints/DeepMLP-2048-128-64-10-bs=256-lr=3e-4-alpha=5e-4/test-scores-256.txt")
# MLP_64 = load("../checkpoints/DeepMLP-2048-128-64-10-bs=256-lr=3e-4-alpha=5e-4/test-scores-256.txt")
print(Deep_MLP_128)
print(Deep_MLP_256)

Deep_MLP_128 = pd.Series(Deep_MLP_128, name="Deep Image MLP 128")
Deep_MLP_256 = pd.Series(Deep_MLP_256, name="Deep Iamge MLP 256")
df = pd.concat([Deep_MLP_128,Deep_MLP_256],axis=1).transpose()
df = df.reset_index(drop=True)
print(df)
sns.barplot(df["Test Accuracy"])
plt.savefig("./bar-graph.png")