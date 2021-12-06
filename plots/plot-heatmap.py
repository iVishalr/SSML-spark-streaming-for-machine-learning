import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("../checkpoints/DeepMLP-Layers:2048-128-64-10-Batch:64-lr:1e-4-alpha:8e-4/confusion-matrix.npy","rb") as f:
    array = np.load(f)
print(array)
df_cm = pd.DataFrame(array, index = [i for i in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']],
                  columns = [i for i in ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']])
plt.figure(figsize = (15,15))
sn.heatmap(df_cm, annot=True, linewidths=.2,cmap="Blues")
plt.savefig("./confusion_matrix.png")