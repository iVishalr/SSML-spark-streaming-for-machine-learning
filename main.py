from trainer import TrainingConfig, SparkConfig, Trainer
from models import MLP,SVM,Kmeans,DeepImageMLP,DeepImageSVM, DeepImage
from transforms import Transforms, RandomHorizontalFlip, Normalize

import os

import transforms
os.system("taskset -p 0xff %d" % os.getpid())
os.environ["OMP_NUM_THREADS"] = "4"

transforms = Transforms([RandomHorizontalFlip(p=0.345), Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))])

if __name__ == "__main__":
    train_config = TrainingConfig()
    spark_config = SparkConfig()
    mlp = MLP(layers=[3072,128,64,10])
    svm = SVM()
    km = Kmeans()
    dimlp = DeepImageMLP()
    dimsvm = DeepImageSVM()
    deepFeature = DeepImage(modelName="ResNet50")
    trainer = Trainer(km,"train", train_config, spark_config)
    trainer.train()
