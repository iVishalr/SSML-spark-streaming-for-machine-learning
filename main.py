from trainer import TrainingConfig, SparkConfig, Trainer
from models import MLP,SVM,Kmeans,DeepImageMLP,DeepImageSVM, DeepImage
from transforms import Transforms, RandomHorizontalFlip, Normalize

import os
os.system("taskset -p 0xff %d" % os.getpid())
os.environ["OMP_NUM_THREADS"] = "4"

transforms = Transforms([RandomHorizontalFlip(p=0.345), Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))])

if __name__ == "__main__":
    train_config = TrainingConfig(batch_size=256,max_epochs=100, learning_rate=3e-4,alpha=5e-4, model_name="test", ckpt_interval_batch=(5e4//256)+1)
    spark_config = SparkConfig(batch_interval=3,port=6100)
    mlp = MLP(layers=[2048,512,64,10])
    svm = SVM()
    km = Kmeans()
    dimlp = DeepImageMLP()
    dimsvm = DeepImageSVM()
    deepFeature = DeepImage(modelName="ResNet50")
    trainer = Trainer(dimlp,"train", train_config, spark_config, transforms)
    # trainer.train()
    trainer.predict()

# Deep SVM - lr = 3e-5 bs = 256 max epochs = 50
# SVM also same configuration
# Deep MLP layers = [2048,512,64,10], bs = 256, max_epochs=100, lr = 3e-4, alpha = 5e-4 IMPORTANT 2048 not 3072! (Why?)