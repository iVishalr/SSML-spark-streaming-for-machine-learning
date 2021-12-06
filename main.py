from trainer import TrainingConfig, SparkConfig, Trainer
from models import MLP,SVM,Kmeans,DeepImageMLP,DeepImageSVM, DeepImage, DeepKmeans
from transforms import Transforms, RandomHorizontalFlip, Normalize

transforms = Transforms([RandomHorizontalFlip(p=0.345), 
                        Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618), 
                                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))])

if __name__ == "__main__":
    train_config = TrainingConfig(batch_size=512, 
                                    max_epochs=100, 
                                    learning_rate=3e-5,
                                    alpha=8e-4, 
                                    model_name="KMEANS-Batch:512", 
                                    ckpt_interval_batch=5e4, 
                                    load_model="epoch-13")

    spark_config = SparkConfig(batch_interval=5,
                                port=6100)

    mlp = MLP(layers=[3072,128,64,10])
    svm = SVM()
    km = Kmeans()
    deep_mlp = DeepImageMLP()
    deep_svm = DeepImageSVM()
    deepFeature = DeepImage(modelName="ResNet50")
    deep_kmeans = DeepKmeans()
    trainer = Trainer(km,"train", train_config, spark_config, transforms)
    trainer.train()
    # trainer.predict()