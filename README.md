# SSML Spark Streaming for Machine Learning

UE19CS322 Project

Currently we are working on mapping the JSON input to Spark DF. We are using CIFAR-10 dataset for classification.

## Dataset

Download and extract the dataset from [here](https://drive.google.com/drive/folders/1hKe06r4TYxqQOwEOUrk6i9e15Vt2EZGC). Copy the CIFAR folder to project directory.

## Streaming Dataset

Execute the following code in terminal to start streaming the dataset.

```bash
$ python3 ./stream.py --file="cifar" --batch-size=32
```

## Executing the spark driver code

Execute the following command in terminal to execute the driver code.

```bash
$ /opt/spark/bin/spark-submit <Absolute_Path_To_Streaming.py> > output.txt 2>outputlog.txt
```
