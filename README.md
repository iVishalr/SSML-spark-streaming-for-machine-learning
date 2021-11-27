# SSML Spark Streaming for Machine Learning

UE19CS322 Project

Currently we are working on implementing deep learning models. We are using CIFAR-10 dataset for classification.

## Dataset

Download and extract the dataset from [here](https://drive.google.com/drive/folders/1hKe06r4TYxqQOwEOUrk6i9e15Vt2EZGC). Copy the CIFAR folder to project directory.

## Requirements

Install python3.7 by executing the following commands :

```bash
$ sudo apt update
$ sudo apt install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt install python3.7
```

Use `Python3.7` to install the following packages

1. numpy
2. matplotlib
3. tqdm
4. pyspark==2.4.0
5. Pillow
6. keras==2.0.4
7. tensorflow==1.13.1
8. jieba
9. tensorflowonspark
10. sparkdl
11. tensorframes
12. kafka-python 

Please use the following command to install the above packages.

```bash
$ python3.7 -m pip install packageName
```

## Build

```bash
$ sudo apt update && sudo apt upgrade
$ sudo apt install openjdk-8-jdk
$ sudo apt install scala
$ wget https://archive.apache.org/dist/hadoop/common/hadoop-2.7.0/hadoop-2.7.0.tar.gz
$ wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
```

For installing hadoop and spark refer [this](https://github.com/aditeyabaral/big-data-installs)

Assuming `spark 2.4.0` and `Hadoop 2.7.0` is installed please execute the following :

```bash
$ cd
$ mkdir .ivy2
$ cd .ivy2
$ mkdir jars
$ cd jars
$ wget https://repos.spark-packages.org/databricks/tensorframes/0.2.9-s_2.11/tensorframes-0.2.9-s_2.11.jar
$ wget https://repos.spark-packages.org/databricks/spark-deep-learning/0.3.0-spark2.2-s_2.11/spark-deep-learning-0.3.0-spark2.2-s_2.11.jar
$ wget https://repo1.maven.org/maven2/org/tensorflow/tensorflow/1.13.1/tensorflow-1.13.1.jar
$ wget https://repo1.maven.org/maven2/org/tensorflow/libtensorflow/1.13.1/libtensorflow-1.13.1.jar
$ wget https://repo1.maven.org/maven2/org/tensorflow/libtensorflow_jni/1.13.1/libtensorflow_jni-1.13.1.jar
```

Copy the following alias into your `.bashrc` file

```bash
$ alias spark-submit="/opt/spark/bin/spark-submit --packages com.typesafe.scala-logging:scala-logging-slf4j_2.10:2.1.2 --jars /home/pes1ug19cs019/.ivy2/jars/tensorframes-0.2.9-s_2.11.jar,/home/pes1ug19cs019/.ivy2/jars/spark-deep-learning-0.3.0-spark2.2-s_2.11.jar,/home/pes1ug19cs019/.ivy2/jars/tensorflow-1.13.1.jar,/home/pes1ug19cs019/.ivy2/jars/libtensorflow-1.13.1.jar,/home/pes1ug19cs019/.ivy2/jars/libtensorflow_jni-1.13.1.jar"
```
Change the following environment varaibles `HADOOP_OPTS` and `LD_LIBRARY_PATH` in your `.bashrc` file to

```bash
$ export HADOOP_OPTS="-Djava.library.path=$HADOOP_COMMON_LIB_NATIVE_DIR"
$ export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH
```

Change the environment variable `PYSPARK_PYTHON` in `.profile` file to 

```bash
$ export PYSPARK_PYTHON="/usr/bin/python3.7"
```

After making the above changes execute ```source ~/.bashrc && source ~/.profile```

## Streaming Dataset

Execute the following code in terminal to start streaming the dataset.

```bash
$ python3.7 ./stream.py --file="cifar" --batch-size=32
```

## Executing the spark driver code

Execute the following command in terminal to execute the driver code.


```bash
$ spark-submit <Absolute_Path_To_Streaming.py> > output.txt 2>outputlog.txt
```
