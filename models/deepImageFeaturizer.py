import numpy as np
import sparkdl
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType, StructField, BinaryType, IntegerType, StringType


class DeepImage:
    def __init__(self, modelName) -> None:
        self.model = DeepImageFeaturizer(inputCol="image", outputCol="feature", modelName=modelName)
        self.image_schema = StructType([StructField("data",BinaryType(),True),StructField("height",IntegerType(),True),StructField("mode",StringType(),True),StructField("nChannels",IntegerType(),True),StructField("width",IntegerType(),True)])
        self.schema = StructType([StructField("image",self.image_schema,True),StructField("label",IntegerType(),True)])

    def featurize(self, df:DataFrame, model: sparkdl.DeepImageFeaturizer ,path) -> None:
        model = Pipeline(stages=[model]).fit(df)
        transformed_df = model.transform(df)
        features = transformed_df.select("feature")
        features = features.rdd.map(lambda x: [x[0].toArray()])
        features = np.vstack(features.collect())
        print(f"Saving Feature_Batch to {path}")
        with open(path,"wb") as f:
            np.save(f,features)
        return [model,0, 0, 0, 0, 0, 0]

    def configure_model(self,configs):
        return self.model