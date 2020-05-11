#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import SQLContext
import pyspark.sql.functions as sf

from slicing.sparked import sparked_utils, sparked_slicer, sparked_union_slicer


ten_binner = sf.udf(lambda arg: int(arg / 10))
fnlwgt_binner = sf.udf(lambda arg: int(arg // 100000))
edu_binner = sf.udf(lambda arg: int(arg / 5))
cap_gain_binner = sf.udf(lambda arg: int(arg / 1000))

conf = SparkConf().setAppName("adults_test").setMaster('local[2]')
num_partitions = 2
model_type = "classification"
label = 'Income'
sparkContext = SparkContext(conf=conf)
sqlContext = SQLContext(sparkContext)
dataset_df = sqlContext.read.csv('/slicing/datasets/adult.csv', header='true', inferSchema='true')
# initializing stages of main transformation pipeline
stages = []
dataset_df = dataset_df.withColumn('age_bin', ten_binner(dataset_df['Age']))
dataset_df = dataset_df.withColumn('fnlwgt_bin', fnlwgt_binner(dataset_df['fnlwgt']))
dataset_df = dataset_df.withColumn('edu_num_bin', edu_binner(dataset_df['EducationNum']))
dataset_df = dataset_df.withColumn('cap_gain_bin', cap_gain_binner(dataset_df['CapitalGain']))
dataset_df = dataset_df.withColumn('hours_per_w_bin', ten_binner(dataset_df['HoursPerWeek']))
dataset_df = dataset_df.withColumn('model_type', sf.lit(1))

# list of categorical features for further hot-encoding
cat_features = ["age_bin", "WorkClass", "fnlwgt_bin", "Education", "edu_num_bin",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "cap_gain_bin", "CapitalLoss", "hours_per_w_bin", "NativeCountry"]

# hot encoding categorical features
for feature in cat_features:
    string_indexer = StringIndexer(inputCol=feature, outputCol=feature + "_index")
    encoder = OneHotEncoderEstimator(inputCols=[string_indexer.getOutputCol()], outputCols=[feature + "_vec"])
    encoder.setDropLast(False)
    stages += [string_indexer, encoder]
assembler_inputs = [feature + "_vec" for feature in cat_features]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs")
stages += [assembler]
assembler_final = VectorAssembler(inputCols=["assembled_inputs"], outputCol="features")
label_indexer = StringIndexer(inputCol=label, outputCol=label+"_idx")
stages += [assembler_final]
stages += [label_indexer]
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(dataset_df)
dataset_transformed = pipeline_model.transform(dataset_df)
cat_dict = []
decode_dict = {}
counter = 0
cat = 0
for feature in cat_features:
    colIdx = dataset_transformed.select(feature, feature + "_index").distinct().rdd.collectAsMap()
    colIdx = {k: v for k, v in sorted(colIdx.items(), key=lambda item: item[1])}
    for item in colIdx:
        decode_dict[counter] = (cat, item, colIdx[item])
        counter = counter + 1
    cat = cat + 1
df_transform_fin = dataset_transformed.select('features', label+"_idx", 'model_type')
splits = df_transform_fin.randomSplit([0.8, 0.2], seed=1234)
train_df = splits[0]
test_df = splits[1]
rf = RandomForestClassifier(featuresCol='features', labelCol=label+"_idx", numTrees=10)
rf_model = rf.fit(train_df)
predictions = rf_model.transform(test_df)
# Select example rows to display.
predictions.select("features", label+"_idx", "prediction", "model_type")
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol=label+"_idx", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
loss = 1.0 - accuracy
pred_df_fin = predictions.withColumn('error', sparked_utils.calc_loss(predictions[label+"_idx"],
                                                                      predictions['prediction'],
                                                                      predictions['model_type']))
predictions = pred_df_fin.select('features', 'error').repartition(num_partitions)
all_features = range(predictions.toPandas().values[0][0].size)
predictions = predictions.collect()
k = 10
SparkContext.broadcast(sparkContext, loss)
SparkContext.broadcast(sparkContext, predictions)
SparkContext.broadcast(sparkContext, all_features)
SparkContext.broadcast(sparkContext, decode_dict)
enumerator = "join"
if enumerator == "join":
    sparked_slicer.parallel_process(all_features, predictions, loss, sparkContext, debug=True, alpha=6, k=k, w=0.5,
                                    loss_type=0, enumerator="join")
elif enumerator == "union":
    sparked_union_slicer.process(all_features, predictions, loss, sparkContext, debug=True, alpha=4, k=k, w=0.5, loss_type=0,
                                 enumerator="union")

