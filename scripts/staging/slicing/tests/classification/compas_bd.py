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

import sys

from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, IndexToString
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SQLContext
import pyspark.sql.functions as sf
from pyspark.sql.functions import udf
from sklearn.model_selection import train_test_split

from slicing.spark_modules import spark_utils, join_data_parallel, union_data_parallel


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        k = int(args[1])
        w = float(args[2].replace(',', '.'))
        alpha = int(args[3])
        if args[4] == "True":
            b_update = True
        else:
            b_update = False
        debug = args[5]
        loss_type = int(args[6])
        enumerator = args[7]
        dataset = args[8]
    else:
        k = 10
        w = 0.1
        alpha = 4
        b_update = True
        debug = True
        loss_type = 0
        dataset = 'compas-scores-two-years.csv'
        enumerator = "union"

    conf = SparkConf().setAppName("salary_test").setMaster('local[8]')
    num_partitions = 8
    model_type = "classification"
    label = "is_recid"
    sparkContext = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext)
    dataset_df = sqlContext.read\
        .option("badRecordsPath", "/tmp/badRecordsPath")\
        .csv(dataset, header='true',
                                     inferSchema='true')
    # fileRDD = sparkContext.textFile(dataset, num_partitions)
    # header = fileRDD.first()
    # head_split = header.split(",")
    # fileRDD = fileRDD.filter(lambda line: line != header)
    # data = fileRDD.map(lambda row: row.split(","))
    # dataset_df = sqlContext.createDataFrame(data, head_split)

    cat_features = ["sex", "age_cat", "race", "decile_score10", "c_charge_degree", "c_charge_desc"]
    stages = []
    dataset_df = dataset_df.withColumn("id", sf.monotonically_increasing_id())
    dataset_df = dataset_df.withColumn('target', dataset_df[label].cast("int"))
    dataset_df = dataset_df.withColumn('model_type', sf.lit(1))
    for feature in cat_features:
        string_indexer = StringIndexer(inputCol=feature, outputCol=feature + "_index").setHandleInvalid("skip")
        encoder = OneHotEncoderEstimator(inputCols=[string_indexer.getOutputCol()], outputCols=[feature + "_vec"])
        encoder.setDropLast(False)
        stages += [string_indexer, encoder]
    assembler_inputs = [feature + "_vec" for feature in cat_features]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs")
    stages += [assembler]
    assembler_final = VectorAssembler(inputCols=["assembled_inputs"], outputCol="features")
    stages += [assembler_final]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(dataset_df)
    dataset_transformed = pipeline_model.transform(dataset_df)
    df_transform_fin = dataset_transformed.select('id', 'features', 'target', 'model_type').toPandas()

    cat = 0
    counter = 0
    decode_dict = {}
    for feature in cat_features:
        colIdx = dataset_transformed.select(feature, feature + "_index").distinct().rdd.collectAsMap()
        colIdx = {k: v for k, v in sorted(colIdx.items(), key=lambda item: item[1])}
        for item in colIdx:
            decode_dict[counter] = (cat, item, colIdx[item])
            counter = counter + 1
        cat = cat + 1

    train, test = train_test_split(df_transform_fin, test_size=0.3, random_state=0)
    train_df = sqlContext.createDataFrame(train)
    test_df = sqlContext.createDataFrame(test)

    rf = RandomForestClassifier(featuresCol='features', labelCol="target", numTrees=10)
    rf_model = rf.fit(train_df)
    predictions = rf_model.transform(test_df)
    # Select example rows to display.
    predictions.select("id", "features", "target", "prediction", "model_type")
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    loss = 1.0 - accuracy
    pred_df_fin = predictions.withColumn('error',
                                         spark_utils.calc_loss(predictions["target"], predictions['prediction'],
                                                               predictions['model_type']))
    predictions = pred_df_fin.select('id', 'features', 'error').repartition(num_partitions)
    all_features = range(predictions.toPandas().values[0][0])

    if enumerator == "join":
        join_data_parallel.parallel_process(all_features, predictions, loss, sparkContext, debug=debug, alpha=alpha, k=k,
                                            w=w, loss_type=loss_type)
    elif enumerator == "union":
        union_data_parallel.parallel_process(all_features, predictions, loss, sparkContext, debug=debug, alpha=alpha, k=k,
                                             w=w, loss_type=loss_type)
