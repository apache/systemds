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
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, IndexToString
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SQLContext
import pyspark.sql.functions as sf
from pyspark.sql.functions import udf
from sklearn.model_selection import train_test_split

from slicing.spark_modules import spark_utils, spark_slicer, spark_union_slicer


binner = udf(lambda arg: int(arg / 5))


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
        w = 0.5
        alpha = 10
        b_update = True
        debug = True
        loss_type = 0
        enumerator = "union"
        dataset = 'slicing/datasets/salaries.csv'
        # dataset = 'slicing/datasets/parallel_attr/salaries/attr3000.csv'

    conf = SparkConf().setAppName("salary_test").setMaster('local[4]')
    num_partitions = 4
    model_type = "regression"
    label = 'salary'
    sparkContext = SparkContext(conf=conf)
    sqlContext = SQLContext(sparkContext)
    dataset_df = sqlContext.read.csv(dataset, header='true',
                                     inferSchema='true')
    # initializing stages of main transformation pipeline
    stages = []
    # list of categorical features for further hot-encoding
    cat_features = ["rank", "discipline", "sincephd_bin", "service_bin", "sex"]  # base
    # cat_features = ["rank", "discipline", "sex"]  # 10
    # cat_features = ["rank", "discipline", "sincephd", "sex"]  # 50
    # cat_features = ["rank", "discipline", "sex", "company", "country", "city", "start", "dept", "card"]  # 100
    # cat_features = ["rank", "discipline", "sex", "company", "country", "city", "skills", "language", "tz", "card",
    #                  "uni", "dept", "race", "code", "job", "size", "previous", "stock"]  # 1013
    # cat_features = ["rank", "discipline", "sex", "company", "country", "city", "skills", "language", "tz", "card",
    #                     "uni", "dept", "origin", "children", "bmi", "code", "job", "size", "previous", "stock", "market",
    #                     "freq", "smoker", "region", "WorkClass", "Education", "EducationNum", "MaritalStatus",
    #                     "Occupation", "Relationship", "color", "sepal_length", "sepal_width", "petal_length",
    #                     "petal_width", "variety", "symboling", "normalized-losses", "make",	"fuel-type", "aspiration",
    #                     "num-of-doors",	"body-style", "drive-wheels", "engine-location", "wheel-base", 	"length", "width",
    #                     "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore",
    #                     "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",	"price",
    #                     "codeNumber", "clump", "cellsize", "cellshape", "adhesion", "epitel", "nuclei", "chromatin",
    #                     "nucleoli", "mitoses", "class", "elevation", "aspect", "slope", "distToHydr"]

#     cat_features = ["rank", "discipline", "sex", "company", "country", "city", "skills", "language", "tz", "card",
#                     "uni", "dept", "origin", "children", "bmi", "code", "job", "size", "previous", "stock", "market",
#                     "freq", "smoker", "region", "WorkClass", "Education", "EducationNum", "MaritalStatus",
#                     "Occupation", "Relationship", "color", "sepal_length", "sepal_width", "petal_length",
#                     "petal_width", "variety", "symboling", "normalized-losses", "make",	"fuel-type", "aspiration",
#                     "num-of-doors",	"body-style", "drive-wheels", "engine-location", "wheel-base", 	"length", "width",
#                     "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore",
#                     "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg",	"price",
#                     "codeNumber", "clump", "cellsize", "cellshape", "adhesion", "epitel", "nuclei", "chromatin",
#                     "nucleoli", "mitoses", "class", "elevation", "aspect", "slope", "distToHydr", "distToRoad",
#                     "hillshadeMorning", "hillSahdeNoon", "hillSadeAfternoon", "distToFire", "DSName", "T", "N", "p",
#                     "k", "Bin", "Cost",	"Sdratio", "correl", "cancor1",	"cancor2", "fract1", "fract2", "skewness",
#                     "kurtosis",	"Hc", "Hx", "Mcx", "EnAttr", "NSRation", "Alg", "error"]   # 3000

    # lines till 87 only for base case
    # removing column with ID field
    # dataset_df = dataset_df.drop('_c0')
    # bining numeric features by local binner udf function (specified for current dataset if needed)
    dataset_df = dataset_df.withColumn('sincephd_bin', binner(dataset_df['sincephd']))
    dataset_df = dataset_df.withColumn('service_bin', binner(dataset_df['service']))
    dataset_df = dataset_df.drop('sincephd', 'service')
    dataset_df = dataset_df.withColumn('model_type', sf.lit(0))
    # hot encoding categorical features
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
    df_transform_fin = dataset_transformed.select('features', label, 'model_type').toPandas()
    train, test = train_test_split(df_transform_fin, test_size=0.3, random_state=0)
    train_df = sqlContext.createDataFrame(train)
    test_df = sqlContext.createDataFrame(test)
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
    lr = LinearRegression(featuresCol='features', labelCol=label, maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_df)
    eval = lr_model.evaluate(test_df)
    f_l2 = eval.meanSquaredError
    pred = eval.predictions
    pred_df_fin = pred.withColumn('error', spark_utils.calc_loss(pred[label], pred['prediction'], pred['model_type']))
    predictions = pred_df_fin.select('features', 'error').repartition(num_partitions)
    converter = IndexToString(inputCol='features', outputCol='cats')
    all_features = list(decode_dict.keys())
    predictions = predictions.collect()
    if enumerator == "join":
        spark_slicer.parallel_process(all_features, predictions, f_l2, sparkContext, debug=debug, alpha=alpha, k=k, w=w,
                                      loss_type=loss_type, enumerator=enumerator)
    elif enumerator == "union":
        spark_union_slicer.process(all_features, predictions, f_l2, sparkContext, debug=debug, alpha=alpha, k=k, w=w,
                                   loss_type=loss_type, enumerator=enumerator)
