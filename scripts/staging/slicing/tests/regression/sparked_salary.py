from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, IndexToString
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SQLContext
import pyspark.sql.functions as sf
from pyspark.sql.functions import udf

from slicing.sparked import sparked_utils, sparked_slicer, sparked_union_slicer


binner = udf(lambda arg: int(arg / 5))

conf = SparkConf().setAppName("salary_test").setMaster('local[2]')
num_partitions = 2
model_type = "regression"
label = 'salary'
sparkContext = SparkContext(conf=conf)
sqlContext = SQLContext(sparkContext)
dataset_df = sqlContext.read.csv('/home/lana/diploma/project/slicing/datasets/s'
                                 'alaries.csv', header='true',
                                 inferSchema='true')
# initializing stages of main transformation pipeline
stages = []
# list of categorical features for further hot-encoding
cat_features = ["rank", "discipline", "sincephd_bin", "service_bin", "sex"]
# removing column with ID field
dataset_df = dataset_df.drop('_c0')
# bining numeric features by local binner udf function (specified for current dataset if needed)
dataset_df = dataset_df.withColumn('sincephd_bin', binner(dataset_df['sincephd']))
dataset_df = dataset_df.withColumn('service_bin', binner(dataset_df['service']))
dataset_df = dataset_df.withColumn('model_type', sf.lit(0))
dataset_df = dataset_df.drop('sincephd', 'service')
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
stages += [assembler_final]
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(dataset_df)
dataset_transformed = pipeline_model.transform(dataset_df)
df_transform_fin = dataset_transformed.select('features', label, 'model_type')
splits = df_transform_fin.randomSplit([0.7, 0.3], seed=1234)
train_df = splits[0]
test_df = splits[1]
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
# trainingSummary = lr_model.summary
eval = lr_model.evaluate(test_df)
f_l2 = eval.meanSquaredError
pred = eval.predictions
pred_df_fin = pred.withColumn('error', sparked_utils.calc_loss(pred[label], pred['prediction'], pred['model_type']))
predictions = pred_df_fin.select('features', 'error').repartition(num_partitions)
converter = IndexToString(inputCol='features', outputCol='cats')
all_features = range(predictions.toPandas().values[0][0].size)
predictions = predictions.collect()
k = 10
SparkContext.broadcast(sparkContext, f_l2)
SparkContext.broadcast(sparkContext, predictions)
SparkContext.broadcast(sparkContext, all_features)
enumerator = "join"
if enumerator == "join":
    sparked_slicer.parallel_process(all_features, predictions, f_l2, sparkContext, debug=True, alpha=6, k=k, w=0.5,
                                    loss_type=0, enumerator="join")
elif enumerator == "union":
    sparked_union_slicer.process(all_features, predictions, f_l2, sparkContext, debug=True, alpha=6, k=k, w=0.5, loss_type=0,
                                 enumerator="union")

