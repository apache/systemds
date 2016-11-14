package org.apache.sysml.api.test.ml

import org.apache.sysml.api.ml.regression.LinearRegressionDS
import java.io.File.separator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

object LinearRegressionDSExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val sparkConf = new SparkConf()
      .setAppName("LInear SVM Exanple ")
      .setMaster("local");

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
    val fPath = this.getClass()
                    .getResource("/sample_libsvm_data.txt")
                    .getPath()
    val data = sqlContext
      .createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect)
      .toDF("label", "features")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)
    
    val lr = new LinearRegressionDS(sc)
    
    val pipeline = new Pipeline().setStages(Array(lr))
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(10)
    val model = crossval.fit(training)
    model.transform(test)
      .select("label", "features", "prediction")
      .show
  }
}