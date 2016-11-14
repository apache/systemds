package org.apache.sysml.api.test.ml

import org.apache.sysml.api.ml.classification.RandomForestClassifier
import org.apache.sysml.api.ml.feature.DummyCodeGenerator
import java.io.File.separator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

object RandomForestExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val sparkConf = new SparkConf()
                .setAppName("LogisticRegression ")
                .setMaster("local");
    
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
    val fPath = this.getClass()
                    .getResource("/sample_libsvm_data.txt")
                    .getPath()
    val data = sqlContext.createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect).toDF("label", "features")
    
    val dummyCoder = new DummyCodeGenerator()
      .setInputCol("label")
      .setOutputCol("dummyCodedLabel")
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val rf = new RandomForestClassifier(sc).setLabelCol("dummyCodedLabel")
    
    val pipeline = new Pipeline().setStages(Array(dummyCoder, rf))
    
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxBins, Array(10, 20))
      .build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(4)
    
    val cvModel = crossval.fit(trainingData)
    cvModel.transform(testData).show()
  }
}