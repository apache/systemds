package org.apache.sysml.api.test.ml

import org.apache.sysml.api.ml.classification.DecisionTreeClassifier
import org.apache.sysml.api.ml.feature.DummyCodeGenerator
import java.io.File
import java.io.File.separator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

object DecisionTreeExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val sparkConf = new SparkConf()
                .setAppName("Decision Tree Exanple ")
                .setMaster("local");
    
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
//    val fPath = this.getClass()
//                    .getResource("/sample_libsvm_data.txt")
//                    .getPath()
    val fPath = "src" + File.separator + "test" + File.separator + "resources" + File.separator + "sample_libsvm_data.txt";

    val data = sqlContext.createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect).toDF("label", "features")
    
    val dummyCoder = new DummyCodeGenerator()
      .setInputCol("label")
      .setOutputCol("dummyCodedLabel")
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val dt = new DecisionTreeClassifier(sc).setLabelCol("dummyCodedLabel")
    
    val pipeline = new Pipeline().setStages(Array(dummyCoder, dt))
    
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxBins, Array(10, 20))
      .build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(4)
    
    val cvModel = crossval.fit(trainingData)
    cvModel.transform(testData).show()
  }
}