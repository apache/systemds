package org.apache.sysml.api.ml

import org.apache.sysml.api.ml.classification._
import org.apache.sysml.api.ml.regression._

import org.apache.spark.Logging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.scalatest.FunSuite
import org.scalatest.Matchers


class MLPipelineTest extends FunSuite with WrapperSparkContext with Matchers with Logging {
  test("run simple logistic regression") {
    val training = sqlContext.createDataFrame(Seq(
      (1.0, 2.0, Vectors.dense(1.0)),
      (2.0, 2.0, Vectors.dense(0.0))))
      .toDF("label", "weight", "features")

    val lr = new org.apache.sysml.api.ml.classification.LogisticRegression(sc)
    val model = lr.fit(training)

    val test0 = sqlContext.createDataFrame(Seq((1.0, Vectors.dense(-1.0)))).toDF("id", "features")
    val result0 = model.transform(test0).head()(3)
    result0 shouldBe 2.0

    val test1 = sqlContext.createDataFrame(Seq((1.0, Vectors.dense(1.0)))).toDF("id", "features")
    val result1 = model.transform(test1).head()(3)
    result1 shouldBe 1.0
  }
  
  test("run simple L2 norm svm") {
    val training = sqlContext.createDataFrame(Seq(
      (1.0, 2.0, Vectors.dense(1.0)),
      (0.0, 2.0, Vectors.dense(0.0))))
      .toDF("label", "weight", "features")
      
    val l_svm = new L2NormSVMClassifier(sc)
    val model = l_svm.fit(training) 
    
    val test0 = sqlContext.createDataFrame(Seq((1.0, Vectors.dense(-1.0)))).toDF("id", "features")
    val result0 = model.transform(test0)
    
    result0.show
  }
  
  test("run pipeline logistic regression") {
    val training = sqlContext.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0))).toDF("id", "text", "label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    val lr = new org.apache.sysml.api.ml.classification.LogisticRegression(sc)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100))
      .addGrid(lr.regParam, Array(0.1)).build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)
    val cvModel = crossval.fit(training)

    val test = sqlContext.createDataFrame(Seq(
      (12L, "spark i j k"),
      (13L, "l m n"),
      (14L, "mapreduce spark"),
      (15L, "apache hadoop"))).toDF("id", "text")

    cvModel.transform(test).show
  }
  
  test("run pipeline naive bayes") {
    val training = sqlContext.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 2.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 2.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 2.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 2.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 2.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 2.0))).toDF("id", "text", "label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val nb = new org.apache.sysml.api.ml.classification.NaiveBayes(sc)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, nb))
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100))
      .addGrid(nb.laplaceCorrection, Array(1, 2)).build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)
    val cvModel = crossval.fit(training)

    val test = sqlContext.createDataFrame(Seq(
      (12L, "spark i j k"),
      (13L, "l m n"),
      (14L, "mapreduce spark"),
      (15L, "apache hadoop"))).toDF("id", "text")

    cvModel.transform(test).show
  }

  test("run pipeline linear regression CG") {
    val fPath = getClass().getResource("/sample_libsvm_data.txt").getPath()
    val data = sqlContext.createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect).toDF("label", "features")
    
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)
    
    val lr = new LinearRegressionCG(sc)
    
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
    model.transform(test).show
  }
  
  test("run pipeline linear regression DS") {
    val fPath = getClass().getResource("/sample_libsvm_data.txt").getPath()
    val data = sqlContext.createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect).toDF("label", "features")
    
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
    model.transform(test).show
  }

}