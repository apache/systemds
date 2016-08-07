/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api.ml

import java.io.File
import org.apache.sysml.api.{ MLContext, MLOutput }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }
import org.apache.spark.{ SparkContext }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.{ Model, Estimator }
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.{ Params, Param, ParamMap, DoubleParam }
import org.apache.spark.ml.param.shared._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.reflect.ClassTag
import scala.collection.immutable.HashMap
import org.apache.spark.sql.functions.udf
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException

trait HasIcpt extends Params {
  final val icpt: Param[Int] = new Param[Int](this, "icpt", "Intercept presence, shifting and rescaling X columns")
  setDefault(icpt, 0)
  final def getIcpt: Int = $(icpt)
}
trait HasMaxOuterIter extends Params {
  final val maxOuterIter: Param[Int] = new Param[Int](this, "maxOuterIter", "max. number of outer (Newton) iterations")
  setDefault(maxOuterIter, 100)
  final def getMaxOuterIte: Int = $(maxOuterIter)
}
trait HasMaxInnerIter extends Params {
  final val maxInnerIter: Param[Int] = new Param[Int](this, "maxInnerIter", "max. number of inner (conjugate gradient) iterations, 0 = no max")
  setDefault(maxInnerIter, 0)
  final def getMaxInnerIter: Int = $(maxInnerIter)
}
trait HasTol extends Params {
  final val tol: DoubleParam = new DoubleParam(this, "tol", "the convergence tolerance for iterative algorithms")
  setDefault(tol, 0.000001)
  final def getTol: Double = $(tol)
}
trait HasRegParam extends Params {
  final val regParam: DoubleParam = new DoubleParam(this, "tol", "the convergence tolerance for iterative algorithms")
  setDefault(regParam, 0.000001)
  final def getRegParam: Double = $(regParam)
}
object LogisticRegression {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "MultiLogReg.dml"
}

/**
 * Logistic Regression Scala API
 */
class LogisticRegression(override val uid: String, val sc: SparkContext) extends Estimator[LogisticRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with HasMaxInnerIter {

  def setIcpt(value: Int) = set(icpt, value)
  def setMaxOuterIter(value: Int) = set(maxOuterIter, value)
  def setMaxInnerIter(value: Int) = set(maxInnerIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)

  override def copy(extra: ParamMap): LogisticRegression = {
    val that = new LogisticRegression(uid, sc)
    copyValues(that, extra)
  }
  override def transformSchema(schema: StructType): StructType = schema
  
  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): LogisticRegressionModel = {
    val ml = new MLContext(sc)
    val labelMapping = new java.util.HashMap[String, Int]
    val revLabelMapping = new java.util.HashMap[Int, String]
    if(y_mb.getNumColumns != 1) {
      throw new RuntimeException("Expected a column vector for y")
    }
    if(y_mb.isInSparseFormat()) {
      throw new DMLRuntimeException("Sparse block is not implemented for fit")
    }
    else {
      val denseBlock = y_mb.getDenseBlock()
      var id:Int = 1
      for(i <- 0 until denseBlock.length) {
        val v = denseBlock(i).toString()
        if(!labelMapping.containsKey(v)) {
          labelMapping.put(v, id)
          revLabelMapping.put(id, v)
          id += 1
        }
        denseBlock.update(i, labelMapping.get(v))
      }  
    }
    
    val mloutput = {
      ml.registerInput("X", X_mb);
      ml.registerInput("Y_vec", y_mb);
      ml.registerOutput("B_out");
      ml.executeScript(ScriptsUtils.getDMLScript(LogisticRegression.scriptPath), getParamMap())
    }
    new LogisticRegressionModel("logisticRegression")(mloutput, revLabelMapping, sc)
  }
  
  def getParamMap():Map[String, String] = {
    Map(
        "icpt" -> this.getIcpt.toString(),
        "reg" -> this.getRegParam.toString(),
        "tol" -> this.getTol.toString,
        "moi" -> this.getMaxOuterIte.toString,
        "mii" -> this.getMaxInnerIter.toString,

        "X" -> " ",
        "Y" -> " ",
        "B" -> " ")
  }
  override def fit(df: DataFrame): LogisticRegressionModel = {
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, df, mcXin, false, "features")
    val temp = df.select("label").distinct.rdd.map(_.apply(0).toString).collect()
    val labelMapping = new java.util.HashMap[String, Int]
    val revLabelMapping = new java.util.HashMap[Int, String]
    for(i <- 0 until temp.length) {
      labelMapping.put(temp(i), i+1)
      revLabelMapping.put(i+1, temp(i))
    }
    val yin = df.select("label").rdd.map( x => labelMapping.get(x.apply(0).toString).toString )
    val mloutput = {
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("Y_vec", yin, "csv");
      ml.registerOutput("B_out");
      ml.executeScript(ScriptsUtils.getDMLScript(LogisticRegression.scriptPath), getParamMap())
    }
    new LogisticRegressionModel("logisticRegression")(mloutput, revLabelMapping, sc)
  }
}
object LogisticRegressionModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "GLM-predict.dml"
}

class LogisticRegressionModelSerializableData(val labelMapping: java.util.HashMap[Int, String]) extends Serializable {
 def mapLabelStr(x:Double):String = {
   if(labelMapping.containsKey(x.toInt))
     labelMapping.get(x.toInt)
   else
     throw new RuntimeException("Incorrect label mapping")
 }
 def mapLabelDouble(x:Double):Double = {
   if(labelMapping.containsKey(x.toInt))
     labelMapping.get(x.toInt).toDouble
   else
     throw new RuntimeException("Incorrect label mapping")
 }
 val mapLabel_udf =  {
      try {
        val it = labelMapping.values().iterator()
        while(it.hasNext()) {
          it.next().toDouble
        }
        udf(mapLabelDouble _)
      } catch {
        case e: Exception => udf(mapLabelStr _)
      }
    }
}
/**
 * Logistic Regression Scala API
 */

class LogisticRegressionModel(
  override val uid: String)(
    val mloutput: MLOutput, val labelMapping: java.util.HashMap[Int, String], val sc: SparkContext) extends Model[LogisticRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with HasMaxInnerIter {
  override def copy(extra: ParamMap): LogisticRegressionModel = {
    val that = new LogisticRegressionModel(uid)(mloutput, labelMapping, sc)
    copyValues(that, extra)
  }
  var outputRawPredictions = true
  def setOutputRawPredictions(outRawPred:Boolean): Unit = { outputRawPredictions = outRawPred }
  override def transformSchema(schema: StructType): StructType = schema
   
  def transform(X: MatrixBlock): MatrixBlock = {
    if(outputRawPredictions) {
      throw new RuntimeException("Outputting raw prediction is not supported")
    }
    else {
      val isSingleNode = true
      val ret = PredictionUtils.computePredictedClassLabelsFromProbability(
          PredictionUtils.doGLMPredict(isSingleNode, null, X, sc, mloutput, "B_out", getPredictParams), 
          isSingleNode, sc).getMatrixBlock("Prediction");
      if(ret.getNumColumns != 1) {
        throw new RuntimeException("Expected predicted label to be a column vector")
      }
      if(ret.isInSparseFormat()) {
        throw new RuntimeException("Since predicted label is a column vector, expected it to be in dense format")
      }
      else {
        updateLabels(true, null, ret, null)
      }
      return ret
    }
  }
  
  def updateLabels(isSingleNode:Boolean, df:DataFrame, X: MatrixBlock, labelColName:String): DataFrame = {
    if(isSingleNode) {
      for(i <- 0 until X.getNumRows) {
        val v:Int = X.getValue(i, 0).toInt
        if(labelMapping.containsKey(v)) {
          X.setValue(i, 0, labelMapping.get(v).toDouble)
        }
        else {
          throw new RuntimeException("No mapping found for " + v + " in " + labelMapping.toString())
        }
      }
      return null
    }
    else {
      val serObj = new LogisticRegressionModelSerializableData(labelMapping)
      return df.withColumn(labelColName, serObj.mapLabel_udf(df(labelColName)))
               .withColumnRenamed(labelColName, "prediction")
    }
  }
  
  
  override def transform(df: DataFrame): DataFrame = {
    val isSingleNode = false
    val glmPredOut = PredictionUtils.doGLMPredict(isSingleNode, df, null, sc, mloutput, "B_out", getPredictParams())
    val predLabelOut = PredictionUtils.computePredictedClassLabelsFromProbability(glmPredOut, isSingleNode, sc)
    val predictedDF = updateLabels(isSingleNode, predLabelOut.getDF(df.sqlContext, "Prediction"), null, "C1").select("ID", "prediction")
    val prob = glmPredOut.getDF(df.sqlContext, "means", true).withColumnRenamed("C1", "probability").select("ID", "probability")
    val dataset = RDDConverterUtils.addIDToDataFrame(df, df.sqlContext, "ID")
    
    if(outputRawPredictions) {
      // Not supported: rawPred = 1 / (1 + exp(- X * t(B_full)) );
    }
    return PredictionUtils.joinUsingID(dataset, PredictionUtils.joinUsingID(prob, predictedDF))
  }
  
  def getPredictParams(): Map[String, String] = {
    Map("X" -> " ",
        "B" -> " ",
        "dfam" -> "3")
  }
}

/**
 * Example code for Logistic Regression
 */
object LogisticRegressionExample {
  import org.apache.spark.{ SparkConf, SparkContext }
  import org.apache.spark.sql.types._
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.regression.LabeledPoint

  def main(args: Array[String]) = {
    val sparkConf: SparkConf = new SparkConf();
    val sc: SparkContext = new SparkContext("local", "TestLocal", sparkConf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc);

    import sqlContext.implicits._
    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.5, 2.2)),
      LabeledPoint(2.0, Vectors.dense(1.6, 0.8, 3.6)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 2.3))))
    val lr = new LogisticRegression("log", sc)
    val lrmodel = lr.fit(training.toDF)
    lrmodel.mloutput.getDF(sqlContext, "B_out").show()

    val testing = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.5, 2.2)),
      LabeledPoint(2.0, Vectors.dense(1.6, 0.8, 3.6)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 2.3))))

    lrmodel.transform(testing.toDF).show
  }
}