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

import org.apache.spark.rdd.RDD
import java.io.File
import org.apache.spark.SparkContext
import org.apache.spark.ml.{ Model, Estimator }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.{ Params, Param, ParamMap, DoubleParam }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._

object LogisticRegression {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "MultiLogReg.dml"
}

/**
 * Logistic Regression Scala API
 */
class LogisticRegression(override val uid: String, val sc: SparkContext) extends Estimator[LogisticRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with HasMaxInnerIter with BaseSystemMLClassifier {

  def setIcpt(value: Int) = set(icpt, value)
  def setMaxOuterIter(value: Int) = set(maxOuterIter, value)
  def setMaxInnerIter(value: Int) = set(maxInnerIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)

  override def copy(extra: ParamMap): LogisticRegression = {
    val that = new LogisticRegression(uid, sc)
    copyValues(that, extra)
  }
  

  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): LogisticRegressionModel = {
    mloutput = baseFit(X_mb, y_mb, sc)
    new LogisticRegressionModel(this)
  }
  
  def fit(df: ScriptsUtils.SparkDataType): LogisticRegressionModel = {
    mloutput = baseFit(df, sc)
    new LogisticRegressionModel(this)
  }
  
  
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(LogisticRegression.scriptPath))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$B", " ")
      .in("$icpt", toDouble(getIcpt))
      .in("$reg", toDouble(getRegParam))
      .in("$tol", toDouble(getTol))
      .in("$moi", toDouble(getMaxOuterIte))
      .in("$mii", toDouble(getMaxInnerIter))
      .out("B_out")
    (script, "X", "Y_vec")
  }
}
object LogisticRegressionModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "GLM-predict.dml"
}

/**
 * Logistic Regression Scala API
 */

class LogisticRegressionModel(override val uid: String)(
    estimator: LogisticRegression, val sc: SparkContext) 
    extends Model[LogisticRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with HasMaxInnerIter with BaseSystemMLClassifierModel {
  override def copy(extra: ParamMap): LogisticRegressionModel = {
    val that = new LogisticRegressionModel(uid)(estimator, sc)
    copyValues(that, extra)
  }
  var outputRawPredictions = true
  def setOutputRawPredictions(outRawPred:Boolean): Unit = { outputRawPredictions = outRawPred }
  def this(estimator:LogisticRegression) =  {
  	this("model")(estimator, estimator.sc)
  }
  def getPredictionScript(isSingleNode:Boolean): (Script, String) =
    PredictionUtils.getGLMPredictionScript(estimator.mloutput.getMatrix("B_out"), isSingleNode, 3)
  
  def baseEstimator():BaseSystemMLEstimator = estimator
  def modelVariables():List[String] = List[String]("B_out")
  
  def transform(X: MatrixBlock): MatrixBlock = baseTransform(X, sc, "means")
  def transform_probability(X: MatrixBlock): MatrixBlock = baseTransformProbability(X, sc, "means")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, sc, "means")
}

/**
 * Example code for Logistic Regression
 */
object LogisticRegressionExample {
  import org.apache.spark.{ SparkConf, SparkContext }
  import org.apache.spark.sql._
  import org.apache.spark.sql.types._
  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.feature.LabeledPoint

  def main(args: Array[String]) = {
    val sparkSession = SparkSession.builder().master("local").appName("TestLocal").getOrCreate();
    val sc: SparkContext = sparkSession.sparkContext;

    import sparkSession.implicits._
    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.5, 2.2)),
      LabeledPoint(2.0, Vectors.dense(1.6, 0.8, 3.6)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 2.3))))
    val lr = new LogisticRegression("log", sc)
    val lrmodel = lr.fit(training.toDF)
    // lrmodel.mloutput.getDF(sparkSession, "B_out").show()

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
