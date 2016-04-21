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
  final val scriptPath = "MultiLogReg.dml"
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
  override def fit(df: DataFrame): LogisticRegressionModel = {
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, df, mcXin, false, "features")
    val yin = df.select("label").rdd.map { _.apply(0).toString() }

    val mloutput = {
      val paramsMap: Map[String, String] = Map(
        "icpt" -> this.getIcpt.toString(),
        "reg" -> this.getRegParam.toString(),
        "tol" -> this.getTol.toString,
        "moi" -> this.getMaxOuterIte.toString,
        "mii" -> this.getMaxInnerIter.toString,

        "X" -> " ",
        "Y" -> " ",
        "B" -> " ")
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("Y_vec", yin, "csv");
      ml.registerOutput("B_out");
      ml.executeScript(ScriptsUtils.getDMLScript(LogisticRegression.scriptPath), paramsMap)
      //ml.execute(ScriptsUtils.resolvePath(LogisticRegression.scriptPath), paramsMap)
    }
    new LogisticRegressionModel("logisticRegression")(mloutput)
  }
}
object LogisticRegressionModel {
  final val scriptPath = "GLM-predict.dml"
}

/**
 * Logistic Regression Scala API
 */

class LogisticRegressionModel(
  override val uid: String)(
    val mloutput: MLOutput) extends Model[LogisticRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with HasMaxInnerIter {
  override def copy(extra: ParamMap): LogisticRegressionModel = {
    val that = new LogisticRegressionModel(uid)(mloutput)
    copyValues(that, extra)
  }
  override def transformSchema(schema: StructType): StructType = schema
  override def transform(df: DataFrame): DataFrame = {
    val ml = new MLContext(df.rdd.sparkContext)

    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(df.rdd.sparkContext, df, mcXin, false, "features")

    val mlscoreoutput = {
      val paramsMap: Map[String, String] = Map(
        "X" -> " ",
        "B" -> " ")
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("B_full", mloutput.getBinaryBlockedRDD("B_out"), mloutput.getMatrixCharacteristics("B_out"));
      ml.registerOutput("means");
      ml.executeScript(ScriptsUtils.getDMLScript(LogisticRegressionModel.scriptPath), paramsMap)
    }

    val prob = mlscoreoutput.getDF(df.sqlContext, "means", true).withColumnRenamed("C1", "probability")

    val mlNew = new MLContext(df.rdd.sparkContext)
    mlNew.registerInput("X", Xin, mcXin);
    mlNew.registerInput("B_full", mloutput.getBinaryBlockedRDD("B_out"), mloutput.getMatrixCharacteristics("B_out"));
    mlNew.registerInput("Prob", mlscoreoutput.getBinaryBlockedRDD("means"), mlscoreoutput.getMatrixCharacteristics("means"));
    mlNew.registerOutput("Prediction");
    mlNew.registerOutput("rawPred");

    val outNew = mlNew.executeScript("Prob = read(\"temp1\"); "
      + "Prediction = rowIndexMax(Prob); "
      + "write(Prediction, \"tempOut\", \"csv\")"
      + "X = read(\"temp2\");"
      + "B_full = read(\"temp3\");"
      + "rawPred = 1 / (1 + exp(- X * t(B_full)) );" // Raw prediction logic: 
      + "write(rawPred, \"tempOut1\", \"csv\")");

    val pred = outNew.getDF(df.sqlContext, "Prediction").withColumnRenamed("C1", "prediction").withColumnRenamed("ID", "ID1")
    val rawPred = outNew.getDF(df.sqlContext, "rawPred", true).withColumnRenamed("C1", "rawPrediction").withColumnRenamed("ID", "ID2")
    var predictionsNProb = prob.join(pred, prob.col("ID").equalTo(pred.col("ID1"))).select("ID", "probability", "prediction")
    predictionsNProb = predictionsNProb.join(rawPred, predictionsNProb.col("ID").equalTo(rawPred.col("ID2"))).select("ID", "probability", "prediction", "rawPrediction")
    val dataset1 = RDDConverterUtils.addIDToDataFrame(df, df.sqlContext, "ID")
    dataset1.join(predictionsNProb, dataset1.col("ID").equalTo(predictionsNProb.col("ID")))
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