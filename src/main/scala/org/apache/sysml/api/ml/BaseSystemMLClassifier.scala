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
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.{ Params, Param, ParamMap, DoubleParam }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt, RDDConverterUtils }
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.spark.sql._

trait HasLaplace extends Params {
  final val laplace: Param[Double] = new Param[Double](this, "laplace", "Laplace smoothing specified by the user to avoid creation of 0 probabilities.")
  setDefault(laplace, 1.0)
  final def getLaplace: Double = $(laplace)
}
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

trait BaseSystemMLEstimator {
  
  def transformSchema(schema: StructType): StructType = schema
  
  // Returns the script and variables for X and y
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)
  
  def toDouble(i:Int): java.lang.Double = {
    double2Double(i.toDouble)
  }
  
  def toDouble(d:Double): java.lang.Double = {
    double2Double(d)
  }
}

trait BaseSystemMLEstimatorModel {
  def toDouble(i:Int): java.lang.Double = {
    double2Double(i.toDouble)
  }
  def toDouble(d:Double): java.lang.Double = {
    double2Double(d)
  }
  
  def transformSchema(schema: StructType): StructType = schema
  
  // Returns the script and variable for X
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String)
}

trait BaseSystemMLClassifier extends BaseSystemMLEstimator {
  
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock, sc: SparkContext): (MLResults, java.util.HashMap[Int, String]) = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    val revLabelMapping = new java.util.HashMap[Int, String]
    PredictionUtils.fillLabelMapping(y_mb, revLabelMapping)
    val ret = getTrainingScript(isSingleNode)
    val script = ret._1.in(ret._2, X_mb).in(ret._3, y_mb)
    (ml.execute(script), revLabelMapping)
  }
  
  def fit(df: ScriptsUtils.SparkDataType, sc: SparkContext): (MLResults, java.util.HashMap[Int, String]) = {
    val isSingleNode = false
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(sc, df.asInstanceOf[DataFrame], mcXin, false, true)
    val revLabelMapping = new java.util.HashMap[Int, String]
    val yin = PredictionUtils.fillLabelMapping(df, revLabelMapping)
    val ret = getTrainingScript(isSingleNode)
    val Xbin = new BinaryBlockMatrix(Xin, mcXin)
    val script = ret._1.in(ret._2, Xbin).in(ret._3, yin)
    (ml.execute(script), revLabelMapping)
  }
}

trait BaseSystemMLClassifierModel extends BaseSystemMLEstimatorModel {
  
  def transform(X: MatrixBlock, mloutput: MLResults, labelMapping: java.util.HashMap[Int, String], sc: SparkContext, probVar:String): MatrixBlock = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    val script = getPredictionScript(mloutput, isSingleNode)
    val modelPredict = ml.execute(script._1.in(script._2, X))
    val ret = PredictionUtils.computePredictedClassLabelsFromProbability(modelPredict, isSingleNode, sc, probVar)
              .getBinaryBlockMatrix("Prediction").getMatrixBlock
              
    if(ret.getNumColumns != 1) {
      throw new RuntimeException("Expected predicted label to be a column vector")
    }
    PredictionUtils.updateLabels(isSingleNode, null, ret, null, labelMapping)
    return ret
  }
  
  def transform(df: ScriptsUtils.SparkDataType, mloutput: MLResults, labelMapping: java.util.HashMap[Int, String], sc: SparkContext, 
      probVar:String, outputProb:Boolean=true): DataFrame = {
    val isSingleNode = false
    val ml = new MLContext(sc)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(df.rdd.sparkContext, df.asInstanceOf[DataFrame], mcXin, false, true)
    val script = getPredictionScript(mloutput, isSingleNode)
    val Xin_bin = new BinaryBlockMatrix(Xin, mcXin)
    val modelPredict = ml.execute(script._1.in(script._2, Xin_bin))
    val predLabelOut = PredictionUtils.computePredictedClassLabelsFromProbability(modelPredict, isSingleNode, sc, probVar)
    val predictedDF = PredictionUtils.updateLabels(isSingleNode, predLabelOut.getDataFrame("Prediction"), null, "C1", labelMapping).select("__INDEX", "prediction")
    if(outputProb) {
      val prob = modelPredict.getDataFrame(probVar, true).withColumnRenamed("C1", "probability").select("__INDEX", "probability")
      val dataset = RDDConverterUtilsExt.addIDToDataFrame(df.asInstanceOf[DataFrame], df.sqlContext, "__INDEX")
      return PredictionUtils.joinUsingID(dataset, PredictionUtils.joinUsingID(prob, predictedDF))
    }
    else {
      val dataset = RDDConverterUtilsExt.addIDToDataFrame(df.asInstanceOf[DataFrame], df.sqlContext, "__INDEX")
      return PredictionUtils.joinUsingID(dataset, predictedDF)
    }
    
  }
}