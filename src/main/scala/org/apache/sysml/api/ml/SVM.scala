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
import org.apache.spark.SparkContext
import org.apache.spark.ml.{ Model, Estimator }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.ParamMap
import org.apache.sysml.api.{ MLContext, MLOutput }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }

object SVM {
  final val scriptPathBinary = "scripts" + File.separator + "algorithms" + File.separator + "l2-svm.dml"
  final val scriptPathMulticlass = "scripts" + File.separator + "algorithms" + File.separator + "m-svm.dml"
}

class SVM (override val uid: String, val sc: SparkContext, val isMultiClass:Boolean=false) extends Estimator[SVMModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter {

  def setIcpt(value: Int) = set(icpt, value)
  def setMaxIter(value: Int) = set(maxOuterIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)
  
  def setModelParams(m:SVMModel):SVMModel = {
    m.setIcpt(this.getIcpt).setMaxIter(this.getMaxOuterIte).setRegParam(this.getRegParam).setTol(this.getTol)
  }
  
  override def copy(extra: ParamMap): Estimator[SVMModel] = {
    val that = new SVM(uid, sc, isMultiClass)
    copyValues(that, extra)
  }
  def transformSchema(schema: StructType): StructType = schema
  
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): SVMModel = {
    val ml = new MLContext(sc)
    val revLabelMapping = new java.util.HashMap[Int, String]
    PredictionUtils.fillLabelMapping(y_mb, revLabelMapping)
    if(y_mb.getNumColumns != 1) {
      throw new RuntimeException("Expected a column vector for y")
    }
    val mloutput = {
      ml.registerInput("X", X_mb);
      ml.registerInput("Y", y_mb);
      ml.registerOutput("w");
      if(isMultiClass)
        ml.executeScript(ScriptsUtils.getDMLScript(SVM.scriptPathMulticlass), getParamMap())
      else {
        ml.executeScript(ScriptsUtils.getDMLScript(SVM.scriptPathBinary), getParamMap())
      }
    }
    setModelParams(new SVMModel("svm")(mloutput, sc, isMultiClass, revLabelMapping))
  }
  
  def getParamMap(): Map[String, String] = {
    Map(  "icpt" -> this.getIcpt.toString(),
          "reg" -> this.getRegParam.toString(),
          "tol" -> this.getTol.toString,
          "maxiter" -> this.getMaxOuterIte.toString,
          "X" -> " ",
          "Y" -> " ",
          "model" -> " ",
          "Log" -> " ")
  }
  
  def fit(df: DataFrame): SVMModel = {
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, df, mcXin, false, "features")
    val revLabelMapping = new java.util.HashMap[Int, String]
    val yin = PredictionUtils.fillLabelMapping(df, revLabelMapping)
    val mloutput = {
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("Y", yin, "csv");
      ml.registerOutput("w");
      if(isMultiClass)
        ml.executeScript(ScriptsUtils.getDMLScript(SVM.scriptPathMulticlass), getParamMap())
      else {
        ml.executeScript(ScriptsUtils.getDMLScript(SVM.scriptPathBinary), getParamMap())
      }
    }
    setModelParams(new SVMModel("svm")(mloutput, sc, isMultiClass, revLabelMapping))
  }
  
}

object SVMModel {
  final val predictionScriptPathBinary = "scripts" + File.separator + "algorithms" + File.separator + "l2-svm-predict.dml"
  final val predictionScriptPathMulticlass = "scripts" + File.separator + "algorithms" + File.separator + "m-svm-predict.dml"
}

class SVMModel (override val uid: String)(val mloutput: MLOutput, val sc: SparkContext, val isMultiClass:Boolean, val labelMapping: java.util.HashMap[Int, String]) extends Model[SVMModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter {
  override def copy(extra: ParamMap): SVMModel = {
    val that = new SVMModel(uid)(mloutput, sc, isMultiClass, labelMapping)
    copyValues(that, extra)
  }
  
  def setIcpt(value: Int) = set(icpt, value)
  def setMaxIter(value: Int) = set(maxOuterIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)
  
  override def transformSchema(schema: StructType): StructType = schema
  
  def transform(df: DataFrame): DataFrame = {
    val ml = new MLContext(sc)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(df.rdd.sparkContext, df, mcXin, false, "features")
    ml.registerInput("X", Xin, mcXin);
    ml.registerOutput("scores");
    val glmPredOut = {
      if(isMultiClass) {
        ml.registerInput("W", mloutput.getBinaryBlockedRDD("w"), mloutput.getMatrixCharacteristics("w"));
        ml.executeScript(ScriptsUtils.getDMLScript(SVMModel.predictionScriptPathMulticlass), getPredictParams())
      }
      else {
        ml.registerInput("w", mloutput.getBinaryBlockedRDD("w"), mloutput.getMatrixCharacteristics("w"));
        ml.executeScript(ScriptsUtils.getDMLScript(SVMModel.predictionScriptPathBinary), getPredictParams())
      }
    }
    val isSingleNode = false
    val predLabelOut = PredictionUtils.computePredictedClassLabelsFromProbability(glmPredOut, isSingleNode, sc, "scores")
    val predictedDF = PredictionUtils.updateLabels(isSingleNode, predLabelOut.getDF(df.sqlContext, "Prediction"), null, "C1", labelMapping).select("ID", "prediction")
    val dataset = RDDConverterUtils.addIDToDataFrame(df, df.sqlContext, "ID")
    return PredictionUtils.joinUsingID(dataset, predictedDF)
  }
  
  def transform(X: MatrixBlock): MatrixBlock =  {
    val ml = new MLContext(sc)
    ml.registerInput("X", X);
    ml.registerInput("w", mloutput.getMatrixBlock("w"), mloutput.getMatrixCharacteristics("w"));
    ml.registerOutput("scores");
    val glmPredOut = {
      if(isMultiClass) {
        ml.registerInput("W", mloutput.getMatrixBlock("w"), mloutput.getMatrixCharacteristics("w"));
        ml.executeScript(ScriptsUtils.getDMLScript(SVMModel.predictionScriptPathMulticlass), getPredictParams())
      }
      else { 
        ml.registerInput("w", mloutput.getMatrixBlock("w"), mloutput.getMatrixCharacteristics("w"));
        ml.executeScript(ScriptsUtils.getDMLScript(SVMModel.predictionScriptPathBinary), getPredictParams())
      }
    }
    val isSingleNode = true
    val ret = PredictionUtils.computePredictedClassLabelsFromProbability(glmPredOut, isSingleNode, sc, "scores").getMatrixBlock("Prediction");
    if(ret.getNumColumns != 1) {
      throw new RuntimeException("Expected predicted label to be a column vector")
    }
    PredictionUtils.updateLabels(true, null, ret, null, labelMapping)
    return ret
  }
  
  
  def getPredictParams(): Map[String, String] = {
    Map(  "icpt" -> this.getIcpt.toString(),
          "reg" -> this.getRegParam.toString(),
          "tol" -> this.getTol.toString,
          "maxiter" -> this.getMaxOuterIte.toString,
          "X" -> " ",
          "Y" -> " ",
          "model" -> " ",
          "Log" -> " ")
  }

}