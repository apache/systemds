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

object SVM {
  final val scriptPathBinary = "scripts" + File.separator + "algorithms" + File.separator + "l2-svm.dml"
  final val scriptPathMulticlass = "scripts" + File.separator + "algorithms" + File.separator + "m-svm.dml"
}

class SVM (override val uid: String, val sc: SparkContext, val isMultiClass:Boolean=false) extends Estimator[SVMModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with BaseSystemMLClassifier {

  def setIcpt(value: Int) = set(icpt, value)
  def setMaxIter(value: Int) = set(maxOuterIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)
  
  override def copy(extra: ParamMap): Estimator[SVMModel] = {
    val that = new SVM(uid, sc, isMultiClass)
    copyValues(that, extra)
  }
  
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(if(isMultiClass) SVM.scriptPathMulticlass else SVM.scriptPathBinary))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$model", " ")
      .in("$Log", " ")
      .in("$icpt", toDouble(getIcpt))
      .in("$reg", toDouble(getRegParam))
      .in("$tol", toDouble(getTol))
      .in("$maxiter", toDouble(getMaxOuterIte))
      .out("w")
    (script, "X", "Y")
  }
  
  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): SVMModel = {
    mloutput = baseFit(X_mb, y_mb, sc)
    new SVMModel(this, isMultiClass)
  }
  
  def fit(df: ScriptsUtils.SparkDataType): SVMModel = {
    mloutput = baseFit(df, sc)
    new SVMModel(this, isMultiClass)
  }
  
}

object SVMModel {
  final val predictionScriptPathBinary = "scripts" + File.separator + "algorithms" + File.separator + "l2-svm-predict.dml"
  final val predictionScriptPathMulticlass = "scripts" + File.separator + "algorithms" + File.separator + "m-svm-predict.dml"
}

class SVMModel (override val uid: String)(estimator:SVM, val sc: SparkContext, val isMultiClass:Boolean) 
  extends Model[SVMModel] with BaseSystemMLClassifierModel {
  override def copy(extra: ParamMap): SVMModel = {
    val that = new SVMModel(uid)(estimator, sc, isMultiClass)
    copyValues(that, extra)
  }
  
  def this(estimator:SVM, isMultiClass:Boolean) =  {
  	this("model")(estimator, estimator.sc, isMultiClass)
  }
  
  def baseEstimator():BaseSystemMLEstimator = estimator
  def modelVariables():List[String] = List[String]("w")
  
  def getPredictionScript(isSingleNode:Boolean): (Script, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(if(isMultiClass) SVMModel.predictionScriptPathMulticlass else SVMModel.predictionScriptPathBinary))
      .in("$X", " ")
      .in("$model", " ")
      .out("scores")
    
    val w = estimator.mloutput.getMatrix("w")
    val wVar = if(isMultiClass) "W" else "w"
      
    val ret = if(isSingleNode) {
      script.in(wVar, w.toMatrixBlock, w.getMatrixMetadata)
    }
    else {
      script.in(wVar, w)
    }
    (ret, "X")
  }
  
  def transform(X: MatrixBlock): MatrixBlock = baseTransform(X, sc, "scores")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, sc, "scores")
}
