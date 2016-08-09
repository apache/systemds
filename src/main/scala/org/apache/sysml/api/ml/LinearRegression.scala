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

object LinearRegression {
  final val scriptPathCG = "scripts" + File.separator + "algorithms" + File.separator + "LinearRegCG.dml"
  final val scriptPathDS = "scripts" + File.separator + "algorithms" + File.separator + "LinearRegDS.dml"
}

// algorithm = "direct-solve", "conjugate-gradient"
class LinearRegression(override val uid: String, val sc: SparkContext, val solver:String="direct-solve") 
  extends Estimator[LinearRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with BaseSystemMLRegressor {
  
  def setIcpt(value: Int) = set(icpt, value)
  def setMaxIter(value: Int) = set(maxOuterIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)
  
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = {
    val that = new LinearRegression(uid, sc, solver)
    copyValues(that, extra)
  }
  
          
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(
        if(solver.compareTo("direct-solve") == 0) LinearRegression.scriptPathDS 
        else if(solver.compareTo("newton-cg") == 0) LinearRegression.scriptPathCG
        else throw new DMLRuntimeException("The algorithm should be direct-solve or newton-cg")))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$B", " ")
      .in("$Log", " ")
      .in("$fmt", "binary")
      .in("$icpt", toDouble(getIcpt))
      .in("$reg", toDouble(getRegParam))
      .in("$tol", toDouble(getTol))
      .in("$maxi", toDouble(getMaxOuterIte))
      .out("beta_out")
    (script, "X", "y")
  }
  
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): LinearRegressionModel = 
    new LinearRegressionModel("lr")(fit(X_mb, y_mb, sc), sc)
    
  def fit(df: ScriptsUtils.SparkDataType): LinearRegressionModel = 
    new LinearRegressionModel("lr")(fit(df, sc), sc)
  
}

class LinearRegressionModel(override val uid: String)(val mloutput: MLResults, val sc: SparkContext) extends Model[LinearRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter with BaseSystemMLRegressorModel {
  override def copy(extra: ParamMap): LinearRegressionModel = {
    val that = new LinearRegressionModel(uid)(mloutput, sc)
    copyValues(that, extra)
  }
  
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String) =
    PredictionUtils.getGLMPredictionScript(mloutput.getBinaryBlockMatrix("beta_out"), isSingleNode)
  
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = transform(df, mloutput, sc, "means")
  
  def transform(X: MatrixBlock): MatrixBlock =  transform(X, mloutput, sc, "means")
  
}