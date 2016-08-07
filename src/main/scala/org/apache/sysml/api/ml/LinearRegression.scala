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

object LinearRegression {
  final val scriptPathCG = "scripts" + File.separator + "algorithms" + File.separator + "LinearRegCG.dml"
  final val scriptPathDS = "scripts" + File.separator + "algorithms" + File.separator + "LinearRegDS.dml"
}

// algorithm = "direct-solve", "conjugate-gradient"
class LinearRegression(override val uid: String, val sc: SparkContext, val solver:String="direct-solve") extends Estimator[LinearRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter {
  
  def setIcpt(value: Int) = set(icpt, value)
  def setMaxIter(value: Int) = set(maxOuterIter, value)
  def setRegParam(value: Double) = set(regParam, value)
  def setTol(value: Double) = set(tol, value)
  
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = {
    val that = new LinearRegression(uid, sc, solver)
    copyValues(that, extra)
  }
  def transformSchema(schema: StructType): StructType = schema
  
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): LinearRegressionModel = {
    val ml = new MLContext(sc)
    if(y_mb.getNumColumns != 1) {
      throw new RuntimeException("Expected a column vector for y")
    }
    val mloutput = {
      ml.registerInput("X", X_mb);
      ml.registerInput("y", y_mb);
      ml.registerOutput("beta_out");
      if(solver.compareTo("direct-solve") == 0)
        ml.executeScript(ScriptsUtils.getDMLScript(LinearRegression.scriptPathDS), getParamMap())
      else if(solver.compareTo("newton-cg") == 0) {
        ml.executeScript(ScriptsUtils.getDMLScript(LinearRegression.scriptPathCG), getParamMap())
      }
      else {
        throw new DMLRuntimeException("The algorithm should be direct-solve or conjugate-gradient")
      }
    }
    new LinearRegressionModel("linearRegression")(mloutput, sc)
  }
  
  def getParamMap(): Map[String, String] = {
    Map(  "icpt" -> this.getIcpt.toString(),
          "reg" -> this.getRegParam.toString(),
          "tol" -> this.getTol.toString,
          "maxi" -> this.getMaxOuterIte.toString,
  
          "X" -> " ",
          "Y" -> " ",
          "B" -> " ", 
          "O" -> " ", 
          "Log" -> " ",
          "fmt" -> "binary")
  }
  
  def fit(df: DataFrame): LinearRegressionModel = {
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, df, mcXin, false, "features")
    val yin = df.select("label")
    val mloutput = {
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("y", yin);
      ml.registerOutput("beta_out");
      if(solver.compareTo("direct-solve") == 0)
        ml.executeScript(ScriptsUtils.getDMLScript(LinearRegression.scriptPathDS), getParamMap())
      else if(solver.compareTo("newton-cg") == 0) {
        ml.executeScript(ScriptsUtils.getDMLScript(LinearRegression.scriptPathCG), getParamMap())
      }
      else {
        throw new DMLRuntimeException("The algorithm should be direct-solve or conjugate-gradient")
      }
    }
    new LinearRegressionModel("linearRegression")(mloutput, sc)
  }
}

class LinearRegressionModel(override val uid: String)(val mloutput: MLOutput, val sc: SparkContext) extends Model[LinearRegressionModel] with HasIcpt
    with HasRegParam with HasTol with HasMaxOuterIter {
  override def copy(extra: ParamMap): LinearRegressionModel = {
    val that = new LinearRegressionModel(uid)(mloutput, sc)
    copyValues(that, extra)
  }
  
  override def transformSchema(schema: StructType): StructType = schema
  
  def transform(df: DataFrame): DataFrame = {
    val isSingleNode = false
    val glmPredOut = PredictionUtils.doGLMPredict(isSingleNode, df, null, sc, mloutput, "beta_out", getPredictParams())
    val predictedDF = glmPredOut.getDF(df.sqlContext, "means").withColumnRenamed("C1", "prediction")
    val dataset = RDDConverterUtils.addIDToDataFrame(df, df.sqlContext, "ID")
    return PredictionUtils.joinUsingID(dataset, predictedDF)
  }
  
  def transform(X: MatrixBlock): MatrixBlock =  {
    val isSingleNode = true
    return PredictionUtils.doGLMPredict(isSingleNode, null, X, sc, mloutput, "beta_out", getPredictParams()).getMatrixBlock("means")
  }
  
  
  def getPredictParams(): Map[String, String] = {
    Map("X" -> " ",
        "B" -> " ",
        // Gaussian distribution
        "dfam" -> "1", "vpow" -> "0.0",
        // identity link function
        "link" -> "1", "lpow" -> "1.0"
//        // Dispersion value: TODO
//        ,"disp" -> "5.0"
        )
  }
}