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
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt, RDDConverterUtils }
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._

trait BaseSystemMLRegressor extends BaseSystemMLEstimator {
  
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock, sc: SparkContext): MLResults = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    val ret = getTrainingScript(isSingleNode)
    val script = ret._1.in(ret._2, X_mb).in(ret._3, y_mb)
    ml.execute(script)
  }
  
  def fit(df: ScriptsUtils.SparkDataType, sc: SparkContext): MLResults = {
    val isSingleNode = false
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(sc, df.asInstanceOf[DataFrame], mcXin, false, true)
    val yin = df.select("label")
    val ret = getTrainingScript(isSingleNode)
    val Xbin = new BinaryBlockMatrix(Xin, mcXin)
    val script = ret._1.in(ret._2, Xbin).in(ret._3, yin)
    ml.execute(script)
  }
}

trait BaseSystemMLRegressorModel extends BaseSystemMLEstimatorModel {
  
  def transform(X: MatrixBlock, mloutput: MLResults, sc: SparkContext, predictionVar:String): MatrixBlock = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    val script = getPredictionScript(mloutput, isSingleNode)
    val modelPredict = ml.execute(script._1.in(script._2, X))
    val ret = modelPredict.getBinaryBlockMatrix(predictionVar).getMatrixBlock
              
    if(ret.getNumColumns != 1) {
      throw new RuntimeException("Expected prediction to be a column vector")
    }
    return ret
  }
  
  def transform(df: ScriptsUtils.SparkDataType, mloutput: MLResults, sc: SparkContext, predictionVar:String): DataFrame = {
    val isSingleNode = false
    val ml = new MLContext(sc)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(df.rdd.sparkContext, df.asInstanceOf[DataFrame], mcXin, false, true)
    val script = getPredictionScript(mloutput, isSingleNode)
    val Xin_bin = new BinaryBlockMatrix(Xin, mcXin)
    val modelPredict = ml.execute(script._1.in(script._2, Xin_bin))
    val predictedDF = modelPredict.getDataFrame(predictionVar).select("__INDEX", "C1").withColumnRenamed("C1", "prediction")
    val dataset = RDDConverterUtilsExt.addIDToDataFrame(df.asInstanceOf[DataFrame], df.sqlContext, "__INDEX")
    return PredictionUtils.joinUsingID(dataset, predictedDF)
  }
}