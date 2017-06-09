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

import org.apache.spark.sql.functions.udf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils
import org.apache.sysml.api.mlcontext.MLResults
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext.Script
import org.apache.sysml.api.mlcontext.Matrix

object PredictionUtils {
  
  def getGLMPredictionScript(B_full: Matrix, isSingleNode:Boolean, dfam:java.lang.Integer=1): (Script, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(LogisticRegressionModel.scriptPath))
      .in("$X", " ")
      .in("$B", " ")
      .in("$dfam", dfam)
      .out("means")
    val ret = if(isSingleNode) {
      script.in("B_full", B_full.toMatrixBlock, B_full.getMatrixMetadata)
    }
    else {
      script.in("B_full", B_full)
    }
    (ret, "X")
  }
  
  def joinUsingID(df1:DataFrame, df2:DataFrame):DataFrame = {
    df1.join(df2, RDDConverterUtils.DF_ID_COLUMN)
  }
  
  def computePredictedClassLabelsFromProbability(mlscoreoutput:MLResults, isSingleNode:Boolean, sc:SparkContext, inProbVar:String): MLResults = {
    val ml = new org.apache.sysml.api.mlcontext.MLContext(sc)
    val script = dml(
        """
        Prob = read("temp1");
        Prediction = rowIndexMax(Prob); # assuming one-based label mapping
        write(Prediction, "tempOut", "csv");
        """).out("Prediction")
    val probVar = mlscoreoutput.getMatrix(inProbVar)
    if(isSingleNode) {
      ml.execute(script.in("Prob", probVar.toMatrixBlock, probVar.getMatrixMetadata))
    }
    else {
      ml.execute(script.in("Prob", probVar))
    }
  }
}
