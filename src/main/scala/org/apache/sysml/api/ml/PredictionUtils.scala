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
import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }

object PredictionUtils {
  
  def doGLMPredict(isSingleNode:Boolean, df:DataFrame, X: MatrixBlock, sc:SparkContext, mloutput:MLOutput, B:String, paramsMap: Map[String, String]): MLOutput = {
    val ml = new MLContext(sc)
    if(isSingleNode) {
      ml.registerInput("X", X);
      ml.registerInput("B_full", mloutput.getMatrixBlock(B), mloutput.getMatrixCharacteristics(B));
    }
    else {
      val mcXin = new MatrixCharacteristics()
      val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(df.rdd.sparkContext, df, mcXin, false, "features")
      ml.registerInput("X", Xin, mcXin);
      ml.registerInput("B_full", mloutput.getBinaryBlockedRDD(B), mloutput.getMatrixCharacteristics(B));  
    }
    ml.registerOutput("means");
    ml.executeScript(ScriptsUtils.getDMLScript(LogisticRegressionModel.scriptPath), paramsMap)
  }
  
  def joinUsingID(df1:DataFrame, df2:DataFrame):DataFrame = {
    val tempDF1 = df1.withColumnRenamed("ID", "ID1")
    tempDF1.join(df2, tempDF1.col("ID1").equalTo(df2.col("ID"))).drop("ID1")
  }
  
  def computePredictedClassLabelsFromProbability(mlscoreoutput:MLOutput, isSingleNode:Boolean, sc:SparkContext): MLOutput = {
    val mlNew = new MLContext(sc)
    if(isSingleNode) {
      mlNew.registerInput("Prob", mlscoreoutput.getMatrixBlock("means"), mlscoreoutput.getMatrixCharacteristics("means"));
    }
    else {
      mlNew.registerInput("Prob", mlscoreoutput.getBinaryBlockedRDD("means"), mlscoreoutput.getMatrixCharacteristics("means"));
    }
    mlNew.registerOutput("Prediction")
    mlNew.executeScript(
      """
        Prob = read("temp1");
        Prediction = rowIndexMax(Prob); # assuming one-based label mapping
        write(Prediction, "tempOut", "csv");
        """)
  }
}