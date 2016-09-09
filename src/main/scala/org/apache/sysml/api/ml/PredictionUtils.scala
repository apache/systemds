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
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }
import org.apache.sysml.api.mlcontext.MLResults
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext.Script
import org.apache.sysml.api.mlcontext.BinaryBlockMatrix

object PredictionUtils {
  
  def getGLMPredictionScript(B_full: BinaryBlockMatrix, isSingleNode:Boolean, dfam:java.lang.Integer=1): (Script, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(LogisticRegressionModel.scriptPath))
      .in("$X", " ")
      .in("$B", " ")
      .in("$dfam", dfam)
      .out("means")
    val ret = if(isSingleNode) {
      script.in("B_full", B_full.getMatrixBlock, B_full.getMatrixMetadata)
    }
    else {
      script.in("B_full", B_full)
    }
    (ret, "X")
  }
  
  def fillLabelMapping(df: ScriptsUtils.SparkDataType, revLabelMapping: java.util.HashMap[Int, String]): RDD[String]  = {
    val temp = df.select("label").distinct.rdd.map(_.apply(0).toString).collect()
    val labelMapping = new java.util.HashMap[String, Int]
    for(i <- 0 until temp.length) {
      labelMapping.put(temp(i), i+1)
      revLabelMapping.put(i+1, temp(i))
    }
    df.select("label").rdd.map( x => labelMapping.get(x.apply(0).toString).toString )
  }
  
  def fillLabelMapping(y_mb: MatrixBlock, revLabelMapping: java.util.HashMap[Int, String]): Unit = {
    val labelMapping = new java.util.HashMap[String, Int]
    if(y_mb.getNumColumns != 1) {
      throw new RuntimeException("Expected a column vector for y")
    }
    if(y_mb.isInSparseFormat()) {
      throw new DMLRuntimeException("Sparse block is not implemented for fit")
    }
    else {
      val denseBlock = y_mb.getDenseBlock()
      var id:Int = 1
      for(i <- 0 until denseBlock.length) {
        val v = denseBlock(i).toString()
        if(!labelMapping.containsKey(v)) {
          labelMapping.put(v, id)
          revLabelMapping.put(id, v)
          id += 1
        }
        denseBlock.update(i, labelMapping.get(v))
      }  
    }
  }
  
  class LabelMappingData(val labelMapping: java.util.HashMap[Int, String]) extends Serializable {
   def mapLabelStr(x:Double):String = {
     if(labelMapping.containsKey(x.toInt))
       labelMapping.get(x.toInt)
     else
       throw new RuntimeException("Incorrect label mapping")
   }
   def mapLabelDouble(x:Double):Double = {
     if(labelMapping.containsKey(x.toInt))
       labelMapping.get(x.toInt).toDouble
     else
       throw new RuntimeException("Incorrect label mapping")
   }
   val mapLabel_udf =  {
        try {
          val it = labelMapping.values().iterator()
          while(it.hasNext()) {
            it.next().toDouble
          }
          udf(mapLabelDouble _)
        } catch {
          case e: Exception => udf(mapLabelStr _)
        }
      }
  }  
  def updateLabels(isSingleNode:Boolean, df:DataFrame, X: MatrixBlock, labelColName:String, labelMapping: java.util.HashMap[Int, String]): DataFrame = {
    if(isSingleNode) {
      if(X.isInSparseFormat()) {
        throw new RuntimeException("Since predicted label is a column vector, expected it to be in dense format")
      }
      for(i <- 0 until X.getNumRows) {
        val v:Int = X.getValue(i, 0).toInt
        if(labelMapping.containsKey(v)) {
          X.setValue(i, 0, labelMapping.get(v).toDouble)
        }
        else {
          throw new RuntimeException("No mapping found for " + v + " in " + labelMapping.toString())
        }
      }
      return null
    }
    else {
      val serObj = new LabelMappingData(labelMapping)
      return df.withColumn(labelColName, serObj.mapLabel_udf(df(labelColName)))
               .withColumnRenamed(labelColName, "prediction")
    }
  }
  
  def joinUsingID(df1:DataFrame, df2:DataFrame):DataFrame = {
    val tempDF1 = df1.withColumnRenamed("__INDEX", "ID1")
    tempDF1.join(df2, tempDF1.col("ID1").equalTo(df2.col("__INDEX"))).drop("ID1")
  }
  
  def computePredictedClassLabelsFromProbability(mlscoreoutput:MLResults, isSingleNode:Boolean, sc:SparkContext, inProbVar:String): MLResults = {
    val ml = new org.apache.sysml.api.mlcontext.MLContext(sc)
    val script = dml(
        """
        Prob = read("temp1");
        Prediction = rowIndexMax(Prob); # assuming one-based label mapping
        write(Prediction, "tempOut", "csv");
        """).out("Prediction")
    val probVar = mlscoreoutput.getBinaryBlockMatrix(inProbVar)
    if(isSingleNode) {
      ml.execute(script.in("Prob", probVar.getMatrixBlock, probVar.getMatrixMetadata))
    }
    else {
      ml.execute(script.in("Prob", probVar))
    }
  }
}