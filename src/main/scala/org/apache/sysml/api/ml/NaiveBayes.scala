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
import org.apache.spark.ml.param.{ Params, Param, ParamMap, DoubleParam }
import org.apache.sysml.api.{ MLContext, MLOutput }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }

trait HasLaplace extends Params {
  final val laplace: Param[Double] = new Param[Double](this, "laplace", "Laplace smoothing specified by the user to avoid creation of 0 probabilities.")
  setDefault(laplace, 1.0)
  final def getLaplace: Double = $(laplace)
}

object NaiveBayes {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes.dml"
}

class NaiveBayes(override val uid: String, val sc: SparkContext) extends Estimator[NaiveBayesModel] with HasLaplace {
  override def copy(extra: ParamMap): Estimator[NaiveBayesModel] = {
    val that = new NaiveBayes(uid, sc)
    copyValues(that, extra)
  }
  def setLaplace(value: Double) = set(laplace, value)
  override def transformSchema(schema: StructType): StructType = schema
  
  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): NaiveBayesModel = {
    val ml = new MLContext(sc)
    val revLabelMapping = new java.util.HashMap[Int, String]
    PredictionUtils.fillLabelMapping(y_mb, revLabelMapping)
    
    val mloutput = {
      ml.registerInput("D", X_mb);
      ml.registerInput("C", y_mb);
      ml.registerOutput("classPrior");
      ml.registerOutput("classConditionals");
      ml.executeScript(ScriptsUtils.getDMLScript(NaiveBayes.scriptPath), getParamMap())
    }
    new NaiveBayesModel("naivebayes")(mloutput, revLabelMapping, sc)
  }
  
  def fit(df: DataFrame): NaiveBayesModel = {
    val ml = new MLContext(df.rdd.sparkContext)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(sc, df, mcXin, false, "features")
    val revLabelMapping = new java.util.HashMap[Int, String]
    val yin = PredictionUtils.fillLabelMapping(df, revLabelMapping)
    val mloutput = {
      ml.registerInput("D", Xin, mcXin);
      ml.registerInput("C", yin, "csv");
      ml.registerOutput("classPrior");
      ml.registerOutput("classConditionals");
      ml.executeScript(ScriptsUtils.getDMLScript(NaiveBayes.scriptPath), getParamMap())
    }
    new NaiveBayesModel("naive")(mloutput, revLabelMapping, sc)
  }
  
  def getParamMap(): Map[String, String] = {
    Map("X" -> " ",
        "Y" -> " ",
        "prior" -> " ",
        "conditionals" -> " ",
        "accuracy" -> " ",
        "laplace" -> getLaplace.toString())
  }
}


object NaiveBayesModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes-predict.dml"
}

class NaiveBayesModel(
  override val uid: String)(
    val mloutput: MLOutput, val labelMapping: java.util.HashMap[Int, String], val sc: SparkContext) extends Model[NaiveBayesModel] with HasLaplace {
  override def copy(extra: ParamMap): NaiveBayesModel = {
    val that = new NaiveBayesModel(uid)(mloutput, labelMapping, sc)
    copyValues(that, extra)
  }
  
  def transformSchema(schema: StructType): StructType = schema
  
  var priorMB: MatrixBlock = null
  var conditionalMB: MatrixBlock = null
  def setPriorAndConditional(prior:MatrixBlock, conditional:MatrixBlock) {
    priorMB = prior
    conditionalMB = conditional
  }
  
  def transform(X: MatrixBlock): MatrixBlock = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    ml.registerInput("D", X)
    ml.registerInput("prior", mloutput.getMatrixBlock("classPrior"), mloutput.getMatrixCharacteristics("classPrior"))
    ml.registerInput("conditionals", mloutput.getMatrixBlock("classConditionals"), mloutput.getMatrixCharacteristics("classConditionals"))
    ml.registerOutput("probs")
    val nbPredict = ml.executeScript(ScriptsUtils.getDMLScript(NaiveBayesModel.scriptPath), getPredictParams())
    val ret = PredictionUtils.computePredictedClassLabelsFromProbability(nbPredict, isSingleNode, sc, "probs").getMatrixBlock("Prediction");
    if(ret.getNumColumns != 1) {
      throw new RuntimeException("Expected predicted label to be a column vector")
    }
    PredictionUtils.updateLabels(isSingleNode, null, ret, null, labelMapping)
    return ret
  }
  
  def transform(df: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame = {
    val isSingleNode = false
    val ml = new MLContext(sc)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.vectorDataFrameToBinaryBlock(df.rdd.sparkContext, df, mcXin, false, "features")
    ml.registerInput("D", Xin, mcXin);
    ml.registerInput("prior", mloutput.getMatrixBlock("classPrior"), mloutput.getMatrixCharacteristics("classPrior"))
    ml.registerInput("conditionals", mloutput.getMatrixBlock("classConditionals"), mloutput.getMatrixCharacteristics("classConditionals"))
    ml.registerOutput("probs")
    val nbPredict = ml.executeScript(ScriptsUtils.getDMLScript(NaiveBayesModel.scriptPath), getPredictParams())
    val predLabelOut = PredictionUtils.computePredictedClassLabelsFromProbability(nbPredict, isSingleNode, sc, "probs")
    val predictedDF = PredictionUtils.updateLabels(isSingleNode, predLabelOut.getDF(df.sqlContext, "Prediction"), null, "C1", labelMapping).select("ID", "prediction")
    val prob = nbPredict.getDF(df.sqlContext, "probs", true).withColumnRenamed("C1", "probability").select("ID", "probability")
    val dataset = RDDConverterUtils.addIDToDataFrame(df, df.sqlContext, "ID")
    return PredictionUtils.joinUsingID(dataset, PredictionUtils.joinUsingID(prob, predictedDF))
  }
  
  def getPredictParams(): Map[String, String] = {
    Map("X" -> " ",
        "prior" -> " ",
        "conditionals" -> " ",
        "probabilities" -> " ")
  }

}