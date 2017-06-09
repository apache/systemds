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

object NaiveBayes {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes.dml"
}

class NaiveBayes(override val uid: String, val sc: SparkContext) extends Estimator[NaiveBayesModel] with HasLaplace with BaseSystemMLClassifier {
  override def copy(extra: ParamMap): Estimator[NaiveBayesModel] = {
    val that = new NaiveBayes(uid, sc)
    copyValues(that, extra)
  }
  def setLaplace(value: Double) = set(laplace, value)
  
  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): NaiveBayesModel = {
    mloutput = baseFit(X_mb, y_mb, sc)
    new NaiveBayesModel(this)
  }
  
  def fit(df: ScriptsUtils.SparkDataType): NaiveBayesModel = {
    mloutput = baseFit(df, sc)
    new NaiveBayesModel(this)
  }
  
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(NaiveBayes.scriptPath))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$prior", " ")
      .in("$conditionals", " ")
      .in("$accuracy", " ")
      .in("$laplace", toDouble(getLaplace))
      .out("classPrior", "classConditionals")
    (script, "D", "C")
  }
}


object NaiveBayesModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "naive-bayes-predict.dml"
}

class NaiveBayesModel(override val uid: String)
  (estimator:NaiveBayes, val sc: SparkContext) 
  extends Model[NaiveBayesModel] with HasLaplace with BaseSystemMLClassifierModel {
  
  def this(estimator:NaiveBayes) =  {
    this("model")(estimator, estimator.sc)
  }
  
  override def copy(extra: ParamMap): NaiveBayesModel = {
    val that = new NaiveBayesModel(uid)(estimator, sc)
    copyValues(that, extra)
  }
  
  def modelVariables():List[String] = List[String]("classPrior", "classConditionals")
  def getPredictionScript(isSingleNode:Boolean): (Script, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(NaiveBayesModel.scriptPath))
      .in("$X", " ")
      .in("$prior", " ")
      .in("$conditionals", " ")
      .in("$probabilities", " ")
      .out("probs")
    
    val classPrior = estimator.mloutput.getMatrix("classPrior")
    val classConditionals = estimator.mloutput.getMatrix("classConditionals")
    val ret = if(isSingleNode) {
      script.in("prior", classPrior.toMatrixBlock, classPrior.getMatrixMetadata)
            .in("conditionals", classConditionals.toMatrixBlock, classConditionals.getMatrixMetadata)
    }
    else {
      script.in("prior", classPrior.toBinaryBlocks, classPrior.getMatrixMetadata)
            .in("conditionals", classConditionals.toBinaryBlocks, classConditionals.getMatrixMetadata)
    }
    (ret, "D")
  }
  
  def baseEstimator():BaseSystemMLEstimator = estimator
  def transform(X: MatrixBlock): MatrixBlock = baseTransform(X, sc, "probs")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, sc, "probs")
  
}
