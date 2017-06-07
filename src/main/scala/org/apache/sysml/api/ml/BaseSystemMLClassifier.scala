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

import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.rdd.RDD
import java.io.File
import org.apache.spark.SparkContext
import org.apache.spark.ml.{ Model, Estimator }
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.{ Params, Param, ParamMap, DoubleParam }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt, RDDConverterUtils }
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.spark.sql._
import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel
import java.util.HashMap
import scala.collection.JavaConversions._


/****************************************************
DESIGN DOCUMENT for MLLEARN API:
The mllearn API supports LogisticRegression, LinearRegression, SVM, NaiveBayes 
and Caffe2DML. Every algorithm in this API has a python wrapper (implemented in the mllearn python package)
and a Scala class where the actual logic is implementation. 
Both wrapper and scala class follow the below hierarchy to reuse code and simplify the implementation.


                  BaseSystemMLEstimator
                          |
      --------------------------------------------
      |                                          |
BaseSystemMLClassifier                  BaseSystemMLRegressor
      ^                                          ^
      |                                          |
SVM, Caffe2DML, ...                          LinearRegression


To conform with MLLib API, for every algorithm, we support two classes for every algorithm:
1. Estimator for training: For example: SVM extends Estimator[SVMModel].
2. Model for prediction: For example: SVMModel extends Model[SVMModel]

Both BaseSystemMLRegressor and BaseSystemMLClassifier implements following methods for training:
1. For compatibility with scikit-learn: baseFit(X_mb: MatrixBlock, y_mb: MatrixBlock, sc: SparkContext): MLResults
2. For compatibility with MLLib: baseFit(df: ScriptsUtils.SparkDataType, sc: SparkContext): MLResults

In the above methods, we execute the DML script for the given algorithm using MLContext.
The missing piece of the puzzle is how does BaseSystemMLRegressor and BaseSystemMLClassifier interfaces
get the DML script. To enable this, each wrapper class has to implement following methods:
1. getTrainingScript(isSingleNode:Boolean):(Script object of mlcontext, variable name of X in the script:String, variable name of y in the script:String)
2. getPredictionScript(isSingleNode:Boolean): (Script object of mlcontext, variable name of X in the script:String)

****************************************************/

trait HasLaplace extends Params {
  final val laplace: Param[Double] = new Param[Double](this, "laplace", "Laplace smoothing specified by the user to avoid creation of 0 probabilities.")
  setDefault(laplace, 1.0)
  final def getLaplace: Double = $(laplace)
}
trait HasIcpt extends Params {
  final val icpt: Param[Int] = new Param[Int](this, "icpt", "Intercept presence, shifting and rescaling X columns")
  setDefault(icpt, 0)
  final def getIcpt: Int = $(icpt)
}
trait HasMaxOuterIter extends Params {
  final val maxOuterIter: Param[Int] = new Param[Int](this, "maxOuterIter", "max. number of outer (Newton) iterations")
  setDefault(maxOuterIter, 100)
  final def getMaxOuterIte: Int = $(maxOuterIter)
}
trait HasMaxInnerIter extends Params {
  final val maxInnerIter: Param[Int] = new Param[Int](this, "maxInnerIter", "max. number of inner (conjugate gradient) iterations, 0 = no max")
  setDefault(maxInnerIter, 0)
  final def getMaxInnerIter: Int = $(maxInnerIter)
}
trait HasTol extends Params {
  final val tol: DoubleParam = new DoubleParam(this, "tol", "the convergence tolerance for iterative algorithms")
  setDefault(tol, 0.000001)
  final def getTol: Double = $(tol)
}
trait HasRegParam extends Params {
  final val regParam: DoubleParam = new DoubleParam(this, "regParam", "regularization parameter")
  setDefault(regParam, 0.000001)
  final def getRegParam: Double = $(regParam)
}

trait BaseSystemMLEstimatorOrModel {
  var enableGPU:Boolean = false
  var forceGPU:Boolean = false
  var explain:Boolean = false
  var statistics:Boolean = false
  var statisticsMaxHeavyHitters:Int = 10
  val config:HashMap[String, String] = new HashMap[String, String]()
  def setGPU(enableGPU1:Boolean):BaseSystemMLEstimatorOrModel = { enableGPU = enableGPU1; this}
  def setForceGPU(enableGPU1:Boolean):BaseSystemMLEstimatorOrModel = { forceGPU = enableGPU1; this}
  def setExplain(explain1:Boolean):BaseSystemMLEstimatorOrModel = { explain = explain1; this}
  def setStatistics(statistics1:Boolean):BaseSystemMLEstimatorOrModel = { statistics = statistics1; this}
  def setStatisticsMaxHeavyHitters(statisticsMaxHeavyHitters1:Int):BaseSystemMLEstimatorOrModel = { statisticsMaxHeavyHitters = statisticsMaxHeavyHitters1; this}
  def setConfigProperty(key:String, value:String):BaseSystemMLEstimatorOrModel = { config.put(key, value); this}
  def updateML(ml:MLContext):Unit = {
    ml.setGPU(enableGPU); ml.setForceGPU(forceGPU);
    ml.setExplain(explain); ml.setStatistics(statistics); ml.setStatisticsMaxHeavyHitters(statisticsMaxHeavyHitters); 
    config.map(x => ml.setConfigProperty(x._1, x._2))
  }
  def copyProperties(other:BaseSystemMLEstimatorOrModel):BaseSystemMLEstimatorOrModel = {
    other.setGPU(enableGPU); other.setForceGPU(forceGPU);
    other.setExplain(explain); other.setStatistics(statistics); other.setStatisticsMaxHeavyHitters(statisticsMaxHeavyHitters);
    config.map(x => other.setConfigProperty(x._1, x._2))
    return other
  }
}

trait BaseSystemMLEstimator extends BaseSystemMLEstimatorOrModel {
  def transformSchema(schema: StructType): StructType = schema
  var mloutput:MLResults = null
  // Returns the script and variables for X and y
  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)
  
  def toDouble(i:Int): java.lang.Double = {
    double2Double(i.toDouble)
  }
  
  def toDouble(d:Double): java.lang.Double = {
    double2Double(d)
  }
  
}

trait BaseSystemMLEstimatorModel extends BaseSystemMLEstimatorOrModel {
  def toDouble(i:Int): java.lang.Double = {
    double2Double(i.toDouble)
  }
  def toDouble(d:Double): java.lang.Double = {
    double2Double(d)
  }
  
  def transformSchema(schema: StructType): StructType = schema
  
  // Returns the script and variable for X
  def getPredictionScript(isSingleNode:Boolean): (Script, String)
  def baseEstimator():BaseSystemMLEstimator
  def modelVariables():List[String]
  // self.model.load(self.sc._jsc, weights, format, sep)
  def load(sc:JavaSparkContext, outputDir:String, sep:String):Unit = {
  	val dmlScript = new StringBuilder
  	dmlScript.append("print(\"Loading the model from " + outputDir + "...\")\n")
		for(varName <- modelVariables) {
			dmlScript.append(varName + " = read(\"" + outputDir + sep + varName + ".mtx\")\n")
		}
  	val script = dml(dmlScript.toString)
		for(varName <- modelVariables) {
			script.out(varName)
		}
	  val ml = new MLContext(sc)
	  baseEstimator.mloutput = ml.execute(script)
  }
  def save(sc:JavaSparkContext, outputDir:String, format:String="binary", sep:String="/"):Unit = {
	  if(baseEstimator.mloutput == null) throw new DMLRuntimeException("Cannot save as you need to train the model first using fit")
	  val dmlScript = new StringBuilder
	  dmlScript.append("print(\"Saving the model to " + outputDir + "...\")\n")
	  for(varName <- modelVariables) {
	  	dmlScript.append("write(" + varName + ", \"" + outputDir + sep + varName + ".mtx\", format=\"" + format + "\")\n")
	  }
	  val script = dml(dmlScript.toString)
		for(varName <- modelVariables) {
			script.in(varName, baseEstimator.mloutput.getMatrix(varName))
		}
	  val ml = new MLContext(sc)
	  ml.execute(script)
	}
}

trait BaseSystemMLClassifier extends BaseSystemMLEstimator {
  def baseFit(X_mb: MatrixBlock, y_mb: MatrixBlock, sc: SparkContext): MLResults = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    updateML(ml)
    y_mb.recomputeNonZeros();
    val ret = getTrainingScript(isSingleNode)
    val script = ret._1.in(ret._2, X_mb).in(ret._3, y_mb)
    ml.execute(script)
  }
  def baseFit(df: ScriptsUtils.SparkDataType, sc: SparkContext): MLResults = {
    val isSingleNode = false
    val ml = new MLContext(df.rdd.sparkContext)
    updateML(ml)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(sc, df.asInstanceOf[DataFrame].select("features"), mcXin, false, true)
    val revLabelMapping = new java.util.HashMap[Int, String]
    val yin = df.select("label")
    val ret = getTrainingScript(isSingleNode)
    val mmXin = new MatrixMetadata(mcXin)
    val Xbin = new Matrix(Xin, mmXin)
    val script = ret._1.in(ret._2, Xbin).in(ret._3, yin)
    ml.execute(script)
  }
}

trait BaseSystemMLClassifierModel extends BaseSystemMLEstimatorModel {

  def baseTransform(X: MatrixBlock, sc: SparkContext, probVar:String): MatrixBlock = {
    val isSingleNode = true
    val ml = new MLContext(sc)
    updateML(ml)
    val script = getPredictionScript(isSingleNode)
    // Uncomment for debugging
    // ml.setExplainLevel(ExplainLevel.RECOMPILE_RUNTIME)
    val modelPredict = ml.execute(script._1.in(script._2, X, new MatrixMetadata(X.getNumRows, X.getNumColumns, X.getNonZeros)))
    val ret = PredictionUtils.computePredictedClassLabelsFromProbability(modelPredict, isSingleNode, sc, probVar)
              .getMatrix("Prediction").toMatrixBlock
              
    if(ret.getNumColumns != 1) {
      throw new RuntimeException("Expected predicted label to be a column vector")
    }
    return ret
  }

  def baseTransform(df: ScriptsUtils.SparkDataType, sc: SparkContext, 
      probVar:String, outputProb:Boolean=true): DataFrame = {
    val isSingleNode = false
    val ml = new MLContext(sc)
    updateML(ml)
    val mcXin = new MatrixCharacteristics()
    val Xin = RDDConverterUtils.dataFrameToBinaryBlock(df.rdd.sparkContext, df.asInstanceOf[DataFrame].select("features"), mcXin, false, true)
    val script = getPredictionScript(isSingleNode)
    val mmXin = new MatrixMetadata(mcXin)
    val Xin_bin = new Matrix(Xin, mmXin)
    val modelPredict = ml.execute(script._1.in(script._2, Xin_bin))
    val predLabelOut = PredictionUtils.computePredictedClassLabelsFromProbability(modelPredict, isSingleNode, sc, probVar)
    val predictedDF = predLabelOut.getDataFrame("Prediction").select(RDDConverterUtils.DF_ID_COLUMN, "C1").withColumnRenamed("C1", "prediction")
      
    if(outputProb) {
      val prob = modelPredict.getDataFrame(probVar, true).withColumnRenamed("C1", "probability").select(RDDConverterUtils.DF_ID_COLUMN, "probability")
      val dataset = RDDConverterUtilsExt.addIDToDataFrame(df.asInstanceOf[DataFrame], df.sparkSession, RDDConverterUtils.DF_ID_COLUMN)
      return PredictionUtils.joinUsingID(dataset, PredictionUtils.joinUsingID(prob, predictedDF))
    }
    else {
      val dataset = RDDConverterUtilsExt.addIDToDataFrame(df.asInstanceOf[DataFrame], df.sparkSession, RDDConverterUtils.DF_ID_COLUMN)
      return PredictionUtils.joinUsingID(dataset, predictedDF)
    }
    
  }
}
