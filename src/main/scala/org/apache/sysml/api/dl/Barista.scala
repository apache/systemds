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
package org.apache.sysml.api.dl

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.api.ml.ScriptsUtils
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import scala.collection.JavaConversions._
import java.util.ArrayList
import caffe.Caffe.Phase
import caffe.Caffe
import java.util.HashSet
import org.apache.sysml.api.DMLScript
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
import org.apache.sysml.api.ml._
import java.util.Random

object Barista  {
  def main(args:Array[String]):Unit = {
    val b = load(10, null, args(0), args(1), 3, 224, 224)
    System.out.println(b.getTrainingScript(true)._1.getScriptString)
  }
  
  def load(numClasses:Int, sc: SparkContext,  solverFilePath:String, numChannels:Int, inputHeight:Int, inputWidth:Int):Barista = {
    val solver = Utils.readCaffeSolver(solverFilePath)
    new Barista(numClasses, sc, solver, numChannels, inputHeight, inputWidth)
  }
  def load(numClasses:Int, sc: SparkContext, solverFilePath:String, networkPath:String, numChannels:Int, inputHeight:Int, inputWidth:Int):Barista = {
    val solver = Utils.readCaffeSolver(solverFilePath)
    new Barista(numClasses, sc, solver, networkPath, numChannels, inputHeight, inputWidth)
  }
  
  def fileSep():String = { if(File.separator.equals("\\")) "\\\\" else File.separator }
  
  def setNNLibraryPath(path:String):Unit = { prefix = path + fileSep + "nn"}  
  // ------------------------------------------------------------------------
  var numTabs = 0
  
  var prefix = "nn";
	val layerDir = prefix + fileSep + "layers" + fileSep;
	val optimDir = prefix + fileSep + "optim" + fileSep;
  val alreadyImported:HashSet[String] = new HashSet[String]
	def source(dmlScript:StringBuilder, sourceFileName:String, dir:String=layerDir):Unit = {
    if(sourceFileName != null && !alreadyImported.contains(sourceFileName)) {
      alreadyImported.add(sourceFileName)
      dmlScript.append("source(\"" + dir +  sourceFileName + ".dml\") as " + sourceFileName + "\n")
    }
  }
}

class Barista(numClasses:Int, sc: SparkContext, solver:CaffeSolver, net:CaffeNetwork, lrPolicy:LearningRatePolicy) extends Estimator[BaristaModel] 
  with BaseSystemMLClassifier {
  
  def this(numClasses:Int, sc: SparkContext, solver1:Caffe.SolverParameter, networkPath:String, numChannels:Int, inputHeight:Int, inputWidth:Int) {
    this(numClasses, sc, Utils.parseSolver(solver1), 
        new CaffeNetwork(networkPath, caffe.Caffe.Phase.TRAIN, numChannels, inputHeight, inputWidth),
        new LearningRatePolicy(solver1))
  }
  def this(numClasses:Int, sc: SparkContext, solver1:Caffe.SolverParameter, numChannels:Int, inputHeight:Int, inputWidth:Int) {
    this(numClasses, sc, solver1, solver1.getNet, numChannels, inputHeight, inputWidth)
  }
  val rand = new Random
  val uid:String = "caffe_classifier_" + rand.nextLong + "_" + rand.nextLong 
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap): Estimator[BaristaModel] = {
    val that = new Barista(numClasses, sc, solver, net, lrPolicy)
    copyValues(that, extra)
  }
  
  // def transformSchema(schema: StructType): StructType = schema

	// --------------------------------------------------------------
	// Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): BaristaModel = {
    val ret = baseFit(X_mb, y_mb, sc)
    new BaristaModel(ret, numClasses, sc, solver, net, lrPolicy, normalizeInput)
  }
  
  def fit(df: ScriptsUtils.SparkDataType): BaristaModel = {
    val ret = baseFit(df, sc)
    new BaristaModel(ret, numClasses, sc, solver, net, lrPolicy, normalizeInput)
  }
	// --------------------------------------------------------------

	var validationPercentage:Double=0.2
	var display:Int=100
	var normalizeInput:Boolean=true
	var maxIter = 10000
	
	var doVisualize = false
	val visDMLScript: StringBuilder = new StringBuilder 
	def visualizeLoss(): Unit = { 
	   doVisualize = true
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"training_loss\", iter, training_loss);\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"validation_loss\", iter, validation_loss);\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"training_accuracy\", iter, training_accuracy);\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"validation_accuracy\", iter, validation_accuracy);\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	}
	
	
	def visualizeLayer(layerName:String, varType:String, aggFn:String): Unit = {
	  // 'weight', 'bias', 'dweight', 'dbias', 'output' or 'doutput'
	  // 'sum', 'mean', 'var' or 'sd'
	  doVisualize = true
	  if(net.getLayers.filter(_.equals(layerName)).size == 0)
	    throw new DMLRuntimeException("Cannot visualize the layer:" + layerName)
	  val dmlVar = {
	    val l = net.getCaffeLayer(layerName)
	    varType match {
	      case "weight" => l.weight
	      case "bias" => l.bias
	      case "dweight" => l.dW
	      case "dbias" => l.dB
	      case "output" => l.outVar
	      case "doutput" => l.dOut
	      case _ => throw new DMLRuntimeException("Cannot visualize the variable of type:" + varType)
	    }
	   }
	  if(dmlVar == null)
	    throw new DMLRuntimeException("Cannot visualize the variable of type:" + varType)
	  visDMLScript.append("viz_counter1 = visualize(\"" + layerName + "\", \"" + varType + "\", \"" + aggFn + "\", iter, " + aggFn + "(" + dmlVar + "))\n")
	  visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	}
	
	def setValidationPercentage(value:Double) = { validationPercentage = value }
	def setDisplay(value:Int) = { display = value }
	def setNormalizeInput(value:Boolean) = { normalizeInput = value }
	def setMaxIter(value: Int) = { maxIter = value }
	
	// Script generator
	def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
	  if(validationPercentage > 1 || validationPercentage < 0) throw new DMLRuntimeException("Incorrect validation percentage. Should be between (0, 1).")
	  if(display > maxIter || display < 0) throw new DMLRuntimeException("display needs to be between (0, " + maxIter + "). Suggested value: 100.")
	  
	  val dmlScript = new StringBuilder
	  dmlScript.append(Utils.license)
	  // Append source statements for each layer
	  Barista.alreadyImported.clear()
	  net.getLayers.map(layer =>  net.getCaffeLayer(layer).source(dmlScript))
	  Barista.source(dmlScript, "l2_reg")
	  solver.source(dmlScript)
	  if(doVisualize) {
	    dmlScript.append("visualize = externalFunction(String layerName, String varType, String aggFn, Double x, Double y) return (Double B) " +
	        "implemented in (classname=\"org.apache.sysml.udf.lib.BaristaVisualizeWrapper\",exectype=\"mem\"); \n")
	    dmlScript.append("viz_counter = 0\n")
	  }
	        
	  dmlScript.append("X_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("y_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("max_iter = " + maxIter + "\n")
	  dmlScript.append("num_images = nrow(y_full)\n")
	  
	  dmlScript.append("# Convert to one-hot encoding (Assumption: 1-based labels) \n")
	  dmlScript.append("y_full = table(seq(1,num_images,1), y_full, num_images, " + numClasses + ")\n")
	  
	  if(normalizeInput) {
	    // Please donot normalize as well as have scale parameter in the data layer
	    dmlScript.append("# Normalize the inputs\n")
	    dmlScript.append("X_full = (X_full - rowMeans(X_full)) / rowSds(X_full)\n")
	  }
	  
	  // Append init() function for each layer
	  dmlScript.append("# Initialize the layers\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).init(dmlScript))
	  // Split into training and validation set
	  if(validationPercentage > 0) {
	    dmlScript.append("num_validation = ceil(" + validationPercentage + " * num_images)\n")
	    dmlScript.append("X = X_full[(num_validation+1):num_images,]; y = y_full[(num_validation+1):num_images,] \n")
	    dmlScript.append("X_val = X_full[1:num_validation,]; y_val = y_full[1:num_validation,] \n")
	    dmlScript.append("num_images = nrow(y)\n")
	  }
	  else {
	    dmlScript.append("X = X_full; y = y_full;\n")
	  }
	  net.getLayers.map(layer => solver.init(dmlScript, net.getCaffeLayer(layer)))
	  dmlScript.append("iter = 0; beg = 1;\n")
	  dmlScript.append("num_iters_per_epoch = ceil(num_images/BATCH_SIZE);\n")
	  dmlScript.append("epoch = 0;\n")
	  dmlScript.append("max_epoch = ceil(max_iter/num_iters_per_epoch);\n")
	  
	  // ----------------------------
	  dmlScript.append("while(iter < max_iter) {\n")
    // Append forward and backward functions for each layer
    numTabs = 1
    dmlScript.append("\tif(iter %% num_iters_per_epoch == 0) {\n")
    dmlScript.append("\t\t# Learning rate\n")
	  lrPolicy.updateLearningRate(dmlScript.append("\t\t"))
	  dmlScript.append("\t\tepoch = epoch + 1\n")
	  dmlScript.append("\t}\n")
	  
    appendBatch(dmlScript, "\t")
    dmlScript.append("\t").append("# Perform forward pass\n")
    net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "\t", false))
    
    dmlScript.append("\t").append("\n\t# Perform backward pass\n")
    net.getLayers.reverse.map(layer => net.getCaffeLayer(layer).backward(dmlScript.append("\t")))
    net.getLayers.map(layer => solver.update(dmlScript, net.getCaffeLayer(layer)))
    if(display > 0 && validationPercentage > 0) {
      numTabs += 1
      dmlScript.append("\t").append("if(iter %% " + display + " == 0) {\n")
      dmlScript.append("\t\t").append("# Compute training loss & accuracy\n")
      dmlScript.append("\t\t").append("loss = 0\n")
      net.getLayers.map(layer => net.getCaffeLayer(layer).computeLoss(dmlScript, "\t\t"))
      dmlScript.append("\t\t").append("training_loss = loss; loss = 0\n")
      dmlScript.append("\t\t").append("training_accuracy = accuracy\n")
      
      dmlScript.append("\t\t").append("# Compute validation loss & accuracy\n")
      dmlScript.append("\t\t").append("Xb = X_val; yb = y_val;\n")
      net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "\t\t", false))
      net.getLayers.map(layer => net.getCaffeLayer(layer).computeLoss(dmlScript, "\t\t"))
      dmlScript.append("\t\t").append("validation_loss = loss\n")
      dmlScript.append("\t\t").append("validation_accuracy = accuracy\n")
      if(doVisualize)
        dmlScript.append(visDMLScript.toString)
      dmlScript.append("\t\t").append("print(\"Iter: \" + iter + \", training (loss / accuracy): (\" + training_loss + \" / \" + training_accuracy + \"%), validation (loss / accuracy): (\" + validation_loss + \" / \" + validation_accuracy + \"%).\")\n")
      dmlScript.append("\t").append("}\n")
      numTabs -= 1
    }
	  dmlScript.append("\n\t").append("iter = iter + 1\n")
	  dmlScript.append("\n}")
	  if(doVisualize)
	    dmlScript.append("print(viz_counter)")
	  numTabs = 0
	  // ----------------------------
	  
	  
	  val trainingScript = dmlScript.toString()
	  // Uncomment for debugging
    System.out.println(trainingScript)
    val script = dml(trainingScript)
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.out(l.weight))
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.out(l.bias))
	  (script, "X_full", "y_full")
	}
	
	def appendBatch(dmlScript:StringBuilder, prefix:String): Unit = {
	  dmlScript.append(prefix).append("end = beg + BATCH_SIZE - 1\n")
    dmlScript.append(prefix).append("if(end > num_images) end = num_images\n")
    dmlScript.append(prefix).append("Xb = X[beg:end,]; yb = y[beg:end,]\n")
    dmlScript.append(prefix).append("beg = beg + BATCH_SIZE\n")
    dmlScript.append(prefix).append("if(beg > num_images) beg = 1\n")
	}
	
	def getPrevLayerIDForSingleInputLayer(prevLayerIDs:List[Int]): Int = {
	  if(prevLayerIDs.size != 1) {
	    throw new LanguageException("Multiple bottom layers is not handled")
	  }
	  prevLayerIDs.get(0)
	}
	
	def load(outputDir:String, format:String="binary", sep:String="/"):BaristaModel = {
	  new BaristaModel(numClasses, sc, solver, net, lrPolicy, normalizeInput, outputDir, format, sep)
	}
	

	// --------------------------------------------------------------
	// DML generation utility functions
	var prevTab:String = null
	var prevNumTabs = -1
	var numTabs = 2
	def tabs():String = if(numTabs == prevNumTabs) prevTab else { prevTab = ""; for(i <- 0 until numTabs) { prevTab += "\t" } ; prevTab  }
	// --------------------------------------------------------------

}

class BaristaModel(val mloutput: MLResults,  
    val numClasses:Int, val sc: SparkContext, val solver:CaffeSolver, val net:CaffeNetwork, val lrPolicy:LearningRatePolicy,
    val normalizeInput:Boolean) 
  extends Model[BaristaModel] with HasMaxOuterIter with BaseSystemMLClassifierModel {
  
  def this(numClasses:Int, sc: SparkContext, solver:CaffeSolver, net:CaffeNetwork, lrPolicy:LearningRatePolicy,
    normalizeInput:Boolean, outputDir1:String, format1:String, sep1:String) {
    this(null, numClasses, sc, solver, net, lrPolicy, normalizeInput)
    outputDir = outputDir1
    format = format1
    sep = sep1
  }
  
  var outputDir:String = null 
  var format:String = null 
  var sep:String = null
  val rand = new Random
  val uid:String = "caffe_model_" + rand.nextLong + "_" + rand.nextLong 
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap): BaristaModel = {
    val that = new BaristaModel(mloutput, numClasses, sc, solver, net, lrPolicy, normalizeInput)
    copyValues(that, extra)
  }
  
  def write(varName:String, fileName:String, format:String):String = "write(" + varName + ", \"" + fileName + "\", format=\"" + format + "\")\n"
  
  def save(outputDir:String, format:String="binary", sep:String="/"):Unit = {
	  if(mloutput == null) throw new DMLRuntimeException("Cannot save as you need to train the model first using fit")
	  
	  val dmlScript = new StringBuilder
	  dmlScript.append("print(\"Saving the model to " + outputDir + "...\")\n")
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => dmlScript.append(write(l.weight, outputDir + sep + l.param.getName + "_weight.mtx", format)))
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => dmlScript.append(write(l.bias, outputDir + sep + l.param.getName + "_bias.mtx", format)))
	  
	  val script = dml(dmlScript.toString)
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.in(l.weight, mloutput.getBinaryBlockMatrix(l.weight)))
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.in(l.bias, mloutput.getBinaryBlockMatrix(l.bias)))
	  val ml = new MLContext(sc)
	  ml.execute(script)
	  
//	  val rdd = sc.parallelize(labelMapping.map(x => x._1.toString + "," + x._2.toString).toList)
//	  rdd.saveAsTextFile(outputDir + sep + "labelMapping.csv")
	}
  
  def read(varName:String, fileName:String):String =  varName + " = read(\"" + fileName + "\")\n"
  
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String)  = {
	  val dmlScript = new StringBuilder
	  dmlScript.append(Utils.license)
	  if(mloutput == null) {
	    // fit was called
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => dmlScript.append(read(l.weight, outputDir + sep + l.param.getName + "_weight.mtx")))
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => dmlScript.append(read(l.bias, outputDir + sep + l.param.getName + "_bias.mtx")))
	  }
	  
	  // Uncomment for debugging
	  // net.getLayers.map(layer =>  System.out.println(net.getCaffeLayer(layer).param.getName + " " + net.getCaffeLayer(layer).outputShape))
	  
	  // Append source statements for each layer
	  Barista.alreadyImported.clear()
	  net.getLayers.map(layer =>  net.getCaffeLayer(layer).source(dmlScript))
	  dmlScript.append("X_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("num_images = nrow(X_full)\n")
	  
	  if(normalizeInput) {
	    // Please donot normalize as well as have scale parameter in the data layer
	    dmlScript.append("# Normalize the inputs\n")
	    dmlScript.append("X_full = (X_full - rowMeans(X_full)) / rowSds(X_full)\n")
	  }
	  
	  // Append init() function for each layer
	  dmlScript.append("Xb = X_full; \n")
    net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "", true))
    net.getLayers.map(layer => net.getCaffeLayer(layer).predict(dmlScript))
    
    val predictionScript = dmlScript.toString()
    // Uncomment for debugging
    // System.out.println(predictionScript)
	  val script = dml(predictionScript).out("Prob")
	  if(mloutput != null) {
	    // fit was called
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.in(l.weight, mloutput.getBinaryBlockMatrix(l.weight)))
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.in(l.bias, mloutput.getBinaryBlockMatrix(l.bias)))
	  }
	  (script, "X_full")
  }
  
  // Prediction
  def transform(X: MatrixBlock): MatrixBlock = baseTransform(X, mloutput, sc, "Prob")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, mloutput, sc, "Prob")
}