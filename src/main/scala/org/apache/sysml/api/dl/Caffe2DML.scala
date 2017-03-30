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
import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory

object Caffe2DML  {
  val LOG = LogFactory.getLog(classOf[Caffe2DML].getName())
  def fileSep():String = { if(File.separator.equals("\\")) "\\\\" else File.separator }
  def setNNLibraryPath(path:String):Unit = { prefix = path + fileSep + "nn"}  
  // ------------------------------------------------------------------------
  var numTabs = 0
  var prefix = Utils.getPrefix()
  def layerDir = prefix + fileSep + "layers" + fileSep
  def optimDir = prefix + fileSep + "optim" + fileSep
  val alreadyImported:HashSet[String] = new HashSet[String]
  def source(dmlScript:StringBuilder, sourceFileName:String, dir:String=layerDir):Unit = {
    if(sourceFileName != null && !alreadyImported.contains(sourceFileName)) {
      alreadyImported.add(sourceFileName)
      dmlScript.append("source(\"" + dir +  sourceFileName + ".dml\") as " + sourceFileName + "\n")
    }
  }
}

class Caffe2DML(val sc: SparkContext, val solverParam:Caffe.SolverParameter, val solver:CaffeSolver, val net:CaffeNetwork, val lrPolicy:LearningRatePolicy) extends Estimator[Caffe2DMLModel] 
  with BaseSystemMLClassifier {
  // Invoked by Python, MLPipeline
  def this(sc: SparkContext, solver1:Caffe.SolverParameter, networkPath:String) {
    this(sc, solver1, Utils.parseSolver(solver1), 
        new CaffeNetwork(networkPath, caffe.Caffe.Phase.TRAIN),
        new LearningRatePolicy(solver1))
  }
  def this(sc: SparkContext, solver1:Caffe.SolverParameter) {
    this(sc, solver1, solver1.getNet)
  } 
  val rand = new Random
  val uid:String = "caffe_classifier_" + rand.nextLong + "_" + rand.nextLong 
  val layersToIgnore:HashSet[String] = new HashSet[String]() 
  def setWeightsToIgnore(layerName:String):Unit = layersToIgnore.add(layerName)
  def setWeightsToIgnore(layerNames:ArrayList[String]):Unit = layersToIgnore.addAll(layerNames)
  
  // -----------------------------------------------------------------------
  // Logic to determine the input shape (i.e. channel, height, width) of the image as we get flattened NumPy array or Vector
  // If the image is square gray-scale (for example: MNIST), the user need not provide the input shape.
  // However, for any other situation (i.e. RGB image or non-square gray-scale image), the user must provide the input shape.
  var input_shape_known = false
  def setInputShape(numChannels:String, height:String, width:String):Unit = {
	input_shape_known = true
	net.dataLayers.map(l => { l.dataOutputShape = (numChannels, height, width) })
  }
  def setInputShape(numChannels:Int, height:Int, width:Int):Unit = setInputShape(numChannels.toString, height.toString, width.toString)
  def setInputShape(X_mb: MatrixBlock):Unit = {
	if(!input_shape_known) {
		val height = Math.sqrt(X_mb.getNumColumns).toInt
		if(Math.sqrt(X_mb.getNumColumns) == height)
			setInputShape(1, height, height)
	    else
	    	throw new DMLRuntimeException("Cannot infer input_shape. Please set(input_shape=[number_channels, image_height, image_width])")
	}
  }
  def setInputShape(df: ScriptsUtils.SparkDataType):Unit = {
	 if(!input_shape_known) {
	  	val numColumns = df.select("features").first().get(0).asInstanceOf[org.apache.spark.ml.linalg.Vector].size
		val height = Math.sqrt(numColumns).toInt
		if(Math.sqrt(numColumns) == height)
			setInputShape(1, height, height)
	    else
	    	throw new DMLRuntimeException("Cannot infer input_shape. Please set(input_shape=[number_channels, image_height, image_width])")
	} 
  }
  // -----------------------------------------------------------------------
		  
  val inputs:java.util.HashMap[String, String] = new java.util.HashMap[String, String]() 
  def setInput(key: String, value:String):Unit = inputs.put(key, value)
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap): Estimator[Caffe2DMLModel] = {
    val that = new Caffe2DML(sc, solverParam, solver, net, lrPolicy)
    copyValues(that, extra)
  }
  
  // def transformSchema(schema: StructType): StructType = schema
  
  

	// --------------------------------------------------------------
	// Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): Caffe2DMLModel = {
	setInputShape(X_mb)
    val ret = baseFit(X_mb, y_mb, sc)
    new Caffe2DMLModel(ret, Utils.numClasses(net), sc, solver, net, lrPolicy, this)
  }
  
  def fit(df: ScriptsUtils.SparkDataType): Caffe2DMLModel = {
	setInputShape(df) 
    val ret = baseFit(df, sc)
    new Caffe2DMLModel(ret, Utils.numClasses(net), sc, solver, net, lrPolicy, this)
  }
	// --------------------------------------------------------------
  var _tensorboardLogDir:String = null
  def setTensorBoardLogDir(log:String): Unit = { _tensorboardLogDir = log }
  def tensorboardLogDir:String = {
    if(_tensorboardLogDir == null) {
      _tensorboardLogDir = java.io.File.createTempFile("temp", System.nanoTime().toString()).getAbsolutePath
    }
    _tensorboardLogDir
  }
  
	var doVisualize = false
	val visDMLScript: StringBuilder = new StringBuilder 
	private def checkTensorBoardDependency():Unit = {
	  try {
	    if(!doVisualize)
	      Class.forName( "com.google.protobuf.GeneratedMessageV3")
	  } catch {
	    case _:ClassNotFoundException => throw new DMLRuntimeException("To use visualize() feature, you will have to include protobuf-java-3.2.0.jar in your classpath. Hint: you can download the jar from http://central.maven.org/maven2/com/google/protobuf/protobuf-java/3.2.0/protobuf-java-3.2.0.jar")   
	  }
	}
	def visualizeLoss(): Unit = {
	   checkTensorBoardDependency()
	   doVisualize = true
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"training_loss\", iter, training_loss, \"" + tensorboardLogDir + "\");\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"validation_loss\", iter, validation_loss, \"" + tensorboardLogDir + "\");\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"training_accuracy\", iter, training_accuracy, \"" + tensorboardLogDir + "\");\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	   visDMLScript.append("viz_counter1 = visualize(\" \", \" \", \"validation_accuracy\", iter, validation_accuracy, \"" + tensorboardLogDir + "\");\n")
	   visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	}
	
	
	def visualizeLayer(layerName:String, varType:String, aggFn:String): Unit = {
	  // 'weight', 'bias', 'dweight', 'dbias', 'output' or 'doutput'
	  // 'sum', 'mean', 'var' or 'sd'
	  checkTensorBoardDependency()
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
	  visDMLScript.append("viz_counter1 = visualize(\"" + layerName + "\", \"" + varType + "\", \"" + aggFn + "\", iter, " + aggFn + "(" + dmlVar + "), \"" + tensorboardLogDir + "\")\n")
	  visDMLScript.append("viz_counter = viz_counter + viz_counter1\n")
	}
	
	def write(varName:String, quotedFileName:String, format:String):String = "write(" + varName + ", " + quotedFileName + ", format=\"" + format + "\")\n"
	
	// Script generator
	def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
	  val startTrainingTime = System.nanoTime()
	  val display = if(inputs.containsKey("$display")) inputs.get("$display").toInt else solverParam.getDisplay
	  val validationPercentage = if(inputs.containsKey("$validation_split")) inputs.get("$validation_split").toDouble else 0.1
	  val DEBUG_TRAINING = if(inputs.containsKey("$debug")) inputs.get("$debug").toLowerCase.toBoolean else false
	  if(validationPercentage > 1 || validationPercentage < 0) throw new DMLRuntimeException("Incorrect validation percentage. Should be between (0, 1).")
	  
	  val snapshot = if(inputs.containsKey("$snapshot")) inputs.get("$snapshot").toInt else solverParam.getSnapshot
	  val snapshotPrefix = if(inputs.containsKey("$snapshot_prefix")) inputs.get("$snapshot_prefix") else solverParam.getSnapshotPrefix
	  
	  val dmlScript = new StringBuilder
	  // Append source statements for each layer
	  Caffe2DML.alreadyImported.clear()
	  net.getLayers.map(layer =>  net.getCaffeLayer(layer).source(dmlScript))
	  Caffe2DML.source(dmlScript, "l2_reg")
	  solver.source(dmlScript)
	  if(doVisualize) {
	    dmlScript.append("visualize = externalFunction(String layerName, String varType, String aggFn, Double x, Double y, String logDir) return (Double B) " +
	        "implemented in (classname=\"org.apache.sysml.udf.lib.Caffe2DMLVisualizeWrapper\",exectype=\"mem\"); \n")
	    dmlScript.append("viz_counter = 0\n")
	    System.out.println("Please use the following command for visualizing: tensorboard --logdir=" + tensorboardLogDir)
	  }
	        
	  dmlScript.append("X_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("y_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("weights = ifdef($weights, \" \")\n")
	  dmlScript.append("snapshot_prefix = ifdef($snapshot_prefix, \" \")\n")
	  dmlScript.append("normalize_input = ifdef($normalize_input, FALSE)\n")
	  dmlScript.append("max_iter = ifdef($max_iter, " + solverParam.getMaxIter + ")\n")
	  dmlScript.append("num_images = nrow(y_full)\n")
	  dmlScript.append("debug = ifdef($debug, FALSE)\n")
	  
	  dmlScript.append("# Convert to one-hot encoding (Assumption: 1-based labels) \n")
	  dmlScript.append("y_full = table(seq(1,num_images,1), y_full, num_images, " + Utils.numClasses(net) + ")\n")
	  
	  dmlScript.append("# Normalize the inputs\n")
	  dmlScript.append("if(normalize_input) { \n")
	  dmlScript.append("\tX_full = (X_full - rowMeans(X_full)) / rowSds(X_full)\n")
	  dmlScript.append("}\n")
	  
	  dmlScript.append("# Initialize the layers\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).init(dmlScript))
	  if(inputs.containsKey("$weights")) {
		  // Loading existing weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias
		  dmlScript.append("# Load the weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias\n")
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => dmlScript.append(read(l.weight, l.param.getName + "_weight.mtx")))
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => dmlScript.append(read(l.bias, l.param.getName + "_bias.mtx")))
	  }
	  
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
      if(DEBUG_TRAINING) {
    	  dmlScript.append("\t\t").append("print(\"Classification report (on current batch):\")\n")
      }
      net.getLayers.map(layer => net.getCaffeLayer(layer).computeLoss(dmlScript, "\t\t"))
      dmlScript.append("\t\t").append("training_loss = loss; loss = 0\n")
      dmlScript.append("\t\t").append("training_accuracy = accuracy\n")
      
      // Debugging prints to figure out learning rate:
      // if(DEBUG_TRAINING) {
	  //    net.getLayers.map(layer => if(net.getCaffeLayer(layer).shouldUpdateWeight) dmlScript.append("\t\t").append("print(\"mean(gradient of " + layer + "'s weight) =\" + mean(" + net.getCaffeLayer(layer).dW + "))\n"))
      // }
      
      dmlScript.append("\t\t").append("# Compute validation loss & accuracy\n")
      dmlScript.append("\t\t").append("Xb = X_val; yb = y_val;\n")
      net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "\t\t", false))
      if(DEBUG_TRAINING) {
    	  dmlScript.append("\t\t").append("print(\"Classification report (on validation set):\")\n")
      }
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
	  if(snapshot > 0) {
		  dmlScript.append("\n\t").append("if(snapshot_prefix != \" \" & iter %% snapshot == 0){\n")
		  dmlScript.append("\n\t").append("\tsnapshot_dir= snapshot_prefix + \"/iter_\" + iter + \"/\"\n")
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => dmlScript.append(write(l.weight, "snapshot_dir + \"" + l.param.getName + "_weight.mtx\"", "binary")))
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => dmlScript.append(write(l.bias, "snapshot_dir + \"" + l.param.getName + "_bias.mtx\"", "binary")))
		  dmlScript.append("\n\t").append("}\n")
	  }
	  dmlScript.append("\n}")
	  if(doVisualize)
	    dmlScript.append("print(viz_counter)")
	  numTabs = 0
	  // ----------------------------
	  
	  
	  val trainingScript = dmlScript.toString()
	  if(DEBUG_TRAINING) {
    	Utils.prettyPrintDMLScript(trainingScript)
	  }
	  val script = dml(trainingScript).in(inputs)
    
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.out(l.weight))
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.out(l.bias))
	  
	  if(DEBUG_TRAINING) {
	    System.out.println("Time taken to generate training script from Caffe proto:" + ((System.nanoTime() - startTrainingTime)*1e-9) + "secs." )
	  }
	  
	  (script, "X_full", "y_full")
	}
	
	def read(varName:String, fileName:String, sep:String="/"):String =  varName + " = read(weights + \"" + sep + fileName + "\")\n"
			
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

	// --------------------------------------------------------------
	// DML generation utility functions
	var prevTab:String = null
	var prevNumTabs = -1
	var numTabs = 2
	def tabs():String = if(numTabs == prevNumTabs) prevTab else { prevTab = ""; for(i <- 0 until numTabs) { prevTab += "\t" } ; prevTab  }
	// --------------------------------------------------------------

}

class Caffe2DMLModel(val mloutput: MLResults,  
    val numClasses:String, val sc: SparkContext, val solver:CaffeSolver, val net:CaffeNetwork, val lrPolicy:LearningRatePolicy,
    val estimator:Caffe2DML) 
  extends Model[Caffe2DMLModel] with HasMaxOuterIter with BaseSystemMLClassifierModel {
  val rand = new Random
  val uid:String = "caffe_model_" + rand.nextLong + "_" + rand.nextLong 
  
  def this(estimator:Caffe2DML) = this(null, Utils.numClasses(estimator.net), estimator.sc, estimator.solver, estimator.net, estimator.lrPolicy, estimator)
  
  override def copy(extra: org.apache.spark.ml.param.ParamMap): Caffe2DMLModel = {
    val that = new Caffe2DMLModel(mloutput, numClasses, sc, solver, net, lrPolicy, estimator)
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
  
  def read(varName:String, fileName:String, sep:String="/"):String =  varName + " = read(weights + \"" + sep + fileName + "\")\n"
  
  def updateMeanAndVariance(enable: Boolean): Unit  = net.getLayers.map(layer =>  {
	    val l = net.getCaffeLayer(layer)
	    if(l.isInstanceOf[BatchNorm])
	      l.asInstanceOf[BatchNorm].update_mean_var = enable
	  })
  
	def updateRemoteParForPrediction(enable: Boolean): Unit  = net.getLayers.map(layer =>  {
	    val l = net.getCaffeLayer(layer)
	    if(l.isInstanceOf[SoftmaxWithLoss])
	      l.asInstanceOf[SoftmaxWithLoss].remote_parfor_prediction = enable
	  })
	  
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String)  = {
    val startPredictionTime = System.nanoTime()
	  val dmlScript = new StringBuilder
	  val DEBUG_PREDICTION = if(estimator.inputs.containsKey("$debug")) estimator.inputs.get("$debug").toLowerCase.toBoolean else false
	  dmlScript.append("weights = ifdef($weights, \" \")\n")
	  if(mloutput == null && estimator.inputs.containsKey("$weights")) {
		  // fit was not called
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => dmlScript.append(read(l.weight, l.param.getName + "_weight.mtx")))
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => dmlScript.append(read(l.bias, l.param.getName + "_bias.mtx")))
	  }
	  else if(mloutput == null) {
		  throw new DMLRuntimeException("Cannot call predict/score without calling either fit or by providing weights")
	  }
	  
	  if(estimator.inputs.containsKey("$debug") && estimator.inputs.get("$debug").equals("TRUE")) {
		  System.out.println("The output shape of layers:")
		  net.getLayers.map(layer =>  System.out.println(net.getCaffeLayer(layer).param.getName + " " + net.getCaffeLayer(layer).outputShape))
	  }
	  
	  // Donot update mean and variance in batchnorm
	  updateMeanAndVariance(false)
	  
	  // Append source statements for each layer
	  Caffe2DML.alreadyImported.clear()
	  net.getLayers.map(layer =>  net.getCaffeLayer(layer).source(dmlScript))
	  dmlScript.append("X_full = read(\" \", format=\"csv\")\n")
	  dmlScript.append("num_images = nrow(X_full)\n")
	  
	  // If there is a convolution or maxpooling layer (eg: CNN), use parfor prediction as this is best we can get.
	  // Else use full-test training to exploit our distributed operators (eg: auto-encoder)
	  val shouldUseParforPrediction = net.getLayers.foldLeft(false)((prev, layer) => prev || net.getCaffeLayer(layer).isInstanceOf[Convolution] || net.getCaffeLayer(layer).isInstanceOf[MaxPooling])
	  if(shouldUseParforPrediction) {
	    updateRemoteParForPrediction(true)
	    val lastLayerShape = net.getCaffeLayer(net.getLayers.last).outputShape
	    dmlScript.append("Prob = matrix(0, rows=num_images, cols=" + lastLayerShape._1 + ")\n")
	    dmlScript.append("parfor(i in 1:num_images) {\n")
	    dmlScript.append("Xb = X_full[i,]; \n")
	    if(estimator.inputs.containsKey("$normalize_input") && estimator.inputs.get("$normalize_input").equalsIgnoreCase("TRUE")) {
	      dmlScript.append("\tXb = (Xb - mean(Xb)) / sd(Xb)\n")
	    }
  		net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "", true))
  		net.getLayers.map(layer => net.getCaffeLayer(layer).predict(dmlScript))
  		dmlScript.append("}\n")
	    updateRemoteParForPrediction(false)
	  }
	  else {
	    if(estimator.inputs.containsKey("$normalize_input") && estimator.inputs.get("$normalize_input").equalsIgnoreCase("TRUE")) {
	      dmlScript.append("\tX_full = (X_full - rowMeans(X_full)) / rowSds(X_full)\n")
	    }
  	  dmlScript.append("Xb = X_full; \n")
  		net.getLayers.map(layer => net.getCaffeLayer(layer).forward(dmlScript, "", true))
  		net.getLayers.map(layer => net.getCaffeLayer(layer).predict(dmlScript))
	  }
		
		val predictionScript = dmlScript.toString()
		if(DEBUG_PREDICTION) {
			Utils.prettyPrintDMLScript(predictionScript)
		}
	  val script = dml(predictionScript).out("Prob").in(estimator.inputs)
	  if(mloutput != null) {
	    // fit was called
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.in(l.weight, mloutput.getBinaryBlockMatrix(l.weight)))
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.in(l.bias, mloutput.getBinaryBlockMatrix(l.bias)))
	  }
	  
	  updateMeanAndVariance(true)
	  
	  if(DEBUG_PREDICTION) {
	    System.out.println("Time taken to generate prediction script from Caffe proto:" + ((System.nanoTime() - startPredictionTime)*1e-9) + "secs." )
	  }
	  
	  (script, "X_full")
  }
  
  // Prediction
  def transform(X: MatrixBlock): MatrixBlock = {
	  estimator.setInputShape(X)
	  baseTransform(X, mloutput, sc, "Prob")
  }
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = {
	  estimator.setInputShape(df)
	  baseTransform(df, mloutput, sc, "Prob")
  }
}