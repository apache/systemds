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

import java.util.HashSet
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;
import org.apache.sysml.runtime.DMLRuntimeException;
import scala.collection.JavaConversions._
import caffe.Caffe

trait BaseDMLGenerator {
  def commaSep(arr:List[String]):String = {
	  if(arr.length == 1) arr(0) else {
	    var ret = arr(0)
	    for(i <- 1 until arr.length) {
	      ret = ret + "," + arr(i)
	    }
	    ret
	  }
	}
  def commaSep(arr:String*):String = {
	  if(arr.length == 1) arr(0) else {
	    var ret = arr(0)
	    for(i <- 1 until arr.length) {
	      ret = ret + "," + arr(i)
	    }
	    ret
	  }
	}
  def int_add(v1:String, v2:String):String = {
    try { (v1.toDouble + v2.toDouble).toInt.toString } catch { case _:Throwable => "("+v1+"+"+v2+")"}
  }
  def int_mult(v1:String, v2:String, v3:String):String = {
    try { (v1.toDouble * v2.toDouble * v3.toDouble).toInt.toString } catch { case _:Throwable => "("+v1+"*"+v2+"*"+v3+")"}
  }
  def isNumber(x: String):Boolean = x forall Character.isDigit
  def transpose(x:String):String = "t(" + x + ")"
  def write(varName:String, fileName:String, format:String):String = "write(" + varName + ", \"" + fileName + "\", format=\"" + format + "\")\n"
  def readWeight(varName:String, fileName:String, sep:String="/"):String =  varName + " = read(weights + \"" + sep + fileName + "\")\n"
  def asDMLString(str:String):String = "\"" + str + "\""
  def assign(dmlScript:StringBuilder, lhsVar:String, rhsVar:String):Unit = {
    dmlScript.append(lhsVar).append(" = ").append(rhsVar).append("\n")
  }
  def sum(dmlScript:StringBuilder, variables:List[String]):StringBuilder = {
    if(variables.length > 1) dmlScript.append("(")
    dmlScript.append(variables(0))
    for(i <- 1 until variables.length) {
      dmlScript.append(" + ").append(variables(i))
    }
    if(variables.length > 1) dmlScript.append(")")
    return dmlScript
  }
  def addAndAssign(dmlScript:StringBuilder, lhsVar:String, rhsVars:List[String]):Unit = {
    dmlScript.append(lhsVar).append(" = ")
    sum(dmlScript, rhsVars)
    dmlScript.append("\n")
  }
  def invoke(dmlScript:StringBuilder, namespace1:String, returnVariables:List[String], functionName:String, arguments:List[String]):Unit = {
    invoke(dmlScript, namespace1, returnVariables, functionName, arguments, true)
  }
  def invoke(dmlScript:StringBuilder, namespace1:String, returnVariables:List[String], functionName:String, arguments:List[String], appendNewLine:Boolean):Unit = {
    if(returnVariables.length == 0) throw new DMLRuntimeException("User-defined functions should have atleast one return value")
    if(returnVariables.length > 1) dmlScript.append("[")
    dmlScript.append(returnVariables(0))
    if(returnVariables.length > 1) {
      for(i <- 1 until returnVariables.length) {
	      dmlScript.append(",").append(returnVariables(i))
	    }
      dmlScript.append("]")
    }
    dmlScript.append(" = ")
    dmlScript.append(namespace1)
    dmlScript.append(functionName)
    dmlScript.append("(")
    if(arguments != null) {
      if(arguments.length != 0) 
        dmlScript.append(arguments(0))
      if(arguments.length > 1) {
        for(i <- 1 until arguments.length) {
  	      dmlScript.append(",").append(arguments(i))
  	    }
      }
    }
    dmlScript.append(")")
    if(appendNewLine) 
      dmlScript.append("\n")
  }
  def invoke(dmlScript:StringBuilder, namespace1:String, returnVariables:List[String], functionName:String, appendNewLine:Boolean, arguments:String*):Unit = {
    invoke(dmlScript, namespace1, returnVariables, functionName, arguments.toList, appendNewLine)
  }
  def invoke(dmlScript:StringBuilder, namespace1:String, returnVariables:List[String], functionName:String, arguments:String*):Unit = {
    invoke(dmlScript, namespace1, returnVariables, functionName, arguments.toList, true)
  }
  def rightIndexing(dmlScript:StringBuilder, varName:String, rl:String, ru:String, cl:String, cu:String):StringBuilder = {
    dmlScript.append(varName).append("[")
    if(rl != null && ru != null) dmlScript.append(rl).append(":").append(ru)
    dmlScript.append(",")
    if(cl != null && cu != null) dmlScript.append(cl).append(":").append(cu)
    dmlScript.append("]")
  }
  // Performs assignVar = ceil(lhsVar/rhsVar)
  def ceilDivide(dmlScript:StringBuilder, assignVar:String, lhsVar:String, rhsVar:String):Unit = 
    dmlScript.append(assignVar).append(" = ").append("ceil(").append(lhsVar).append(" / ").append(rhsVar).append(")\n")
  def print(arg:String):String = "print(" + arg + ")\n"
  def dmlConcat(arg:String*):String = {
    val ret = new StringBuilder
    ret.append(arg(0))
    for(i <- 1 until arg.length) {
      ret.append(" + ").append(arg(i))
    }
    ret.toString
  }
  def matrix(init:String, rows:String, cols:String):String = "matrix(" + init + ", rows=" + rows + ", cols=" + cols + ")" 
  def nrow(m:String):String = "nrow(" + m + ")"
  def ncol(m:String):String = "ncol(" + m + ")"
  def customAssert(cond:Boolean, msg:String) = if(!cond) throw new DMLRuntimeException(msg)
  def multiply(v1:String, v2:String):String = v1 + "*" + v2
  def colSums(m:String):String = "colSums(" + m + ")"
  def ifdef(cmdLineVar:String, defaultVal:String):String = "ifdef(" + cmdLineVar + ", " + defaultVal + ")"
  def ifdef(cmdLineVar:String):String = ifdef(cmdLineVar, "\" \"")
  def read(filePathVar:String, format:String):String = "read(" + filePathVar + ", format=\""+ format + "\")"
}

trait TabbedDMLGenerator extends BaseDMLGenerator {
  def tabDMLScript(dmlScript:StringBuilder, numTabs:Int):StringBuilder =  tabDMLScript(dmlScript, numTabs, false)
  def tabDMLScript(dmlScript:StringBuilder, numTabs:Int, prependNewLine:Boolean):StringBuilder =  {
    if(prependNewLine) dmlScript.append("\n")
	  for(i <- 0 until numTabs) dmlScript.append("\t")
	  dmlScript
  }
}

trait SourceDMLGenerator extends TabbedDMLGenerator {
  val alreadyImported:HashSet[String] = new HashSet[String]
  def source(dmlScript:StringBuilder, numTabs:Int, sourceFileName:String, dir:String):Unit = {
	  if(sourceFileName != null && !alreadyImported.contains(sourceFileName)) {
      tabDMLScript(dmlScript, numTabs).append("source(\"" + dir +  sourceFileName + ".dml\") as " + sourceFileName + "\n")
      alreadyImported.add(sourceFileName)
	  }
  }
  def source(dmlScript:StringBuilder, numTabs:Int, net:CaffeNetwork, solver:CaffeSolver, otherFiles:Array[String]):Unit = {
	  // Add layers with multiple source files
	  if(net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[SoftmaxWithLoss]).length > 0) {
	    source(dmlScript, numTabs, "softmax", Caffe2DML.layerDir)
	    source(dmlScript, numTabs, "cross_entropy_loss", Caffe2DML.layerDir)
	  }
	  net.getLayers.map(layer =>  source(dmlScript, numTabs, net.getCaffeLayer(layer).sourceFileName, Caffe2DML.layerDir))
	  if(solver != null)
	    source(dmlScript, numTabs, solver.sourceFileName, Caffe2DML.optimDir)
	  if(otherFiles != null)
	    otherFiles.map(sourceFileName => source(dmlScript, numTabs, sourceFileName, Caffe2DML.layerDir))
	}
}

trait NextBatchGenerator extends TabbedDMLGenerator {
	def min(lhs:String, rhs:String): String = "min(" + lhs + ", " + rhs + ")"
	
	def assignBatch(dmlScript:StringBuilder, Xb:String, X:String, yb:String, y:String, indexPrefix:String, N:String, i:String):StringBuilder = {
	  dmlScript.append(indexPrefix).append("beg = ((" + i + "-1) * " + Caffe2DML.batchSize + ") %% " + N + " + 1; ")
	  dmlScript.append(indexPrefix).append("end = min(beg + " +  Caffe2DML.batchSize + " - 1, " + N + "); ")
	  dmlScript.append(Xb).append(" = ").append(X).append("[").append(indexPrefix).append("beg:").append(indexPrefix).append("end,]; ")
	  if(yb != null && y != null)
	    dmlScript.append(yb).append(" = ").append(y).append("[").append(indexPrefix).append("beg:").append(indexPrefix).append("end,]; ")
	  dmlScript.append("\n")
	}
	def getTestBatch(tabDMLScript:StringBuilder):Unit = {
    assignBatch(tabDMLScript, "Xb", Caffe2DML.X, null, null, "", Caffe2DML.numImages, "i")
  } 
  def getTrainingBatch(tabDMLScript:StringBuilder):Unit = {
    assignBatch(tabDMLScript, "Xb", Caffe2DML.X, "yb", Caffe2DML.y, "", Caffe2DML.numImages, "i")
  }
	def getTrainingBatch(tabDMLScript:StringBuilder, X:String, y:String, numImages:String):Unit = {
	  assignBatch(tabDMLScript, "Xb", X, "yb", y, "", numImages, "i")
  }
  def getTrainingMaxiBatch(tabDMLScript:StringBuilder):Unit = {
    assignBatch(tabDMLScript, "X_group_batch", Caffe2DML.X, "y_group_batch", Caffe2DML.y, "group_", Caffe2DML.numImages, "g")
  }
  def getValidationBatch(tabDMLScript:StringBuilder):Unit = {
    assignBatch(tabDMLScript, "Xb", Caffe2DML.XVal, "yb", Caffe2DML.yVal, "", Caffe2DML.numValidationImages, "iVal")
  }
}

trait VisualizeDMLGenerator extends TabbedDMLGenerator {
  var doVisualize = false
  var _tensorboardLogDir:String = null
  def setTensorBoardLogDir(log:String): Unit = { _tensorboardLogDir = log }
  def tensorboardLogDir:String = {
    if(_tensorboardLogDir == null) {
      _tensorboardLogDir = java.io.File.createTempFile("temp", System.nanoTime().toString()).getAbsolutePath
    }
    _tensorboardLogDir
  }
  def visualizeLoss(): Unit = {
	   checkTensorBoardDependency()
	   doVisualize = true
	   // Visualize for both training and validation
	   visualize(" ", " ", "training_loss", "iter", "training_loss", true)
	   visualize(" ", " ", "training_accuracy", "iter", "training_accuracy", true)
	   visualize(" ", " ", "validation_loss", "iter", "validation_loss", false)
	   visualize(" ", " ", "validation_accuracy", "iter", "validation_accuracy", false)
	}
  val visTrainingDMLScript: StringBuilder = new StringBuilder 
  val visValidationDMLScript: StringBuilder = new StringBuilder
	def checkTensorBoardDependency():Unit = {
	  try {
	    if(!doVisualize)
	      Class.forName( "com.google.protobuf.GeneratedMessageV3")
	  } catch {
	    case _:ClassNotFoundException => throw new DMLRuntimeException("To use visualize() feature, you will have to include protobuf-java-3.2.0.jar in your classpath. Hint: you can download the jar from http://central.maven.org/maven2/com/google/protobuf/protobuf-java/3.2.0/protobuf-java-3.2.0.jar")   
	  }
	}
  private def visualize(layerName:String, varType:String, aggFn:String, x:String, y:String,  isTraining:Boolean) = {
    val dmlScript = if(isTraining) visTrainingDMLScript else visValidationDMLScript
    dmlScript.append("viz_counter1 = visualize(" + 
        commaSep(asDMLString(layerName), asDMLString(varType), asDMLString(aggFn), x, y, asDMLString(tensorboardLogDir))
        + ");\n")
    dmlScript.append("viz_counter = viz_counter + viz_counter1\n")
  }
  def visualizeLayer(net:CaffeNetwork, layerName:String, varType:String, aggFn:String): Unit = {
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
	      case "dweight" => l.dWeight
	      case "dbias" => l.dBias
	      case "output" => l.out
	      // case "doutput" => l.dX
	      case _ => throw new DMLRuntimeException("Cannot visualize the variable of type:" + varType)
	    }
	   }
	  if(dmlVar == null)
	    throw new DMLRuntimeException("Cannot visualize the variable of type:" + varType)
	  // Visualize for both training and validation
	  visualize(layerName, varType, aggFn, "iter", aggFn + "(" + dmlVar + ")", true)
	  visualize(layerName, varType, aggFn, "iter", aggFn + "(" + dmlVar + ")", false)
	}
  
  def appendTrainingVisualizationBody(dmlScript:StringBuilder, numTabs:Int): Unit = {
    if(doVisualize)
        tabDMLScript(dmlScript, numTabs).append(visTrainingDMLScript.toString)
  }
  def appendValidationVisualizationBody(dmlScript:StringBuilder, numTabs:Int): Unit = {
    if(doVisualize)
        tabDMLScript(dmlScript, numTabs).append(visValidationDMLScript.toString)
  }
}

trait DMLGenerator extends SourceDMLGenerator with NextBatchGenerator with VisualizeDMLGenerator {
  // Also makes "code reading" possible for Caffe2DML :)
	var dmlScript = new StringBuilder
	var numTabs = 0
	def reset():Unit = {
	  dmlScript.clear()
	  alreadyImported.clear()
	  numTabs = 0
	  visTrainingDMLScript.clear()
	  visValidationDMLScript.clear()
	  doVisualize = false
	}
	// -------------------------------------------------------------------------------------------------
	// Helper functions that calls super class methods and simplifies the code of this trait
	def tabDMLScript():StringBuilder = tabDMLScript(dmlScript, numTabs, false)
	def tabDMLScript(prependNewLine:Boolean):StringBuilder = tabDMLScript(dmlScript, numTabs, prependNewLine)
	def source(net:CaffeNetwork, solver:CaffeSolver, otherFiles:Array[String]):Unit = {
	  source(dmlScript, numTabs, net, solver, otherFiles)
	}
	// -------------------------------------------------------------------------------------------------
	
	def ifBlock(cond:String)(op: => Unit) {
	  tabDMLScript.append("if(" + cond + ") {\n")
	  numTabs += 1
	  op
	  numTabs -= 1
	  tabDMLScript.append("}\n")
	}
	def forBlock(iterVarName:String, startVal:String, endVal:String)(op: => Unit) {
	  tabDMLScript.append("for(" + iterVarName + " in " + startVal + ":" + endVal + ") {\n")
	  numTabs += 1
	  op
	  numTabs -= 1
	  tabDMLScript.append("}\n")
	}
	def parForBlock(iterVarName:String, startVal:String, endVal:String)(op: => Unit) {
	  tabDMLScript.append("parfor(" + iterVarName + " in " + startVal + ":" + endVal + ") {\n")
	  numTabs += 1
	  op
	  numTabs -= 1
	  tabDMLScript.append("}\n")
	}
	
	def printClassificationReport():Unit = {
    ifBlock("debug"){
      assign(tabDMLScript, "num_rows_error_measures", min("10", ncol("yb")))
      assign(tabDMLScript, "error_measures", matrix("0", "num_rows_error_measures", "5"))
      forBlock("class_i", "1", "num_rows_error_measures") {
        assign(tabDMLScript, "tp", "sum( (true_yb == predicted_yb) * (true_yb == class_i) )")
        assign(tabDMLScript, "tp_plus_fp", "sum( (predicted_yb == class_i) )")
        assign(tabDMLScript, "tp_plus_fn", "sum( (true_yb == class_i) )")
        assign(tabDMLScript, "precision", "tp / tp_plus_fp")
        assign(tabDMLScript, "recall", "tp / tp_plus_fn")
        assign(tabDMLScript, "f1Score", "2*precision*recall / (precision+recall)")
        assign(tabDMLScript, "error_measures[class_i,1]", "class_i")
        assign(tabDMLScript, "error_measures[class_i,2]", "precision")
        assign(tabDMLScript, "error_measures[class_i,3]", "recall")
        assign(tabDMLScript, "error_measures[class_i,4]", "f1Score")
        assign(tabDMLScript, "error_measures[class_i,5]", "tp_plus_fn")
      }
      val dmlTab = "\\t"
      val header = "class    " + dmlTab + "precision" + dmlTab + "recall  " + dmlTab + "f1-score" + dmlTab + "num_true_labels\\n"
      val errorMeasures = "toString(error_measures, decimal=7, sep=" + asDMLString(dmlTab) + ")"
      tabDMLScript.append(print(dmlConcat(asDMLString(header), errorMeasures)))
    }
  }
	
	// Appends DML corresponding to source and externalFunction statements. 
  def appendHeaders(net:CaffeNetwork, solver:CaffeSolver, isTraining:Boolean):Unit = {
    // Append source statements for layers as well as solver
	  source(net, solver, if(isTraining) Array[String]("l2_reg") else null)
	  
	  if(isTraining) {
  	  // Append external built-in function headers:
  	  // 1. visualize external built-in function header
      if(doVisualize) {
  	    tabDMLScript.append("visualize = externalFunction(String layerName, String varType, String aggFn, Double x, Double y, String logDir) return (Double B) " +
  	        "implemented in (classname=\"org.apache.sysml.udf.lib.Caffe2DMLVisualizeWrapper\",exectype=\"mem\"); \n")
  	    tabDMLScript.append("viz_counter = 0\n")
  	    System.out.println("Please use the following command for visualizing: tensorboard --logdir=" + tensorboardLogDir)
  	  }
  	  // 2. update_nesterov external built-in function header
  	  if(Caffe2DML.USE_NESTEROV_UDF) {
  	    tabDMLScript.append("update_nesterov = externalFunction(matrix[double] X, matrix[double] dX, double lr, double mu, matrix[double] v, double lambda) return (matrix[double] X, matrix[double] v) implemented in (classname=\"org.apache.sysml.udf.lib.SGDNesterovUpdate\",exectype=\"mem\");  \n")
  	  }
	  }
  }
  
  def readMatrix(varName:String, cmdLineVar:String):Unit = {
    val pathVar = varName + "_path"
    assign(tabDMLScript, pathVar, ifdef(cmdLineVar))
    // Uncomment the following lines if we want to the user to pass the format
    // val formatVar = varName + "_fmt"
    // assign(tabDMLScript, formatVar, ifdef(cmdLineVar + "_fmt", "\"csv\""))
    // assign(tabDMLScript, varName, "read(" + pathVar + ", format=" + formatVar + ")")
    assign(tabDMLScript, varName, "read(" + pathVar + ")")
  }
  
  def readInputData(net:CaffeNetwork, isTraining:Boolean):Unit = {
    // Read and convert to one-hot encoding
    readMatrix("X_full", "$X")
	  if(isTraining) {
	    readMatrix("y_full", "$y")
  	  tabDMLScript.append(Caffe2DML.numImages).append(" = nrow(y_full)\n")
  	  tabDMLScript.append("# Convert to one-hot encoding (Assumption: 1-based labels) \n")
	    tabDMLScript.append("y_full = table(seq(1," + Caffe2DML.numImages + ",1), y_full, " + Caffe2DML.numImages + ", " + Utils.numClasses(net) + ")\n")
	  }
	  else {
	    tabDMLScript.append(Caffe2DML.numImages + " = nrow(X_full)\n")
	  }
  }
  
  def initWeights(net:CaffeNetwork, solver:CaffeSolver, readWeights:Boolean): Unit = {
    initWeights(net, solver, readWeights, new HashSet[String]())
  }
  
  def initWeights(net:CaffeNetwork, solver:CaffeSolver, readWeights:Boolean, layersToIgnore:HashSet[String]): Unit = {
    tabDMLScript.append("weights = ifdef($weights, \" \")\n")
	  // Initialize the layers and solvers
	  tabDMLScript.append("# Initialize the layers and solvers\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).init(tabDMLScript))
	  if(readWeights) {
		  // Loading existing weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias
		  tabDMLScript.append("# Load the weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias\n")
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => tabDMLScript.append(readWeight(l.weight, l.param.getName + "_weight.mtx")))
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => tabDMLScript.append(readWeight(l.bias, l.param.getName + "_bias.mtx")))
	  }
	  net.getLayers.map(layer => solver.init(tabDMLScript, net.getCaffeLayer(layer)))
  }
  
  def getLossLayers(net:CaffeNetwork):List[IsLossLayer] = {
    val lossLayers = net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[IsLossLayer]).map(layer => net.getCaffeLayer(layer).asInstanceOf[IsLossLayer])
	  if(lossLayers.length != 1) 
	    throw new DMLRuntimeException("Expected exactly one loss layer, but found " + lossLayers.length + ":" + net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[IsLossLayer]))
	  lossLayers
  }
  
  def updateMeanVarianceForBatchNorm(net:CaffeNetwork, value:Boolean):Unit = {
    net.getLayers.filter(net.getCaffeLayer(_).isInstanceOf[BatchNorm]).map(net.getCaffeLayer(_).asInstanceOf[BatchNorm].update_mean_var = value)
  }
}