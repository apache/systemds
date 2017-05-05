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
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer


object Caffe2DML  {
  val LOG = LogFactory.getLog(classOf[Caffe2DML].getName()) 
  // ------------------------------------------------------------------------
  def layerDir = "nn/layers/"
  def optimDir = "nn/optim/"
  
  // Naming conventions:
  val X = "X"; val y = "y"; val batchSize = "BATCH_SIZE"; val numImages = "num_images"; val numValidationImages = "num_validation"
  val XVal = "X_val"; val yVal = "y_val"
  
  var USE_NESTEROV_UDF = {
    // Developer environment variable flag 'USE_NESTEROV_UDF' until codegen starts working.
    // Then, we will remove this flag and also the class org.apache.sysml.udf.lib.SGDNesterovUpdate
    val envFlagNesterovUDF = System.getenv("USE_NESTEROV_UDF")
    envFlagNesterovUDF != null && envFlagNesterovUDF.toBoolean
  }
}

class Caffe2DML(val sc: SparkContext, val solverParam:Caffe.SolverParameter, 
    val solver:CaffeSolver, val net:CaffeNetwork, 
    val lrPolicy:LearningRatePolicy, val numChannels:String, val height:String, val width:String) extends Estimator[Caffe2DMLModel] 
  with BaseSystemMLClassifier with DMLGenerator {
  // --------------------------------------------------------------
  // Invoked by Python, MLPipeline
  def this(sc: SparkContext, solver1:Caffe.SolverParameter, networkPath:String, numChannels:String, height:String, width:String) {
    this(sc, solver1, Utils.parseSolver(solver1), 
        new CaffeNetwork(networkPath, caffe.Caffe.Phase.TRAIN, numChannels, height, width),
        new LearningRatePolicy(solver1), numChannels, height, width)
  }
  def this(sc: SparkContext, solver1:Caffe.SolverParameter, numChannels:String, height:String, width:String) {
    this(sc, solver1, Utils.parseSolver(solver1), new CaffeNetwork(solver1.getNet, caffe.Caffe.Phase.TRAIN, numChannels, height, width), 
        new LearningRatePolicy(solver1), numChannels, height, width)
  } 
  val uid:String = "caffe_classifier_" + (new Random).nextLong
  override def copy(extra: org.apache.spark.ml.param.ParamMap): Estimator[Caffe2DMLModel] = {
    val that = new Caffe2DML(sc, solverParam, solver, net, lrPolicy, numChannels, height, width)
    copyValues(that, extra)
  }
  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): Caffe2DMLModel = {
    val ret = baseFit(X_mb, y_mb, sc)
    new Caffe2DMLModel(ret, Utils.numClasses(net), sc, solver, net, lrPolicy, this)
  }
  def fit(df: ScriptsUtils.SparkDataType): Caffe2DMLModel = {
    val ret = baseFit(df, sc)
    new Caffe2DMLModel(ret, Utils.numClasses(net), sc, solver, net, lrPolicy, this)
  }
	// --------------------------------------------------------------
  
  // Used for simplifying transfer learning
  private val layersToIgnore:HashSet[String] = new HashSet[String]() 
  def setWeightsToIgnore(layerName:String):Unit = layersToIgnore.add(layerName)
  def setWeightsToIgnore(layerNames:ArrayList[String]):Unit = layersToIgnore.addAll(layerNames)
  	  
  // Input parameters to prediction and scoring script
  val inputs:java.util.HashMap[String, String] = new java.util.HashMap[String, String]()
  def setInput(key: String, value:String):Unit = inputs.put(key, value)
  customAssert(solverParam.getTestIterCount <= 1, "Multiple test_iter variables are not supported")
  customAssert(solverParam.getMaxIter > 0, "Please set max_iter to a positive value")
  customAssert(net.getLayers.filter(net.getCaffeLayer(_).isInstanceOf[IsLossLayer]).length == 1, "Expected exactly one loss layer")
    
  // TODO: throw error or warning if user tries to set solver_mode == GPU instead of using setGPU method
  
  // Method called by Python mllearn to visualize variable of certain layer
  def visualizeLayer(layerName:String, varType:String, aggFn:String): Unit = visualizeLayer(net, layerName, varType, aggFn)
  
  // -------------------------------------------------------------------------------------------
  // Helper functions to generate DML
  // Initializes Caffe2DML.X, Caffe2DML.y, Caffe2DML.XVal, Caffe2DML.yVal and Caffe2DML.numImages
  private def trainTestSplit(numValidationBatches:Int):Unit = {
    if(numValidationBatches > 0) {
      if(solverParam.getDisplay <= 0) 
        throw new DMLRuntimeException("Since test_iter and test_interval is greater than zero, you should set display to be greater than zero")
      tabDMLScript.append(Caffe2DML.numValidationImages).append(" = " + numValidationBatches + " * " + Caffe2DML.batchSize + "\n")
      tabDMLScript.append("# Sanity check to ensure that validation set is not too large\n")
      val maxValidationSize = "ceil(0.3 * " + Caffe2DML.numImages + ")"
      ifBlock(Caffe2DML.numValidationImages  + " > " + maxValidationSize) {
        assign(tabDMLScript, "max_test_iter", "floor(" + maxValidationSize + " / " + Caffe2DML.batchSize + ")")
        tabDMLScript.append("stop(" +
            dmlConcat(asDMLString("Too large validation size. Please reduce test_iter to "), "max_test_iter") 
            + ")\n")
      }
      val one = "1"
      val rl = int_add(Caffe2DML.numValidationImages, one)
      rightIndexing(tabDMLScript.append(Caffe2DML.X).append(" = "), "X_full", rl, Caffe2DML.numImages, null, null)
      tabDMLScript.append("; ")
      rightIndexing(tabDMLScript.append(Caffe2DML.y).append(" = "), "y_full", rl, Caffe2DML.numImages, null, null)
      tabDMLScript.append("; ")
      rightIndexing(tabDMLScript.append(Caffe2DML.XVal).append(" = "), "X_full", one, Caffe2DML.numValidationImages, null, null)
      tabDMLScript.append("; ")
      rightIndexing(tabDMLScript.append(Caffe2DML.yVal).append(" = "), "y_full", one, Caffe2DML.numValidationImages, null, null)
      tabDMLScript.append("; ")
      tabDMLScript.append(Caffe2DML.numImages).append(" = nrow(y)\n")
    }
    else {
      assign(tabDMLScript, Caffe2DML.X, "X_full")
	    assign(tabDMLScript, Caffe2DML.y, "y_full")
	    tabDMLScript.append(Caffe2DML.numImages).append(" = nrow(" + Caffe2DML.y + ")\n")
    }
  }
  
  private def printClassificationReport():Unit = {
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
  
  // Append the DML to display training and validation loss
  private def displayLoss(lossLayer:IsLossLayer, shouldValidate:Boolean):Unit = {
    if(solverParam.getDisplay > 0) {
      // Append the DML to compute training loss
      tabDMLScript.append("# Compute training loss & accuracy\n")
      ifBlock("iter  %% " + solverParam.getDisplay + " == 0") {
        assign(tabDMLScript, "loss", "0"); assign(tabDMLScript, "accuracy", "0")
        lossLayer.computeLoss(dmlScript, numTabs)
        assign(tabDMLScript, "training_loss", "loss"); assign(tabDMLScript, "training_accuracy", "accuracy")
        tabDMLScript.append(print( dmlConcat( asDMLString("Iter:"), "iter", 
            asDMLString(", training loss:"), "training_loss", asDMLString(", training accuracy:"), "training_accuracy" )))
        appendTrainingVisualizationBody(dmlScript, numTabs)
        printClassificationReport
      }
      if(shouldValidate) {
        // Append the DML to compute validation loss
        val numValidationBatches = if(solverParam.getTestIterCount > 0) solverParam.getTestIter(0) else 0
        tabDMLScript.append("# Compute validation loss & accuracy\n")
        ifBlock("iter  %% " + solverParam.getTestInterval + " == 0") {
          assign(tabDMLScript, "loss", "0"); assign(tabDMLScript, "accuracy", "0")
          solverParam.getTestAlgo.toLowerCase match {
            case "minibatch" => {
              assign(tabDMLScript, "validation_loss", "0")
              assign(tabDMLScript, "validation_accuracy", "0")
              forBlock("iVal", "1", "num_iters_per_epoch") {
    	          getValidationBatch(tabDMLScript)
    	          tabDMLScript.append("iter = start_iter + i\n")
    	          forward;  lossLayer.computeLoss(dmlScript, numTabs)
                tabDMLScript.append("validation_loss = validation_loss + loss\n")
                tabDMLScript.append("validation_accuracy = validation_accuracy + accuracy\n")
    	        }
              tabDMLScript.append("validation_accuracy = validation_accuracy / num_iters_per_epoch\n")
            }
            case "batch" => {
              assign(tabDMLScript, "Xb", Caffe2DML.XVal); assign(tabDMLScript, "yb", Caffe2DML.yVal)
              net.getLayers.map(layer => net.getCaffeLayer(layer).forward(tabDMLScript, false))
              lossLayer.computeLoss(dmlScript, numTabs)
              assign(tabDMLScript, "validation_loss", "loss"); assign(tabDMLScript, "validation_accuracy", "accuracy")
              
            }
            case _ => throw new DMLRuntimeException("Unsupported test algo:" + solverParam.getTestAlgo)
          }
          tabDMLScript.append(print( dmlConcat( asDMLString("Iter:"), "iter", 
              asDMLString(", validation loss:"), "validation_loss", asDMLString(", validation accuracy:"), "validation_accuracy" )))
          appendValidationVisualizationBody(dmlScript, numTabs)
        }
      }
    }
  }
  
  private def performSnapshot():Unit = {
    if(solverParam.getSnapshot > 0) {
      ifBlock("iter %% snapshot == 0") {
        tabDMLScript.append("snapshot_dir= \"" + solverParam.getSnapshotPrefix + "\" + \"/iter_\" + iter + \"/\"\n")
        net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => tabDMLScript.append(write(l.weight, "snapshot_dir + \"" + l.param.getName + "_weight.mtx\"", "binary")))
  		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => tabDMLScript.append(write(l.bias, "snapshot_dir + \"" + l.param.getName + "_bias.mtx\"", "binary")))
      }
  	}
  }
  
  private def forward():Unit = {
    tabDMLScript.append("# Perform forward pass\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).forward(tabDMLScript, false))
  }
  private def backward():Unit = backward("")
  private def backward(suffix:String):Unit = {
    tabDMLScript.append("# Perform backward pass\n")
    net.getLayers.reverse.map(layer => net.getCaffeLayer(layer).backward(tabDMLScript, suffix))
  }
  private def update():Unit = {
    tabDMLScript.append("# Update the parameters\n")
    net.getLayers.map(layer => solver.update(tabDMLScript, net.getCaffeLayer(layer)))
  }
  private def initAggGradients():Unit = {
    tabDMLScript.append("# Data structure to store gradients computed in parallel")
    net.getLayers.map(layer => net.getCaffeLayer(layer)).map(l => {
      if(l.shouldUpdateWeight) assign(tabDMLScript, l.dWeight + "_agg", matrix("0", "parallel_batches", multiply(nrow(l.weight), ncol(l.weight))))
      if(l.shouldUpdateBias) assign(tabDMLScript, l.dBias + "_agg", matrix("0", "parallel_batches", multiply(nrow(l.bias), ncol(l.bias)))) 
    })
  }
  private def flattenAndStoreAggGradients_j():Unit = {
    tabDMLScript.append("# Flatten and store gradients for this parallel execution\n")
    net.getLayers.map(layer => net.getCaffeLayer(layer)).map(l => {
      if(l.shouldUpdateWeight) assign(tabDMLScript, l.dWeight + "_agg[j,]", 
          matrix(l.dWeight, "1", multiply(nrow(l.weight), ncol(l.weight)))) 
      if(l.shouldUpdateWeight) assign(tabDMLScript, l.dBias + "_agg[j,]", 
          matrix(l.dBias, "1", multiply(nrow(l.bias), ncol(l.bias))))
    })
  }
  private def aggregateAggGradients():Unit = {
    tabDMLScript.append("# Aggregate the gradients\n")
    net.getLayers.map(layer => net.getCaffeLayer(layer)).map(l => {
      if(l.shouldUpdateWeight) assign(tabDMLScript, l.dWeight, 
          matrix(colSums(l.dWeight + "_agg"), nrow(l.weight), ncol(l.weight))) 
      if(l.shouldUpdateWeight) assign(tabDMLScript, l.dBias, 
          matrix(colSums(l.dBias + "_agg"), nrow(l.bias), ncol(l.bias)))
    })
  }
  // -------------------------------------------------------------------------------------------
  
  private def multiply(v1:String, v2:String):String = v1 + "*" + v2
  private def colSums(m:String):String = "colSums(" + m + ")"
  
	// Script generator
	def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
	  val startTrainingTime = System.nanoTime()
	  val DEBUG_TRAINING = if(inputs.containsKey("$debug")) inputs.get("$debug").toLowerCase.toBoolean else false
    reset()
	  
	  // Add source for layers as well as solver as well as visualization header
	  source(net, solver, Array[String]("l2_reg"))
	  appendVisualizationHeaders(dmlScript, numTabs)
	  
	  if(Caffe2DML.USE_NESTEROV_UDF) {
	    tabDMLScript(dmlScript, numTabs).append("update_nesterov = externalFunction(matrix[double] X, matrix[double] dX, double lr, double mu, matrix[double] v) return (matrix[double] X, matrix[double] v) implemented in (classname=\"org.apache.sysml.udf.lib.SGDNesterovUpdate\",exectype=\"mem\");  \n")
	  }
	  
	  // Read and convert to one-hote encoding
	  assign(tabDMLScript, "X_full", "read(\" \", format=\"csv\")")
	  assign(tabDMLScript, "y_full", "read(\" \", format=\"csv\")")
	  tabDMLScript.append(Caffe2DML.numImages).append(" = nrow(y_full)\n")
	  tabDMLScript.append("weights = ifdef($weights, \" \")\n")
	  tabDMLScript.append("debug = ifdef($debug, FALSE)\n")
	  tabDMLScript.append("# Convert to one-hot encoding (Assumption: 1-based labels) \n")
	  tabDMLScript.append("y_full = table(seq(1," + Caffe2DML.numImages + ",1), y_full, " + Caffe2DML.numImages + ", " + Utils.numClasses(net) + ")\n")
	  
	  // Initialize the layers and solvers
	  tabDMLScript.append("# Initialize the layers and solvers\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).init(tabDMLScript))
	  if(inputs.containsKey("$weights")) {
		  // Loading existing weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias
		  tabDMLScript.append("# Load the weights. Note: keeping the initialization code in case the layer wants to initialize non-weights and non-bias\n")
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => tabDMLScript.append(read(l.weight, l.param.getName + "_weight.mtx")))
		  net.getLayers.filter(l => !layersToIgnore.contains(l)).map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => tabDMLScript.append(read(l.bias, l.param.getName + "_bias.mtx")))
	  }
	  net.getLayers.map(layer => solver.init(tabDMLScript, net.getCaffeLayer(layer)))
	  
	  // Split into training and validation set
	  // Initializes Caffe2DML.X, Caffe2DML.y, Caffe2DML.XVal, Caffe2DML.yVal and Caffe2DML.numImages
	  val shouldValidate = solverParam.getTestInterval > 0 && solverParam.getTestIterCount > 0 && solverParam.getTestIter(0) > 0
	  trainTestSplit(if(shouldValidate) solverParam.getTestIter(0) else 0)
	  
	  // Set iteration-related variables such as max_epochs, num_iters_per_epoch, lr, etc.
	  val lossLayers = net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[IsLossLayer]).map(layer => net.getCaffeLayer(layer).asInstanceOf[IsLossLayer])
	  if(lossLayers.length != 1) throw new DMLRuntimeException("Expected exactly one loss layer")
	  solverParam.getTrainAlgo.toLowerCase match {
	    case "batch" => 
	      assign(tabDMLScript, "max_epochs", solverParam.getMaxIter.toString)
	    case _ => {
	      ceilDivide(tabDMLScript, "num_iters_per_epoch", Caffe2DML.numImages, Caffe2DML.batchSize)
	      ceilDivide(tabDMLScript, "max_epochs", solverParam.getMaxIter.toString, "num_iters_per_epoch")
	    }
	  }
	  assign(tabDMLScript, "start_iter", "0")
	  assign(tabDMLScript, "lr", solverParam.getBaseLr.toString)
	  
	  // ----------------------------------------------------------------------------
	  // Main logic
	  forBlock("e", "1", "max_epochs") {
	    solverParam.getTrainAlgo.toLowerCase match {
	      case "minibatch" => 
	        forBlock("i", "1", "num_iters_per_epoch") {
	          getTrainingBatch(tabDMLScript)
	          tabDMLScript.append("iter = start_iter + i\n")
	          forward; backward; update
	          displayLoss(lossLayers(0), shouldValidate)
            performSnapshot
	        }
	      case "batch" => {
          tabDMLScript.append("iter = start_iter + i\n")
          forward; backward; update
          displayLoss(lossLayers(0), shouldValidate)
          performSnapshot
	      }
	      case "allreduce" => {
	        forBlock("i", "1", "num_iters_per_epoch") {
	          getTrainingBatch(tabDMLScript)
	          assign(tabDMLScript, "X_group_batch", "Xb")
	          assign(tabDMLScript, "y_group_batch", "yb")
	          tabDMLScript.append("iter = start_iter + i\n")
	          initAggGradients
	          parForBlock("j", "1", "nrow(y_group_batch)") {
	            assign(tabDMLScript, "Xb", "X_group_batch[j,]")
	            assign(tabDMLScript, "yb", "y_group_batch[j,]")
	            forward; backward("_agg")
              flattenAndStoreAggGradients_j
	          }
	          aggregateAggGradients
            tabDMLScript.append("iter = start_iter + parallel_batches\n")    
	          update
            displayLoss(lossLayers(0), shouldValidate)
            performSnapshot
	        }
	      }
	      case _ => throw new DMLRuntimeException("Unsupported train algo:" + solverParam.getTrainAlgo)
	    }
	    // After every epoch, update the learning rate
	    tabDMLScript.append("# Learning rate\n")
	    lrPolicy.updateLearningRate(tabDMLScript)
	    tabDMLScript.append("start_iter = start_iter + num_iters_per_epoch\n")
	  }
	  // ----------------------------------------------------------------------------
	  
	  // Check if this is necessary
	  if(doVisualize) tabDMLScript.append("print(" + asDMLString("Visualization counter:") + " + viz_counter)")
	  
	  val trainingScript = tabDMLScript.toString()
	  // Print script generation time and the DML script on stdout
	  System.out.println("Time taken to generate training script from Caffe proto: " + ((System.nanoTime() - startTrainingTime)*1e-9) + " seconds." )
	  if(DEBUG_TRAINING) Utils.prettyPrintDMLScript(trainingScript)
	  
	  // Set input/output variables and execute the script
	  val script = dml(trainingScript).in(inputs)
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.out(l.weight))
	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.out(l.bias))
	  (script, "X_full", "y_full")
	}
}

class Caffe2DMLModel(val mloutput: MLResults,  
    val numClasses:String, val sc: SparkContext, val solver:CaffeSolver,
    val net:CaffeNetwork, val lrPolicy:LearningRatePolicy,
    val estimator:Caffe2DML) 
  extends Model[Caffe2DMLModel] with HasMaxOuterIter with BaseSystemMLClassifierModel with DMLGenerator {
  // --------------------------------------------------------------
  // Invoked by Python, MLPipeline
  val uid:String = "caffe_model_" + (new Random).nextLong 
  def this(estimator:Caffe2DML) =  {
    this(null, Utils.numClasses(estimator.net), estimator.sc, estimator.solver,
        estimator.net,
        // new CaffeNetwork(estimator.solverParam.getNet, caffe.Caffe.Phase.TEST, estimator.numChannels, estimator.height, estimator.width), 
        estimator.lrPolicy, estimator) 
  }
      
  override def copy(extra: org.apache.spark.ml.param.ParamMap): Caffe2DMLModel = {
    val that = new Caffe2DMLModel(mloutput, numClasses, sc, solver, net, lrPolicy, estimator)
    copyValues(that, extra)
  }
  // --------------------------------------------------------------
  
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
	}
    
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String)  = {
    reset()
    val startPredictionTime = System.nanoTime()
	  val DEBUG_PREDICTION = if(estimator.inputs.containsKey("$debug")) estimator.inputs.get("$debug").toLowerCase.toBoolean else false
	  
	  // Append source statements for each layer
	  source(net, solver, null)
    tabDMLScript.append("weights = ifdef($weights, \" \")\n")
	  // Initialize the layers and solvers
	  tabDMLScript.append("# Initialize the layers and solvers\n")
	  net.getLayers.map(layer => net.getCaffeLayer(layer).init(tabDMLScript))
	  if(mloutput == null && estimator.inputs.containsKey("$weights")) {
		  // fit was not called
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => tabDMLScript.append(read(l.weight, l.param.getName + "_weight.mtx")))
		  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => tabDMLScript.append(read(l.bias, l.param.getName + "_bias.mtx")))
	  }
	  else if(mloutput == null) {
		  throw new DMLRuntimeException("Cannot call predict/score without calling either fit or by providing weights")
	  }
	  net.getLayers.map(layer => solver.init(tabDMLScript, net.getCaffeLayer(layer)))
	  
//	  if(estimator.inputs.containsKey("$debug") && estimator.inputs.get("$debug").equals("TRUE")) {
//		  System.out.println("The output shape of layers:")
//		  net.getLayers.map(layer =>  System.out.println(net.getCaffeLayer(layer).param.getName + " " + net.getCaffeLayer(layer).outputShape))
//	  }
	  
	  // Donot update mean and variance in batchnorm
	  net.getLayers.filter(net.getCaffeLayer(_).isInstanceOf[BatchNorm]).map(net.getCaffeLayer(_).asInstanceOf[BatchNorm].update_mean_var = false)
	  tabDMLScript.append("X_full = read(\" \", format=\"csv\")\n")
	  assign(tabDMLScript, "X", "X_full")
	  tabDMLScript.append(Caffe2DML.numImages + " = nrow(X_full)\n")
	  
	  val lossLayers = net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[IsLossLayer]).map(layer => net.getCaffeLayer(layer).asInstanceOf[IsLossLayer])
	  customAssert(lossLayers.length == 1, "Expected exactly one loss layer, but found " + lossLayers.length + ":" + net.getLayers.filter(layer => net.getCaffeLayer(layer).isInstanceOf[IsLossLayer]))
	  assign(tabDMLScript, "Prob", matrix("0", Caffe2DML.numImages, numClasses))
	  estimator.solverParam.getTestAlgo.toLowerCase match {
      case "minibatch" => {
        ceilDivide(tabDMLScript(), "num_iters", Caffe2DML.numImages, Caffe2DML.batchSize)
        forBlock("i", "1", "num_iters") {
          getTestBatch(tabDMLScript)
          net.getLayers.map(layer => net.getCaffeLayer(layer).forward(tabDMLScript, true))
          assign(tabDMLScript, "Prob[beg:end,]", lossLayers(0).out)
        }
      }
      case "batch" => {
        assign(tabDMLScript, "Xb", "X_full")
        net.getLayers.map(layer => net.getCaffeLayer(layer).forward(tabDMLScript, true))
        assign(tabDMLScript, "Prob", lossLayers(0).out)
      }
      case "allreduce" => {
        ceilDivide(tabDMLScript(), "num_iters", Caffe2DML.numImages, Caffe2DML.batchSize)
        parForBlock("i", "1", "num_iters") {
          getTestBatch(tabDMLScript)
          net.getLayers.map(layer => net.getCaffeLayer(layer).forward(tabDMLScript, true))
          assign(tabDMLScript, "Prob[beg:end,]", lossLayers(0).out)
        }
      }
      case _ => throw new DMLRuntimeException("Unsupported test algo:" + estimator.solverParam.getTestAlgo)
    }
		
		val predictionScript = dmlScript.toString()
		System.out.println("Time taken to generate prediction script from Caffe proto:" + ((System.nanoTime() - startPredictionTime)*1e-9) + "secs." )
		if(DEBUG_PREDICTION) Utils.prettyPrintDMLScript(predictionScript)
		
		// Reset
		net.getLayers.filter(net.getCaffeLayer(_).isInstanceOf[BatchNorm]).map(net.getCaffeLayer(_).asInstanceOf[BatchNorm].update_mean_var = true)
		
	  val script = dml(predictionScript).out("Prob").in(estimator.inputs)
	  if(mloutput != null) {
	    // fit was called
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.weight != null).map(l => script.in(l.weight, mloutput.getBinaryBlockMatrix(l.weight)))
  	  net.getLayers.map(net.getCaffeLayer(_)).filter(_.bias != null).map(l => script.in(l.bias, mloutput.getBinaryBlockMatrix(l.bias)))
	  }
	  
	  (script, "X_full")
  }
  
  // Prediction
  def transform(X: MatrixBlock): MatrixBlock = {
	  baseTransform(X, mloutput, sc, "Prob")
  }
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = {
	  baseTransform(df, mloutput, sc, "Prob")
  }
}