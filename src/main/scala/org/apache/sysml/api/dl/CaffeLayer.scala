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

import caffe.Caffe.LayerParameter
import scala.collection.JavaConversions._
import org.apache.sysml.parser.LanguageException
import java.util.HashSet
import java.io.File
import org.apache.sysml.api.DMLScript
import org.apache.sysml.runtime.util.ConvolutionUtils
import caffe.Caffe.EltwiseParameter.EltwiseOp
import org.apache.sysml.runtime.DMLRuntimeException;
import java.util.ArrayList

trait CaffeLayer extends BaseDMLGenerator {
  // -------------------------------------------------
  // Any layer that wants to reuse SystemML-NN has to override following methods that help in generating the DML for the given layer:
  def sourceFileName:String;
  def init(dmlScript:StringBuilder):Unit;
  def forward(dmlScript:StringBuilder, isPrediction:Boolean):Unit;
  def backward(dmlScript:StringBuilder, outSuffix:String):Unit;
  var computedOutputShape:(String, String, String) = null
  def outputShape:(String, String, String) = {
    if(computedOutputShape == null) computedOutputShape = bottomLayerOutputShape
    computedOutputShape
  }
  // -------------------------------------------------
  var computedBottomLayerOutputShape:(String, String, String) = null
  def bottomLayerOutputShape:(String, String, String) = {
    if(computedBottomLayerOutputShape == null) {
      val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
      if(ret.size == 0) throw new LanguageException("Expected atleast 1 bottom layer for " + param.getName)
      computedBottomLayerOutputShape = ret(0).outputShape
    }
    computedBottomLayerOutputShape
  }
  def param:LayerParameter
  def id:Int
  def net:CaffeNetwork
  // --------------------------------------------------------------------------------------
  // No need to override these methods in subclasses
  // Exception: Only Data layer overrides "out" method to use 'Xb' for consistency
  // Naming of the below methods is consistent with the nn library:
  // X (feature map from the previous layer) ----> Forward pass  ----> out (feature map to the next layer)
  // dX (errors to the previous layer)       <---- Backward pass <---- dout (errors from the next layer)
  def out = "out" + id  
  var computedX:String = null
  def X:String = {
    if(computedX == null) {
      val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
      if(ret.size == 0) throw new LanguageException("Expected atleast 1 bottom layer for " + param.getName)
      else if(ret.size == 1)    computedX = ret(0).out
      else                      computedX = sum(new StringBuilder, ret.map(_.out).toList).toString()
    }
    computedX
  }
  var computedDout:String = null
  def dout: String = {
    if(computedDout == null) {
      val ret = net.getTopLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
      if(ret.size == 0) throw new LanguageException("Expected atleast 1 top layer for " + param.getName)
      else if(ret.size == 1)     computedDout = ret(0).dX(id)
      else                       computedDout = sum(new StringBuilder, ret.map(_.dX(id)).toList).toString()
    }
    computedDout
  }
  def dX(bottomLayerID:Int) = "dOut" + id + "_" + bottomLayerID
  // --------------------------------------------------------------------------------------
  // No need to override these methods in subclasses, instead classes that have weights and biases 
  // should implement HasWeight and HasBias traits.
  def dWeight():String = throw new DMLRuntimeException("dWeight is not implemented in super class")
  def dBias():String = throw new DMLRuntimeException("dBias is not implemented in super class")
  def weight():String = null;
  def weightShape():Array[Int];
  def bias():String = null;
  def biasShape():Array[Int];
  def shouldUpdateWeight():Boolean = if(weight != null) true else false
  def shouldUpdateBias():Boolean = if(bias != null) true else false
  // --------------------------------------------------------------------------------------
  // Helper methods to simplify the code of subclasses
  def invokeInit(dmlScript:StringBuilder, returnVariables:List[String], arguments:String*):Unit = {
    invoke(dmlScript, sourceFileName + "::", returnVariables, "init", arguments.toList)
  }
  def invokeForward(dmlScript:StringBuilder, returnVariables:List[String], arguments:String*):Unit = {
    invoke(dmlScript, sourceFileName + "::", returnVariables, "forward", arguments.toList)
  }
  // -----------------------------------------------------------------------------------
  // All the layers (with the exception of Concat) call one of the below methods in the backward function.
  // The preceding layer expects that 'dX(bottomLayerID) + outSuffix' is assigned. 
  // l1 <--- dX(1) <-----|
  //                     |-- [current layer: dOut3 (computed by backward)] <---- "dOut" + id + outSuffix
  // l2 <--- dX(2) <-----|
  // The below functions perform two functions:
  // 1. Compute backward: either call dml file's backward (for example: invokeBackward) or just propagate next layers errors (assignDoutToDX)
  // 2. Then make sure that all the preceding layer get the errors using:
  //        bottomLayerIDs.map(bottomLayerID => dmlScript.append( dX(bottomLayerID) + outSuffix + " = " + "dOut" + id + outSuffix + "; "))
  
  // The layers that have a corresponding dml script call this method.
  // Assumption: the first variable of resultVariables is always dX
  def invokeBackward(dmlScript:StringBuilder, outSuffix:String, resultVariables:List[String],  arguments:String*):Unit = {
    invoke(dmlScript, sourceFileName + "::", resultVariables.map(_ + outSuffix), "backward", arguments.toList, false)
    val bottomLayerIDs = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l).id)
    dmlScript.append("; ")
    bottomLayerIDs.map(bottomLayerID => dmlScript.append( dX(bottomLayerID) + outSuffix + " = " + resultVariables(0) + outSuffix + "; "))
    dmlScript.append("\n")
  }
  // On-the-fly layers (such as Scale and Elementwise) call this function to propagate next layers errors to the previous layer
  def assignDoutToDX(dmlScript:StringBuilder, outSuffix:String):Unit = {
    dmlScript.append("dOut" + id  + outSuffix + " = " + dout)
    val bottomLayerIDs = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l).id)
    dmlScript.append("; ")
    bottomLayerIDs.map(bottomLayerID => dmlScript.append( dX(bottomLayerID) + outSuffix + " = " + "dOut" + id + outSuffix + "; "))
    dmlScript.append("\n")
  }
  // --------------------------------------------------------------------------------------
}


trait IsLossLayer extends CaffeLayer {
  def computeLoss(dmlScript:StringBuilder, numTabs:Int):Unit
}

trait HasWeight extends CaffeLayer {
  override def weight = param.getName + "_weight"
  override def dWeight = param.getName + "_dWeight"
}

trait HasBias extends CaffeLayer {
  override def bias = param.getName + "_bias"
  override def dBias = param.getName + "_dBias"
}

class Data(val param:LayerParameter, val id:Int, val net:CaffeNetwork, val numChannels:String, val height:String, val width:String) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = {
    if(param.hasTransformParam && param.getTransformParam.hasScale) {
      dmlScript.append("X_full = X_full * " + param.getTransformParam.getScale + "\n")
    }
    if(param.hasDataParam && param.getDataParam.hasBatchSize) {
      dmlScript.append("BATCH_SIZE = " + param.getDataParam.getBatchSize + "\n")
    }
    else {
      Caffe2DML.LOG.debug("Using default batch size of 64 as batch size is not set with DataParam")
      dmlScript.append("BATCH_SIZE = 64\n")
    }
  }
  var dataOutputShape = ("$num_channels", "$height", "$width")
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = { }
  override def out = "Xb"
  override def backward(dmlScript:StringBuilder, outSuffix:String) = { }
  override def outputShape = (numChannels, height, width)
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
  // -------------------------------------------------
}


// ------------------------------------------------------------------
// weight is ema_mean and bias is ema_var
// Fuse 
class BatchNorm(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // val scale =  
  override def sourceFileName = "batch_norm2d"
  /*
   * Initialize the parameters of this layer.
   *
   * Note: This is just a convenience function, and parameters
   * may be initialized manually if needed.
   *
   * Inputs:
   *  - C: Number of input channels (dimensionality of input depth).
   *
   * Outputs:
   *  - gamma: Scale parameters, of shape (C, 1).
   *  - beta: Shift parameters, of shape (C, 1).
   *  - ema_mean: Exponential moving average of the mean, of
   *      shape (C, 1).
   *  - ema_var: Exponential moving average of the variance, of
   *      shape (C, 1).
   */
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](gamma, beta, ema_mean, ema_var), numChannels)
  var update_mean_var = true
  /*
   * Computes the forward pass for a 2D (spatial) batch normalization
   * layer.  The input data has N examples, each represented as a 3D
   * volume unrolled into a single vector.
   *
   * A spatial batch normalization layer uses the per-channel sample
   * mean and per-channel uncorrected sample variance during training
   * to normalize each channel of the input data.  Additionally, it
   * introduces learnable parameters (gamma, beta) to control the
   * amount of normalization.
   *
   *   `y = ((x-mean) / sqrt(var+eps)) * gamma + beta`
   *
   * This implementation maintains exponential moving averages of the
   * mean and variance during training for use during testing.
   *
   * Reference:
   *  - Batch Normalization: Accelerating Deep Network Training by
   *    Reducing Internal Covariate Shift, S. Ioffe & C. Szegedy, 2015
   *    - https://arxiv.org/abs/1502.03167
   *
   * Inputs:
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - gamma: Scale parameters, of shape (C, 1).
   *  - beta: Shift parameters, of shape (C, 1).
   *  - C: Number of input channels (dimensionality of input depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - mode: 'train' or 'test' to indicate if the model is currently
   *      being trained or tested.  During training, the current batch
   *      mean and variance will be used to normalize the inputs, while
   *      during testing, the exponential average of the mean and
   *      variance over all previous batches will be used.
   *  - ema_mean: Exponential moving average of the mean, of
   *      shape (C, 1).
   *  - ema_var: Exponential moving average of the variance, of
   *      shape (C, 1).
   *  - mu: Momentum value for moving averages.
   *      Typical values are in the range of [0.9, 0.999].
   *  - epsilon: Smoothing term to avoid divide by zero errors.
   *      Typical values are in the range of [1e-5, 1e-3].
   *
   * Outputs:
   *  - out: Outputs, of shape (N, C*Hin*Win).
   *  - ema_mean_upd: Updated exponential moving average of the mean,
   *      of shape (C, 1).
   *  - ema_var_upd: Updated exponential moving average of the variance,
   *      of shape (C, 1).
   *  - cache_mean: Cache of the batch mean, of shape (C, 1).
   *      Note: This is used for performance during training.
   *  - cache_var: Cache of the batch variance, of shape (C, 1).
   *      Note: This is used for performance during training.
   *  - cache_norm: Cache of the normalized inputs, of
   *      shape (C, N*Hin*Win). Note: This is used for performance
   *      during training.
   */
  def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = {
    val mode = if(isPrediction) "\"test\"" else "\"train\""
    invokeForward(dmlScript, List[String](out, withSuffix(ema_mean), withSuffix(ema_var), withSuffix(cache_mean), withSuffix(cache_var), withSuffix(cache_norm)), 
        X, gamma, beta, numChannels, Hin, Win, mode, ema_mean, ema_var,  ma_fraction, eps)  
  }
  /*
   * Computes the backward pass for a 2D (spatial) batch normalization
   * layer.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of shape (N, C*Hin*Win).
   *  - out: Outputs from the forward pass, of shape (N, C*Hin*Win).
   *  - ema_mean_upd: Updated exponential moving average of the mean
   *      from the forward pass, of shape (C, 1).
   *  - ema_var_upd: Updated exponential moving average of the variance
   *      from the forward pass, of shape (C, 1).
   *  - cache_mean: Cache of the batch mean from the forward pass, of
   *      shape (C, 1).  Note: This is used for performance during
   *      training.
   *  - cache_var: Cache of the batch variance from the forward pass,
   *      of shape (C, 1).  Note: This is used for performance during
   *      training.
   *  - cache_norm: Cache of the normalized inputs from the forward
   *      pass, of shape (C, N*Hin*Win).  Note: This is used for
   *      performance during training.
   *  - X: Input data matrix to the forward pass, of
   *      shape (N, C*Hin*Win).
   *  - gamma: Scale parameters, of shape (C, 1).
   *  - beta: Shift parameters, of shape (C, 1).
   *  - C: Number of input channels (dimensionality of input depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - mode: 'train' or 'test' to indicate if the model is currently
   *      being trained or tested.  During training, the current batch
   *      mean and variance will be used to normalize the inputs, while
   *      during testing, the exponential average of the mean and
   *      variance over all previous batches will be used.
   *  - ema_mean: Exponential moving average of the mean, of
   *      shape (C, 1).
   *  - ema_var: Exponential moving average of the variance, of
   *      shape (C, 1).
   *  - mu: Momentum value for moving averages.
   *      Typical values are in the range of [0.9, 0.999].
   *  - epsilon: Smoothing term to avoid divide by zero errors.
   *      Typical values are in the range of [1e-5, 1e-3].
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of shape (N, C*Hin*Win).
   *  - dgamma: Gradient wrt `W`, of shape (C, 1).
   *  - dbeta: Gradient wrt `b`, of shape (C, 1).
   *
   */
  def backward(dmlScript: StringBuilder, outSuffix:String): Unit = {
    invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id, dgamma, dbeta), dout, out, ema_mean, ema_var, cache_mean, cache_var, cache_norm, X, gamma, beta, numChannels, 
          Hin, Win, "\"train\"", ema_mean, ema_var,  ma_fraction, eps)
  }
  
  private def withSuffix(str:String):String = if(update_mean_var) str else str + "_ignore"
  override def weight = "ema_mean" + id
  override def weightShape():Array[Int] = Array(numChannels.toInt, 1)
  override def biasShape():Array[Int] = Array(numChannels.toInt, 1)
  override def bias = "ema_var" + id
  def cache_mean(): String = "cache_mean" + id
  def cache_var():String = "cache_mean" + id
  def cache_norm():String = "cache_norm" + id
  var scaleLayer:Scale = null
  def gamma():String = { checkNextLayer(); scaleLayer.weight }
  def ma_fraction():String = if(param.getBatchNormParam.hasMovingAverageFraction()) param.getBatchNormParam.getMovingAverageFraction.toString else "0.999"
  def eps():String = if(param.getBatchNormParam.hasEps()) param.getBatchNormParam.getEps.toString else "1e-5"
  def beta():String = { checkNextLayer(); scaleLayer.bias }
  def dgamma():String = { checkNextLayer();  scaleLayer.dWeight }
  def dbeta():String = { checkNextLayer();  scaleLayer.dBias }
  override def shouldUpdateWeight():Boolean = false
  override def shouldUpdateBias():Boolean = false
  def ema_mean(): String = weight
  def ema_var(): String = bias
  def checkNextLayer(): Unit = {
    if(scaleLayer == null) {
      val topLayers = net.getTopLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
      if(topLayers.length != 1 && !topLayers(0).isInstanceOf[Scale]) throw new LanguageException("Only one top layer of type Scale allowed for BatchNorm")
      scaleLayer = topLayers(0).asInstanceOf[Scale]
    }
  }
  def numChannels = bottomLayerOutputShape._1
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
}
// weight is gamma and bias is beta
class Scale(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  if(!param.getScaleParam.getBiasTerm) throw new LanguageException("Add \"scale_param { bias_term: true }\" to the layer " + param.getName)
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  // TODO: Generalize this !!
  def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = assign(dmlScript, out, X)
  override def backward(dmlScript: StringBuilder, outSuffix:String): Unit = assignDoutToDX(dmlScript, outSuffix)
  override def weightShape():Array[Int] = Array(bottomLayerOutputShape._1.toInt, 1)
  override def biasShape():Array[Int] = Array(bottomLayerOutputShape._1.toInt, 1)
}
// ------------------------------------------------------------------

class Elementwise(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  if(param.getEltwiseParam.hasOperation && param.getEltwiseParam.getOperation != EltwiseOp.SUM)
    throw new LanguageException("Currently only elementwise sum operation supported")
  override def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = {
    addAndAssign(dmlScript, out, param.getBottomList.map(b => net.getCaffeLayer(b).out).toList)
  }
  override def backward(dmlScript: StringBuilder, outSuffix:String): Unit = assignDoutToDX(dmlScript, outSuffix)
  override def outputShape = {
    if(_out == null) _out = net.getCaffeLayer(net.getBottomLayers(param.getName).take(1).toSeq.get(0)).outputShape
    _out
  }
  var _out:(String, String, String) = null
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
}

class Concat(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  var _childLayers:List[CaffeLayer] = null
  
  // Utility function to create string of format:
  // fn(fn(fn(_childLayers(0).out, _childLayers(1).out), _childLayers(2).out), ...)
  // This is useful because we do not support multi-input cbind and rbind in DML.
  def _getMultiFn(fn:String):String = {
    if(_childLayers == null) _childLayers = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    var tmp = fn + "(" + _childLayers(0).out + ", " + _childLayers(1).out + ")"
    for(i <- 2 until _childLayers.size) {
      tmp = fn + "(" + tmp + ", " +  _childLayers(i).out + ")"
    }
    tmp
  }
  
  /*
   * Computes the forward pass for a concatenation layer.
   *
   * Inputs:
   *  - n_i * c_i * h * w for each input blob i from 1 to K.
   *
   * Outputs:
   *  - out: Outputs, of shape 
   *    - if axis = 0: (n_1 + n_2 + ... + n_K) * c_1 * h * w, and all input c_i should be the same.
   *    - if axis = 1: n_1 * (c_1 + c_2 + ... + c_K) * h * w, and all input n_i should be the same.
   */
  override def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = {
    if(param.getConcatParam.getAxis == 0) {
      // rbind the inputs
      assign(dmlScript, out, _getMultiFn("rbind"))
    }
    else if(param.getConcatParam.getAxis == 1) {
      // cbind the inputs
      assign(dmlScript, out, _getMultiFn("cbind"))
    }
    else {
      throw new DMLRuntimeException("Incorrect axis parameter for the layer " + param.getName)
    }
  }
  
  def startIndex(outSuffix:String):String = "concat_start_index_" + outSuffix
  def endIndex(outSuffix:String):String = "concat_start_index_" + outSuffix
  def getConcatIndex(bottomLayerOut:String, outSuffix:String):String = 
     startIndex(outSuffix) + " = " + endIndex(outSuffix) + " + 1; " +
     endIndex(outSuffix) + " = " + startIndex(outSuffix) + " + nrow(" + bottomLayerOut + "); "
  
  /*
   * Computes the backward pass for a concatenation layer.
   *
   * The top gradients are deconcatenated back to the inputs.
   *
   */
  override def backward(dmlScript: StringBuilder, outSuffix:String): Unit = {
    val bottomLayers = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    val dOutVar = "dOut" + id  + outSuffix
    // concat_end_index = 0
    dmlScript.append(dOutVar + " = " + dout + "; concat_end_index" + outSuffix + " = 0; ")
    
    val indexString = "concat_start_index" + outSuffix + " : concat_end_index" + outSuffix
    val doutVarAssignment = if(param.getConcatParam.getAxis == 0) " = " + dOutVar + "[" +  indexString + ", ]; "
                            else " = " + dOutVar + "[," +  indexString + " ]; "
    
    // concat_start_index = concat_end_index + 1
    // concat_end_index = concat_start_index + $$ - 1
    val initializeIndexString = "concat_start_index" + outSuffix + " = concat_end_index" + outSuffix + " + 1; concat_end_index" + outSuffix + 
        " = concat_start_index" + outSuffix + " + $$ - 1; "
    if(param.getConcatParam.getAxis == 0) {
      bottomLayers.map(l => {
        dmlScript.append(initializeIndexString.replaceAll("$$", nrow(l.out)))
                  // X1 = Z[concat_start_index:concat_end_index,]
                 .append( dX(l.id) + outSuffix + doutVarAssignment)
      })
    }
    else {
      bottomLayers.map(l => {
        dmlScript.append(initializeIndexString.replaceAll("$$", int_mult(l.outputShape._1, l.outputShape._2, l.outputShape._3) ))
                  // X1 = Z[concat_start_index:concat_end_index,]
                 .append( dX(l.id) + outSuffix + doutVarAssignment)
      })
    }
    dmlScript.append("\n")
  }
  def sumChannels():String = {
    val channels = _childLayers.map(_.outputShape._1)
    try { 
      channels.reduce((c1, c2) => (c1.toInt + c2.toInt).toString())
    }
    catch { 
      case _:Throwable => sum(new StringBuilder, channels).toString
    }
  }
  override def outputShape = {
    if(_out == null) {
      if(_childLayers == null) _childLayers = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
      if(param.getConcatParam.getAxis == 0) {
        _out = _childLayers(0).outputShape
      }
      else if(param.getConcatParam.getAxis == 1) {
        _out = (sumChannels(), _childLayers(0).outputShape._2, _childLayers(0).outputShape._3)
      }
      else {
        throw new DMLRuntimeException("Incorrect axis parameter for the layer " + param.getName)
      }
    }
    _out
  }
  var _out:(String, String, String) = null
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
}

class SoftmaxWithLoss(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with IsLossLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out), scores)
  override def backward(dmlScript:StringBuilder, outSuffix:String) =  {
    invoke(dmlScript, "cross_entropy_loss::", List[String]("dProbs" + outSuffix), "backward", false, out, "yb")
    dmlScript.append("; ") 
    invoke(dmlScript, "softmax::", List[String]("dOut" + id + outSuffix), "backward", false, "dProbs", scores)
    val bottomLayerIDs = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l).id)
    dmlScript.append("; ")
    bottomLayerIDs.map(bottomLayerID => dmlScript.append( dX(bottomLayerID) + outSuffix + " = " + "dOut" + id + outSuffix + "; "))
    dmlScript.append("\n")
  }
  override def computeLoss(dmlScript:StringBuilder, numTabs:Int) = {
    val tabBuilder = new StringBuilder
    for(i <- 0 until numTabs) tabBuilder.append("\t")
    val tabs = tabBuilder.toString
    dmlScript.append("tmp_loss = cross_entropy_loss::forward(" + commaSep(out, "yb") + ")\n")
    dmlScript.append(tabs).append("loss = loss + tmp_loss\n")
    dmlScript.append(tabs).append("true_yb = rowIndexMax(yb)\n")
    dmlScript.append(tabs).append("predicted_yb = rowIndexMax(" + out + ")\n")
    dmlScript.append(tabs).append("accuracy = mean(predicted_yb == true_yb)*100\n")
  }
  def scores():String = {
	  val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
	  if(ret.size == 1) return ret.get(0).out
	  else if(ret.size == 2) {
		  val ret1 = if(!ret.get(0).out.equals("Xb")) ret.get(0).out else ""; 
		  val ret2 = if(!ret.get(1).out.equals("Xb")) ret.get(1).out else "";
		  if(!ret1.equals("") && !ret2.equals("")) throw new LanguageException("Atleast one of the output of previous layer should be Xb")
		  else if(!ret1.equals("")) return ret1
		  else return ret2
	  }
	  else 
		  throw new LanguageException("More than 2 bottom layers is not supported")
  }
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
  // -------------------------------------------------
}

class ReLU(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // TODO: Leaky ReLU: negative_slope [default 0]: specifies whether to leak the negative part by multiplying it with the slope value rather than setting it to 0.
  // -------------------------------------------------
  override def sourceFileName = "relu"
  override def init(dmlScript:StringBuilder) = { }
  /*
   * Computes the forward pass for a ReLU nonlinearity layer.
   *
   * Performs an element-wise evaluation of `f(input) = max(0, input)`.
   *
   * Inputs:
   *  - X: Inputs, of shape (any, any).
   *
   * Outputs:
   *  - out: Outputs, of same shape as `X`.
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = invokeForward(dmlScript, List[String](out), X)
  /*
   * Computes the backward pass for a ReLU nonlinearity layer.
   *
   * Essentially performs a pass-through of the upstream gradient
   * for cells > 0.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of same shape as `X`.
   *  - X: Previous input data matrix, of shape (any, any).
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of same shape as `X`.
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id), dout, X)
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
  // -------------------------------------------------
}

class Softmax(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def init(dmlScript:StringBuilder) = { }
  /*
   * Computes the forward pass for a softmax classifier.  The inputs
   * are interpreted as unnormalized, log-probabilities for each of
   * N examples, and the softmax function transforms them to normalized
   * probabilities.
   *
   * This can be interpreted as a generalization of the sigmoid
   * function to multiple classes.
   *
   *   `probs_ij = e^scores_ij / sum(e^scores_i)`
   *
   * Inputs:
   *  - scores: Inputs, of shape (N, D).
   *
   * Outputs:
   *  - probs: Outputs, of shape (N, D).
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = invokeForward(dmlScript, List[String](out), X)
  /*
   * Computes the backward pass for a softmax classifier.
   *
   * Note that dscores_ij has multiple source branches:
   *
   *   ```
   *   dprobs_ij/dscores_ij = probs_ij * (1 - probs_ij)
   *   dprobs_ik/dscores_ij = -probs_ik * probs_ij, for all k != j
   *
   *   dloss/dscores_ij =
   *      (dloss/dprobs_ij * dprobs_ij/dscores_ij)
   *      + sum_{k!=j}(dloss/dprobs_ik * dprobs_ik/dscores_ij)
   *   ```
   *
   * Inputs:
   *  - dprobs: Gradient wrt `probs` from upstream, of shape (N, D).
   *  - scores: Inputs, of shape (N, D).
   *
   * Outputs:
   *  - dscores: Gradient wrt `scores`, of shape (N, D).
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id), dout, X)
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
  // -------------------------------------------------
}


class Threshold(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = { }
  val threshold = if(param.getThresholdParam.hasThreshold) param.getThresholdParam.getThreshold else 0
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = assign(dmlScript, out, X + " > " + threshold)
  override def backward(dmlScript:StringBuilder, outSuffix:String) = throw new DMLRuntimeException("Backward operation for Threshold layer is not supported.")
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
}


class Dropout(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "dropout"
  override def init(dmlScript:StringBuilder) = { }
  /*
   * Computes the forward pass for an inverted dropout layer.
   *
   * Drops the inputs element-wise with a probability p, and divides
   * by p to maintain the expected values of those inputs (which are
   * the outputs of neurons) at test time.
   *
   * Inputs:
   *  - X: Inputs, of shape (any, any).
   *  - p: Probability of keeping a neuron output.
   *  - seed: [Optional: -1] Random number generator seed to allow for
   *      deterministic evaluation.  Set to -1 for a random seed.
   *
   * Outputs:
   *  - out: Outputs, of same shape as `X`.
   *  - mask: Dropout mask used to compute the output.
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) =
    if(!isPrediction)
      invokeForward(dmlScript, List[String](out, mask), X, p, seed)
    else
      assign(dmlScript, out, X) // Forward-pass not required to be performed during prediction for Dropout layer
  /*
   * Computes the backward pass for an inverted dropout layer.
   *
   * Applies the mask to the upstream gradient, and divides by p to
   * maintain the expected values at test time.
   *
   * Inputs:
   *  - dout: Gradient wrt `out`, of same shape as `X`.
   *  - X: Inputs, of shape (any, any).
   *  - p: Probability of keeping a neuron output.
   *  - mask: Dropout mask used to compute the output.
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of same shape as `X`.
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = invokeBackward(dmlScript, outSuffix, 
      List[String]("dOut" + id), dout, X, p, mask)
  // -------------------------------------------------
  def mask = "mask" + id
  // dropout ratio
  def p = if(param.getDropoutParam.hasDropoutRatio()) param.getDropoutParam.getDropoutRatio.toString else "0.5"
  def seed = "-1"
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
}

class InnerProduct(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // -------------------------------------------------
  // TODO: bias_filler [default type: 'constant' value: 0]; bias_term [default true]: specifies whether to learn and apply a set of additive biases to the filter outputs
  override def sourceFileName = "affine"
  /*
   * Initialize the parameters of this layer.
   *
   * Note: This is just a convenience function, and parameters
   * may be initialized manually if needed.
   *
   * We use the heuristic by He et al., which limits the magnification
   * of inputs/gradients during forward/backward passes by scaling
   * unit-Gaussian weights by a factor of sqrt(2/n), under the
   * assumption of relu neurons.
   *  - http://arxiv.org/abs/1502.01852
   *
   * Inputs:
   *  - D: Dimensionality of the input features (number of features).
   *  - M: Number of neurons in this layer.
   *
   * Outputs:
   *  - W: Weights, of shape (D, M).
   *  - b: Biases, of shape (1, M).
   */
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](weight, bias), numFeatures, numNeurons)
  /*
   * Computes the forward pass for an affine (fully-connected) layer
   * with M neurons.  The input data has N examples, each with D
   * features.
   *
   * Inputs:
   *  - X: Inputs, of shape (N, D).
   *  - W: Weights, of shape (D, M).
   *  - b: Biases, of shape (1, M).
   *
   * Outputs:
   *  - out: Outputs, of shape (N, M).
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
      invokeForward(dmlScript, List[String](out), X, weight, bias)
  /*
   * Computes the backward pass for a fully-connected (affine) layer
   * with M neurons.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of shape (N, M).
   *  - X: Inputs, of shape (N, D).
   *  - W: Weights, of shape (D, M).
   *  - b: Biases, of shape (1, M).
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of shape (N, D).
   *  - dW: Gradient wrt `W`, of shape (D, M).
   *  - db: Gradient wrt `b`, of shape (1, M).
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
      invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id, dWeight, dBias), dout, X, weight, bias)
  // -------------------------------------------------
  // num_output (c_o): the number of filters
  def numNeurons = param.getInnerProductParam.getNumOutput.toString
  def numFeatures = int_mult(bottomLayerOutputShape._1, bottomLayerOutputShape._2, bottomLayerOutputShape._3)
  // n * c_o * 1 * 1
  override def outputShape = ( param.getInnerProductParam.getNumOutput.toString, "1", "1" )
  override def weightShape():Array[Int] = Array(numFeatures.toInt, numNeurons.toInt)
  override def biasShape():Array[Int] = Array(1, numNeurons.toInt)
}


class MaxPooling(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "max_pool2d_builtin"
  override def init(dmlScript:StringBuilder) = {}
  /*
   * Computes the forward pass for a 2D spatial max pooling layer.
   * The input data has N examples, each represented as a 3D volume
   * unrolled into a single vector.
   *
   * This implementation uses a built-in operator for higher
   * performance.
   *
   * Inputs:
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - C: Number of input channels (dimensionality of input depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *      A typical value is 0.
   *  - padw: Padding for left and right sides.
   *      A typical value is 0.
   *
   * Outputs:
   *  - out: Outputs, of shape (N, C*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out, "ignoreHout_"+id, "ignoreWout_"+id), 
        X, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  /*
   * Computes the backward pass for a 2D spatial max pooling layer.
   * The input data has N examples, each represented as a 3D volume
   * unrolled into a single vector.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of
   *      shape (N, C*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - C: Number of input channels (dimensionality of input depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *      A typical value is 0.
   *  - padw: Padding for left and right sides.
   *      A typical value is 0.
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of shape (N, C*Hin*Win).
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
    invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id), dout, Hout, Wout, X, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  // n * c * h_o * w_o, where h_o and w_o are computed in the same way as convolution.
  override def outputShape = ( numChannels, Hout, Wout )
  // -------------------------------------------------
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
  def Hout = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h)
  def Wout =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
  def poolingParam = param.getPoolingParam
  def numChannels = bottomLayerOutputShape._1
  // kernel_size (or kernel_h and kernel_w): specifies height and width of each filter
  def kernel_h = if(poolingParam.hasKernelH) poolingParam.getKernelH.toString 
                   else poolingParam.getKernelSize.toString 
  def kernel_w = if(poolingParam.hasKernelW) poolingParam.getKernelW.toString 
                   else poolingParam.getKernelSize.toString
  // stride (or stride_h and stride_w) [default 1]: specifies the intervals at which to apply the filters to the input
  def stride_h = if(poolingParam.hasStrideH) poolingParam.getStrideH.toString 
                   else if(poolingParam.hasStride) poolingParam.getStride.toString
                   else "1"
  def stride_w = if(poolingParam.hasStrideW) poolingParam.getStrideW.toString 
                   else if(poolingParam.hasStride) poolingParam.getStride.toString
                   else "1"
  // pad (or pad_h and pad_w) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input
  def pad_h =   if(poolingParam.hasPadH) poolingParam.getPadH.toString 
                   else if(poolingParam.hasPad) poolingParam.getPad.toString
                   else "0"
  def pad_w =   if(poolingParam.hasPadW) poolingParam.getPadW.toString 
                   else if(poolingParam.hasPad) poolingParam.getPad.toString
                   else "0"
  override def weightShape():Array[Int] = null
  override def biasShape():Array[Int] = null
}

class Convolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // -------------------------------------------------
  override def sourceFileName = "conv2d_builtin";
  /*
   * Initialize the parameters of this layer.
   *
   * Note: This is just a convenience function, and parameters
   * may be initialized manually if needed.
   *
   * We use the heuristic by He et al., which limits the magnification
   * of inputs/gradients during forward/backward passes by scaling
   * unit-Gaussian weights by a factor of sqrt(2/n), under the
   * assumption of relu neurons.
   *  - http://arxiv.org/abs/1502.01852
   *
   * Inputs:
   *  - F: Number of filters.
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *
   * Outputs:
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   */
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](weight, bias), numKernels, numChannels, kernel_h, kernel_w)
  /*
   * Computes the forward pass for a 2D spatial convolutional layer with
   * F filters.  The input data has N examples, each represented as a 3D
   * volume unrolled into a single vector.
   *
   * This implementation uses a built-in operator for higher
   * performance.
   *
   * Inputs:
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *      For same output height as input, set `padh = (Hf - 1) / 2`,
   *      assuming `strideh = 1`.
   *      More generally, `padh = (Hin*(strideh-1) + Hf - strideh) / 2`
   *      preserves the spatial dimensions of the input.
   *  - padw: Padding for left and right sides.
   *      For same output width as input, set `padw = (Wf - 1) / 2`,
   *      assuming `stridew = 1`.
   *      More generally, `padw = (Win*(stridew-1) + Wf - stridew) / 2`
   *      preserves the spatial dimensions of the input.
   *
   * Outputs:
   *  - out: Outputs, of shape (N, F*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   */
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out, "ignoreHout_"+id, "ignoreWout_"+id), 
        X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  /*
   * Computes the backward pass for a 2D spatial convolutional layer
   * with F filters.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of
   *      shape (N, F*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *      For same output height as input, set `padh = (Hf - 1) / 2`,
   *      assuming `strideh = 1`.
   *      More generally, `padh = (Hin*(strideh-1) + Hf - strideh) / 2`
   *      preserves the spatial dimensions of the input.
   *  - padw: Padding for left and right sides.
   *      For same output width as input, set `padw = (Wf - 1) / 2`,
   *      assuming `stridew = 1`.
   *      More generally, `padw = (Win*(stridew-1) + Wf - stridew) / 2`
   *      preserves the spatial dimensions of the input.
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of shape (N, C*Hin*Win).
   *  - dW: Gradient wrt `W`, of shape (F, C*Hf*Wf).
   *  - db: Gradient wrt `b`, of shape (F, 1).
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
    invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id, dWeight, dBias), dout, Hout, Wout, X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  // n * c_o * h_o * w_o, where h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1 and w_o likewise.
  override def outputShape = ( numKernels, Hout, Wout )
  // -------------------------------------------------
  def numChannels = bottomLayerOutputShape._1
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
  def Hout = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h) 
  def Wout =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
  // -------------------------------------------------
  def convParam = param.getConvolutionParam
  override def weightShape():Array[Int] = Array(numKernels.toInt, int_mult(numChannels, kernel_h, kernel_w).toInt)
  override def biasShape():Array[Int] = Array(numKernels.toInt, 1)
  // num_output (c_o): the number of filters
  def numKernels = convParam.getNumOutput.toString
  // kernel_size (or kernel_h and kernel_w): specifies height and width of each filter
  def kernel_h = if(convParam.hasKernelH) convParam.getKernelH.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def kernel_w = if(convParam.hasKernelW) convParam.getKernelW.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  // stride (or stride_h and stride_w) [default 1]: specifies the intervals at which to apply the filters to the input
  def stride_h = if(convParam.hasStrideH) convParam.getStrideH.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else "1"
  def stride_w = if(convParam.hasStrideW) convParam.getStrideW.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else "1"
  // pad (or pad_h and pad_w) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input
  def pad_h =   if(convParam.hasPadH) convParam.getPadH.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else "0"
  def pad_w =   if(convParam.hasPadW) convParam.getPadW.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else "0"
}

class DeConvolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  override def sourceFileName: String = "conv2d_transpose"
  /*
   * Utility function to initialize the parameters of this layer.
   *
   * We use the heuristic by He et al., which limits the magnification
   * of inputs/gradients during forward/backward passes by scaling
   * unit-Gaussian weights by a factor of sqrt(2/n), under the
   * assumption of relu neurons.
   *  - http://arxiv.org/abs/1502.01852
   *
   * Inputs:
   *  - F: Number of filters.
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *
   * Outputs:
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   */
  override def init(dmlScript: StringBuilder): Unit = 
    invokeInit(dmlScript, List[String](weight, bias), numKernels, numChannels, kernel_h, kernel_w)
    
  override def weightShape():Array[Int] = Array(numKernels.toInt, int_mult(numChannels, kernel_h, kernel_w).toInt)
  override def biasShape():Array[Int] = Array(numKernels.toInt, 1)
    
  /*
   * Computes the forward pass for a 2D spatial transpose convolutional
   * layer with F filters.  The input data has N examples, each
   * represented as a 3D tensor flattened into a single vector.
   *
   * Inputs:
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *  - padw: Padding for left and right sides.
   *  - out_padh: extra padding for top side. This should 
   *      lie in [0, strideh-1].
   *  - out_padw: extra padding for right side. This should
   *      lie in [0, stridew-1].
   *
   * Outputs:
   *  - out: Outputs, of shape (N, F*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   */
  override def forward(dmlScript: StringBuilder,isPrediction: Boolean): Unit =
    invokeForward(dmlScript, List[String](out, "ignoreHout_"+id, "ignoreWout_"+id), 
        X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, "0", "0")
        
  /*
   * Computes the backward pass for a 2D spatial transpose
   * convolutional layer with F filters.
   *
   * Inputs:
   *  - dout: Gradient wrt `out` from upstream, of
   *      shape (N, F*Hout*Wout).
   *  - Hout: Output height.
   *  - Wout: Output width.
   *  - X: Inputs, of shape (N, C*Hin*Win).
   *  - W: Weights, of shape (F, C*Hf*Wf).
   *  - b: Biases, of shape (F, 1).
   *  - C: Number of input channels (dimensionality of depth).
   *  - Hin: Input height.
   *  - Win: Input width.
   *  - Hf: Filter height.
   *  - Wf: Filter width.
   *  - strideh: Stride over height.
   *  - stridew: Stride over width.
   *  - padh: Padding for top and bottom sides.
   *  - padw: Padding for left and right sides.
   *
   * Outputs:
   *  - dX: Gradient wrt `X`, of shape (N, C*Hin*Win).
   *  - dW: Gradient wrt `W`, of shape (F, C*Hf*Wf).
   *  - db: Gradient wrt `b`, of shape (F, 1).
   */
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
    invokeBackward(dmlScript, outSuffix, List[String]("dOut" + id, dWeight, dBias), 
        dout, Hout, Wout, X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  // n * c_o * h_o * w_o, where h_o = (h_i + 2 * pad_h - kernel_h) / stride_h + 1 and w_o likewise.
  override def outputShape = ( numChannels, Hout, Wout )
  // -------------------------------------------------
  def numChannels = bottomLayerOutputShape._1
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
  // Hout = strideh * (Hin-1) - 2*padh + Hf + out_padh
  def Hout:String =  try { 
    (stride_h.toInt * (Hin.toInt-1) - 2*pad_h.toInt + kernel_h.toInt).toString()
  }
  catch { 
    case _:Throwable => stride_h + " * " +  "(" + Hin + "-1) - 2*" + pad_h + " + " + kernel_h
  }
  // Wout = stridew * (Win-1) - 2*padw + Wf + out_padw
  def Wout:String =  try { 
    (stride_w.toInt * (Win.toInt-1) - 2*pad_w.toInt + kernel_w.toInt).toString()
  }
  catch { 
    case _:Throwable => stride_w + " * " +  "(" + Win + "-1) - 2*" + pad_w + " + " + kernel_w
  }
  // -------------------------------------------------
  def convParam = param.getConvolutionParam
  // num_output (c_o): the number of filters
  def numKernels = convParam.getNumOutput.toString
  // kernel_size (or kernel_h and kernel_w): specifies height and width of each filter
  def kernel_h = if(convParam.hasKernelH) convParam.getKernelH.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def kernel_w = if(convParam.hasKernelW) convParam.getKernelW.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  // stride (or stride_h and stride_w) [default 1]: specifies the intervals at which to apply the filters to the input
  def stride_h = if(convParam.hasStrideH) convParam.getStrideH.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else "1"
  def stride_w = if(convParam.hasStrideW) convParam.getStrideW.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else "1"
  // pad (or pad_h and pad_w) [default 0]: specifies the number of pixels to (implicitly) add to each side of the input
  def pad_h =   if(convParam.hasPadH) convParam.getPadH.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else "0"
  def pad_w =   if(convParam.hasPadW) convParam.getPadW.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else "0"
}
