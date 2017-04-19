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
      else if(ret.size == 1)     computedDout = ret(0).dX
      else                       computedDout = sum(new StringBuilder, ret.map(_.dX).toList).toString()
    }
    computedDout
  }
  val dX = "dOut" + id
  // --------------------------------------------------------------------------------------
  // No need to override these methods in subclasses, instead classes that have weights and biases 
  // should implement HasWeight and HasBias traits.
  def dWeight():String = throw new DMLRuntimeException("dWeight is not implemented in super class")
  def dBias():String = throw new DMLRuntimeException("dBias is not implemented in super class")
  def weight():String = null;
  def bias():String = null;
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
  def invokeBackward(dmlScript:StringBuilder, outSuffix:String, resultVariables:List[String],  arguments:String*):Unit = {
    invoke(dmlScript, sourceFileName + "::", resultVariables.map(_ + outSuffix), "backward", arguments.toList)
  }
  // --------------------------------------------------------------------------------------
}


trait IsLossLayer extends CaffeLayer {
  def computeLoss(dmlScript:StringBuilder, numTabs:Int):Unit
}

trait HasWeight extends CaffeLayer {
  override def weight = "W" + id
  override def dWeight = "dW" + id
}

trait HasBias extends CaffeLayer {
  override def bias = "b" + id
  override def dBias = "db" + id
}

class Data(val param:LayerParameter, val id:Int, val net:CaffeNetwork, val numChannels:String, val height:String, val width:String) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = {
    if(param.hasTransformParam && param.getTransformParam.hasScale) {
      dmlScript.append("X_full = X_full * " + param.getTransformParam.getScale + "\n")
    }
    dmlScript.append("BATCH_SIZE = " + param.getDataParam.getBatchSize + "\n")
  }
  var dataOutputShape = ("$num_channels", "$height", "$width")
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = { }
  override def out = "Xb"
  override def backward(dmlScript:StringBuilder, outSuffix:String) = { }
  override def outputShape = (numChannels, height, width)
  // -------------------------------------------------
}


// ------------------------------------------------------------------
// weight is ema_mean and bias is ema_var
// Fuse 
class BatchNorm(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // val scale =  
  override def sourceFileName = "batch_norm2d"
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](gamma, beta, ema_mean, ema_var), numChannels)
  var update_mean_var = true
  def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = {
    val mode = if(isPrediction) "\"test\"" else "\"train\""
    invokeForward(dmlScript, List[String](out, withSuffix(ema_mean), withSuffix(ema_var), withSuffix(cache_mean), withSuffix(cache_var), withSuffix(cache_norm)), 
        X, gamma, beta, numChannels, Hin, Win, mode, ema_mean, ema_var,  ma_fraction, eps)  
  }
  
  def backward(dmlScript: StringBuilder, outSuffix:String): Unit = {
    invokeBackward(dmlScript, outSuffix, List[String](dX, dgamma, dbeta), dout, out, ema_mean, ema_var, cache_mean, cache_var, cache_norm, X, gamma, beta, numChannels, 
          Hin, Win, "\"train\"", ema_mean, ema_var,  ma_fraction, eps)
  }
  
  private def withSuffix(str:String):String = if(update_mean_var) str else str + "_ignore"
  override def weight = "ema_mean" + id
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
  def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = assign(dmlScript, out, X)
  override def backward(dmlScript: StringBuilder, outSuffix:String): Unit = assign(dmlScript, dX + outSuffix, dout)
}
// ------------------------------------------------------------------

class Elementwise(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  if(param.getEltwiseParam.hasOperation && param.getEltwiseParam.getOperation != EltwiseOp.SUM)
    throw new LanguageException("Currently only elementwise sum operation supported")
  def forward(dmlScript: StringBuilder, isPrediction: Boolean): Unit = {
    addAndAssign(dmlScript, out, param.getBottomList.map(b => net.getCaffeLayer(b).out).toList)
  }
  override def backward(dmlScript: StringBuilder, outSuffix:String): Unit = assign(dmlScript, dX + outSuffix, dout)
  override def outputShape = {
    if(_out == null) _out = net.getCaffeLayer(net.getBottomLayers(param.getName).take(1).toSeq.get(0)).outputShape
    _out
  }
  var _out:(String, String, String) = null
  
}

class SoftmaxWithLoss(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with IsLossLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out), scores)
  override def backward(dmlScript:StringBuilder, outSuffix:String) =  {
    invoke(dmlScript, "cross_entropy_loss::", List[String]("dProbs" + outSuffix), "backward", out, "yb")
    invoke(dmlScript.append("\t"), "softmax::", List[String](dX + outSuffix), "backward", "dProbs", scores)
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
  // -------------------------------------------------
}

class ReLU(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "relu"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = invokeForward(dmlScript, List[String](out), X)
  override def backward(dmlScript:StringBuilder, outSuffix:String) = invokeBackward(dmlScript, outSuffix, List[String](dX), dout, X)
  // -------------------------------------------------
}

class Dropout(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "dropout"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) =
    if(!isPrediction)
      invokeForward(dmlScript, List[String](out, mask), X, p, seed)
    else
      assign(dmlScript, out, X) // Forward-pass not required to be performed during prediction for Dropout layer
  override def backward(dmlScript:StringBuilder, outSuffix:String) = invokeBackward(dmlScript, outSuffix, List[String](dX), dout, X, p, mask)
  // -------------------------------------------------
  def mask = "mask" + id
  def p = param.getDropoutParam.getDropoutRatio.toString
  def seed = "-1"
}

class InnerProduct(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // -------------------------------------------------
  override def sourceFileName = "affine"
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](weight, bias), numFeatures, numNeurons)
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
      invokeForward(dmlScript, List[String](out), X, weight, bias)
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
      invokeBackward(dmlScript, outSuffix, List[String](dX, dWeight, dBias), dout, X, weight, bias)
  // -------------------------------------------------
  def numNeurons = param.getInnerProductParam.getNumOutput.toString
  def numFeatures = int_mult(bottomLayerOutputShape._1, bottomLayerOutputShape._2, bottomLayerOutputShape._3)
  override def outputShape = ( param.getInnerProductParam.getNumOutput.toString, "1", "1" )
}

class MaxPooling(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "max_pool2d_builtin"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out, "ignoreHout_"+id, "ignoreWout_"+id), 
        X, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
    invokeBackward(dmlScript, outSuffix, List[String](dX), dout, Hout, Wout, X, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  override def outputShape = ( numChannels, Hout, Wout )
  // -------------------------------------------------
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
  def Hout = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h)
  def Wout =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
  def poolingParam = param.getPoolingParam
  def numChannels = bottomLayerOutputShape._1
  def kernel_h = if(poolingParam.hasKernelH) poolingParam.getKernelH.toString 
                   else poolingParam.getKernelSize.toString 
  def kernel_w = if(poolingParam.hasKernelW) poolingParam.getKernelW.toString 
                   else poolingParam.getKernelSize.toString
  def stride_h = if(poolingParam.hasStrideH) poolingParam.getStrideH.toString 
                   else poolingParam.getStride.toString
  def stride_w = if(poolingParam.hasStrideW) poolingParam.getStrideW.toString 
                   else poolingParam.getStride.toString
  def pad_h =   if(poolingParam.hasPadH) poolingParam.getPadH.toString 
                   else poolingParam.getPad.toString
  def pad_w =   if(poolingParam.hasPadW) poolingParam.getPadW.toString 
                   else poolingParam.getPad.toString
}

class Convolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer with HasWeight with HasBias {
  // -------------------------------------------------
  override def sourceFileName = "conv2d_builtin";
  override def init(dmlScript:StringBuilder) = invokeInit(dmlScript, List[String](weight, bias), numKernels, numChannels, kernel_h, kernel_w)
  override def forward(dmlScript:StringBuilder, isPrediction:Boolean) = 
    invokeForward(dmlScript, List[String](out, "ignoreHout_"+id, "ignoreWout_"+id), 
        X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  override def backward(dmlScript:StringBuilder, outSuffix:String) = 
    invokeBackward(dmlScript, outSuffix, List[String](dX, dWeight, dBias), dout, Hout, Wout, X, weight, bias, numChannels, Hin, Win, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
  override def outputShape = ( numKernels, Hout, Wout )
  // -------------------------------------------------
  def numChannels = bottomLayerOutputShape._1
  def Hin = bottomLayerOutputShape._2
  def Win = bottomLayerOutputShape._3
  def Hout = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h) 
  def Wout =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
  def convParam = param.getConvolutionParam
  def numKernels = convParam.getNumOutput.toString
  def kernel_h = if(convParam.hasKernelH) convParam.getKernelH.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def kernel_w = if(convParam.hasKernelW) convParam.getKernelW.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def stride_h = if(convParam.hasStrideH) convParam.getStrideH.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else throw new LanguageException("Incorrect stride parameters:" + convParam.getStrideH + " " + convParam.getStrideList + " " + param.getName)
  def stride_w = if(convParam.hasStrideW) convParam.getStrideW.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else throw new LanguageException("Incorrect stride parameters")
  def pad_h =   if(convParam.hasPadH) convParam.getPadH.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else throw new LanguageException("Incorrect pad parameters")
  def pad_w =   if(convParam.hasPadW) convParam.getPadW.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else throw new LanguageException("Incorrect pad parameters")
}