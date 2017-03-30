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

trait CaffeLayer {
  // -------------------------------------------------
  // Any layer that wants to reuse SystemML-NN has to override following methods that help in generating the DML for the given layer:
  def sourceFileName:String;
  def init(dmlScript:StringBuilder):Unit;
  def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean):Unit;
  def backward(dmlScript:StringBuilder):Unit;
  def outputShape:(String, String, String) = bottomLayerOutputShape
  def weight():String = null;
  def bias():String = null;
  def dW():String = null;
  def dB():String = null;
  def computeLoss(dmlScript:StringBuilder, prefix:String):Unit = {}
  def predict(dmlScript:StringBuilder):Unit = {}
  def shouldUpdateWeight():Boolean = if(weight != null) true else false
  def shouldUpdateBias():Boolean = if(bias != null) true else false
  // -------------------------------------------------
  
  def source(dmlScript:StringBuilder):Unit = {
    Caffe2DML.source(dmlScript, sourceFileName, Caffe2DML.layerDir)
  }
  def bottomLayerOutputShape:(String, String, String) = {
    val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    if(ret.size == 0) throw new LanguageException("Expected atleast 1 bottom layer for " + param.getName)
    else ret(0).outputShape
  }
  def bottomLayerOutVar:String = {
    val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    if(ret.size == 0) throw new LanguageException("Expected atleast 1 bottom layer for " + param.getName)
    else if(ret.size == 1) ret(0).outVar
    else {
      // Instead of throwing an new LanguageException("Expected only 1 bottom layer")
      // TODO: Double-check this logic, should outVar = sum of all bottom layer outVar (or average)
      val dml = new StringBuilder
      dml.append("(")
      var first = true
      for(outVar <- ret.map(_.outVar)) {
        if(first) { first = false } else dml.append(" + ")
        dml.append(outVar)
      }
      dml.append(")").toString
    }
  }
  def topLayerDout: String = {
    val ret = net.getTopLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    if(ret.size == 0) throw new LanguageException("Expected atleast 1 top layer for " + param.getName)
    else if(ret.size == 1) ret(0).dOut
    else {
      // Instead of throwing an LanguageException("Expected only 1 top layer")
      // TODO: Double-check this logic, should dout = sum of all top layer dout (or average) 
      val dml = new StringBuilder
      dml.append("(")
      var first = true
      for(doutVar <- ret.map(_.dOut)) {
        if(first) { first = false } else dml.append(" + ")
        dml.append(doutVar)
      }
      dml.append(")").toString
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
  def int_mult(v1:String, v2:String, v3:String):String = try { (v1.toDouble * v2.toDouble * v3.toDouble).toInt.toString } catch { case _:Throwable => "("+v1+"*"+v2+"*"+v3+")"}
  def addTab(dmlScript:StringBuilder):StringBuilder = { (0 until Caffe2DML.numTabs).map(dmlScript.append("\t")); dmlScript }
  def param:LayerParameter
  def id:Int
  def net:CaffeNetwork
  def namespace:String = sourceFileName + "::"
  def outVar = "out" + id
  def dOut = "dOut" + id
  
  def isNumber(x: String):Boolean = x forall Character.isDigit
  def transpose(x:String):String = "t(" + x + ")"
}

class Data(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = {
    if(param.hasTransformParam && param.getTransformParam.hasScale) {
      dmlScript.append("X_full = X_full * " + param.getTransformParam.getScale + "\n")
    }
    dmlScript.append("BATCH_SIZE = " + batchSize + "\n")
  }
  var dataOutputShape = ("$num_channels", "$height", "$width")
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = { }
  override def outVar = "Xb"
  override def backward(dmlScript:StringBuilder) = { }
  override def outputShape = dataOutputShape
  // -------------------------------------------------
  def batchSize = param.getDataParam.getBatchSize
}


// ------------------------------------------------------------------
// weight is ema_mean and bias is ema_var
class BatchNorm(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // val scale =  
  override def sourceFileName = "spatial_batch_norm"
  override def init(dmlScript:StringBuilder) = {
    // Currently only supports per-channel batch normalization
    dmlScript.append(
      "[" + commaSep(ema_mean, ema_var, gamma, beta) + "] = " + namespace + "init(" + numChannels + ")\n")
    
  }
  
  var update_mean_var = true
  def forward(dmlScript: StringBuilder, prefix: String, isPrediction: Boolean): Unit = {
    val mode = if(isPrediction) "\"test\"" else "\"train\""
    dmlScript.append(
      "[" + commaSep(outVar, if(update_mean_var) ema_mean else ema_mean + "_ignore", if(update_mean_var) ema_var else ema_var + "_ignore") + "] = " + namespace + "forward(" + 
       commaSep(bottomLayerOutVar, ema_mean, ema_var, numChannels, H, W, gamma, beta, ma_fraction, eps, mode) + ")\n")
  }

  def backward(dmlScript: StringBuilder): Unit = {
    dmlScript.append(
      "[" + commaSep(dOut, dgamma, dbeta) + "] = " + namespace + "backward(" + 
       commaSep(topLayerDout, bottomLayerOutVar, outVar, ema_mean, ema_var, numChannels, H, W, gamma, eps) + ")\n")
  }
  
  override def weight = "ema_mean" + id
  override def bias = "ema_var" + id
  var scaleLayer:Scale = null
  def gamma():String = { checkNextLayer(); scaleLayer.weight }
  def ma_fraction():String = if(param.getBatchNormParam.hasMovingAverageFraction()) param.getBatchNormParam.getMovingAverageFraction.toString else "0.999"
  def eps():String = if(param.getBatchNormParam.hasEps()) param.getBatchNormParam.getEps.toString else "1e-5"
  def beta():String = { checkNextLayer(); scaleLayer.bias }
  def dgamma():String = { checkNextLayer();  scaleLayer.dW }
  def dbeta():String = { checkNextLayer();  scaleLayer.dB }
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
  def H = bottomLayerOutputShape._2
  def W = bottomLayerOutputShape._3
}
// weight is gamma and bias is beta
class Scale(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  if(!param.getScaleParam.getBiasTerm) throw new LanguageException("Add \"scale_param { bias_term: true }\" to the layer " + param.getName)
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  def forward(dmlScript: StringBuilder, prefix: String, isPrediction: Boolean): Unit = {
    dmlScript.append(outVar + " = " + bottomLayerOutVar + "\n")
  }
  override def backward(dmlScript: StringBuilder): Unit = { dmlScript.append(dOut + " = " + topLayerDout + "\n") }
  override def weight = "gamma" + id
  override def bias = "beta" + id
  override def dW = "dgamma" + id
  override def dB = "dbeta" + id
}
// ------------------------------------------------------------------

class Elementwise(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  override def sourceFileName = null
  override def init(dmlScript: StringBuilder): Unit = {}
  if(param.getEltwiseParam.hasOperation && param.getEltwiseParam.getOperation != EltwiseOp.SUM)
    throw new LanguageException("Currently only elementwise sum operation supported")
  def forward(dmlScript: StringBuilder, prefix: String, isPrediction: Boolean): Unit = {
    dmlScript.append(outVar + " = ")
    var first = true
    for(b <- param.getBottomList) {
      if(first) { first = false } else dmlScript.append(" + ")
      dmlScript.append(net.getCaffeLayer(b).outVar)
    }
    dmlScript.append("\n")
  }
  override def backward(dmlScript: StringBuilder): Unit = { dmlScript.append(dOut + " = " + topLayerDout + "\n") }
  override def outputShape = {
    if(_out == null) {
      _out = net.getCaffeLayer(net.getBottomLayers(param.getName).take(1).toSeq.get(0)).outputShape
    }
    _out
  }
  var _out:(String, String, String) = null
  
}

class SoftmaxWithLoss(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def source(dmlScript:StringBuilder):Unit = {
    if(!Caffe2DML.alreadyImported.contains("softmax")) Caffe2DML.source(dmlScript, "softmax", Caffe2DML.layerDir)
    if(!Caffe2DML.alreadyImported.contains("cross_entropy_loss")) Caffe2DML.source(dmlScript, "cross_entropy_loss", Caffe2DML.layerDir)
    Caffe2DML.alreadyImported.add("softmax")
    Caffe2DML.alreadyImported.add("cross_entropy_loss")
  }
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + getNonXbBottomLayer + ")\n")
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "dProbs = cross_entropy_loss::backward(" + commaSep(outVar, "yb") + ")\n\t" + 
      dOut + " = " + namespace + "backward(" + commaSep("dProbs", getNonXbBottomLayer) + ")\n")
  override def computeLoss(dmlScript:StringBuilder, prefix:String) = {
    dmlScript.append(prefix).append("tmp_loss = cross_entropy_loss::forward(" + commaSep(outVar, "yb") + ")\n")
    dmlScript.append(prefix).append("loss = loss + tmp_loss\n")
    dmlScript.append(prefix).append("true_yb = rowIndexMax(yb)\n")
    dmlScript.append(prefix).append("predicted_yb = rowIndexMax(" + outVar + ")\n")
    dmlScript.append(prefix).append("accuracy = mean(predicted_yb == true_yb)*100\n")
    dmlScript.append(prefix).append("if(debug) {\n")
    dmlScript.append(prefix).append("\tnum_rows_error_measures = min(10, ncol(yb)) \n")
    dmlScript.append(prefix).append("\terror_measures = matrix(0, rows=num_rows_error_measures, cols=5) \n")
    dmlScript.append(prefix).append("\tfor(class_i in 1:num_rows_error_measures) {\n") 
    dmlScript.append(prefix).append("\t\ttp = sum( (true_yb == predicted_yb) * (true_yb == class_i) )\n")
    dmlScript.append(prefix).append("\t\ttp_plus_fp = sum( (predicted_yb == class_i) )\n")
    dmlScript.append(prefix).append("\t\ttp_plus_fn = sum( (true_yb == class_i) )\n")
    dmlScript.append(prefix).append("\t\tprecision = tp / tp_plus_fp \n")
    dmlScript.append(prefix).append("\t\trecall = tp / tp_plus_fn \n")
    dmlScript.append(prefix).append("\t\tf1Score = 2*precision*recall / (precision+recall) \n")
    dmlScript.append(prefix).append("\t\terror_measures[class_i,1] = class_i\n")
    dmlScript.append(prefix).append("\t\terror_measures[class_i,2] = precision\n")
    dmlScript.append(prefix).append("\t\terror_measures[class_i,3] = recall\n")
    dmlScript.append(prefix).append("\t\terror_measures[class_i,4] = f1Score\n")
    dmlScript.append(prefix).append("\t\terror_measures[class_i,5] = tp_plus_fn\n")
    dmlScript.append(prefix).append("\t}\n")
    dmlScript.append(prefix).append("\tprint(\"class    \\tprecision\\trecall  \\tf1-score\\tnum_true_labels\\n\" + toString(error_measures, decimal=7, sep=\"\\t\"))\n")
    dmlScript.append(prefix).append("}\n")
  }
  var remote_parfor_prediction = false
  override def predict(dmlScript:StringBuilder):Unit = {
    if(remote_parfor_prediction)
      dmlScript.append("Prob[i,] = " + outVar + "\n")
    else
      dmlScript.append("Prob = " + outVar + "\n")
  }
  def getNonXbBottomLayer():String = {
	  val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
	  if(ret.size == 1) return ret.get(0).outVar
	  else if(ret.size == 2) {
		  val ret1 = if(!ret.get(0).outVar.equals("Xb")) ret.get(0).outVar else ""; 
		  val ret2 = if(!ret.get(1).outVar.equals("Xb")) ret.get(1).outVar else "";
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
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + bottomLayerOutVar + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayerDout, bottomLayerOutVar) + ")\n")
  // -------------------------------------------------
}

class Dropout(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "dropout"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) =
    if(!isPrediction)
      dmlScript.append(prefix).append(
        "[" + commaSep(outVar, maskVar) + "] = " + namespace + "forward(" + commaSep(bottomLayerOutVar, p, seed) + ")\n")
    else
      dmlScript.append(prefix).append(outVar + " = " + bottomLayerOutVar + "\n")
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayerDout, bottomLayerOutVar, p, maskVar) + ")\n")
  // -------------------------------------------------
  def maskVar = "mask" + id
  def p = param.getDropoutParam.getDropoutRatio.toString
  def seed = "-1"
}

class InnerProduct(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "affine"
  override def init(dmlScript:StringBuilder) = {
    dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + "init(" + commaSep(
          int_mult(bottomLayerOutputShape._1, bottomLayerOutputShape._2, bottomLayerOutputShape._3), numNeurons) + ")\n")
  }
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + commaSep(bottomLayerOutVar, weight, bias) + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace + 
      "backward(" + commaSep(topLayerDout, bottomLayerOutVar, weight, bias) + ")\n")
  
      
  // -------------------------------------------------
  def numNeurons = param.getInnerProductParam.getNumOutput.toString
  override def outputShape = ( param.getInnerProductParam.getNumOutput.toString, "1", "1" )
  override def weight = "W" + id
  override def bias = "b" + id
  override def dW = "dW" + id
  override def dB = "db" + id
}

class MaxPooling(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "max_pool_builtin"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = {
    val out2:String = if(isNumber(outputShape._2)) "ignore1_"+id else outputShape._2
    val out3:String = if(isNumber(outputShape._3)) "ignore2_"+id else outputShape._3
    dmlScript.append(prefix).append(
      "[" + commaSep(outVar, out2, out3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayerOutVar, numChannels,  bottomLayerOutputShape._2, bottomLayerOutputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) + ")\n")
  }
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayerDout, outputShape._2, outputShape._3, bottomLayerOutVar, 
                  numChannels, bottomLayerOutputShape._2, bottomLayerOutputShape._3, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)+ ")\n")
  override def outputShape = ( numChannels, outputHeight, outputWidth )
  // -------------------------------------------------
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
  def outputHeight = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h)
  def outputWidth =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
}

class Convolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "conv_builtin";
  override def init(dmlScript:StringBuilder) = { 
    val C = numChannels
    dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + 
      "init(" + commaSep(numKernels, C, kernel_h, kernel_w) + ")\n")
  }
  
  override def forward(dmlScript:StringBuilder, prefix:String, isPrediction:Boolean) = {
    
    val out2:String = if(isNumber(outputShape._2)) "ignore1_"+id else outputShape._2
    val out3:String = if(isNumber(outputShape._3)) "ignore2_"+id else outputShape._3
    
    dmlScript.append(prefix).append(
      "[" + commaSep(outVar, out2, out3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayerOutVar, weight, bias, numChannels,  bottomLayerOutputShape._2, bottomLayerOutputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) + ")\n")
  }
  
  override def outputShape = ( numKernels, outputHeight, outputWidth )
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace +
      "backward(" + commaSep(topLayerDout, outputShape._2, outputShape._3, bottomLayerOutVar, weight, bias,
                  numChannels, bottomLayerOutputShape._2, bottomLayerOutputShape._3, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)+ ")\n")
  override def weight = "W" + id
  override def bias = "b" + id
  override def dW = "dW" + id
  override def dB = "db" + id
  // -------------------------------------------------
  def outputHeight = ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._2, kernel_h, stride_h, pad_h) 
  def outputWidth =  ConvolutionUtils.getConv2dOutputMap(bottomLayerOutputShape._3, kernel_w, stride_w, pad_w)
  def convParam = param.getConvolutionParam
  def numKernels = convParam.getNumOutput.toString
  def numChannels = bottomLayerOutputShape._1
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