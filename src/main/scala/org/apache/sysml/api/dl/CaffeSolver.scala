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

import caffe.Caffe.SolverParameter
import org.apache.sysml.runtime.DMLRuntimeException
import caffe.Caffe

trait CaffeSolver {
  def sourceFileName:String;
  def update(dmlScript:StringBuilder, layer:CaffeLayer):Unit;
  def init(dmlScript:StringBuilder, layer:CaffeLayer):Unit;
  
  // ----------------------------------------------------------------
  // Used for Fine-tuning
  private def getLayerLr(layer:CaffeLayer, paramIndex:Int):String = {
    val param = layer.param.getParamList
    if(param == null || param.size() <= paramIndex)
      return "lr"
    else
      // TODO: Ignoring param.get(index).getDecayMult for now
      return "(lr * " + param.get(paramIndex).getLrMult + ")"
  }
  // the first param { } is for the weights and the second is for the biases.
  def getWeightLr(layer:CaffeLayer):String = getLayerLr(layer, 0)
  def getBiasLr(layer:CaffeLayer):String = getLayerLr(layer, 1)
  // ----------------------------------------------------------------
  
  def commaSep(arr:String*):String = {
	  if(arr.length == 1) arr(0) else {
	    var ret = arr(0)
	    for(i <- 1 until arr.length) {
	      ret = ret + "," + arr(i)
	    }
	    ret
	  }
	}
  
  def l2reg_update(lambda:Double, dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    // val donotRegularizeLayers:Boolean = layer.isInstanceOf[BatchNorm] || layer.isInstanceOf[Scale];
    if(lambda != 0 && layer.shouldUpdateWeight) {
      dmlScript.append("\t").append(layer.dWeight + "_reg = l2_reg::backward(" + layer.weight + ", " + lambda + ")\n")
      dmlScript.append("\t").append(layer.dWeight + " = " + layer.dWeight + " + " + layer.dWeight + "_reg\n")
    }
  }
}

class LearningRatePolicy(lr_policy:String="exp", base_lr:Double=0.01) {
  def this(solver:Caffe.SolverParameter) {
    this(solver.getLrPolicy, solver.getBaseLr)
    if(solver.hasGamma) setGamma(solver.getGamma)
    if(solver.hasStepsize) setStepsize(solver.getStepsize)
    if(solver.hasPower()) setPower(solver.getPower)
  }
  var gamma:Double = 0.95
  var step:Double = 100000
  var power:Double = 0.75
  def setGamma(gamma1:Double):Unit = { gamma = gamma1 } 
  def setStepsize(step1:Double):Unit = { step = step1 } 
  def setPower(power1:Double): Unit = { power = power1 }
  def updateLearningRate(dmlScript:StringBuilder):Unit = {
    val new_lr = lr_policy.toLowerCase match {
      case "fixed" => base_lr.toString
      case "step" => "(" + base_lr + " * " +  gamma + " ^ " + " floor(e/" + step + "))"
      case "exp" => "(" + base_lr + " * " + gamma + "^e)"
      case "inv" =>  "(" + base_lr + "* (1 + " + gamma + " * e) ^ (-" + power + "))"
      case "poly" => "(" + base_lr  + " * (1 - e/ max_epochs) ^ " + power + ")"
      case "sigmoid" => "(" + base_lr + "( 1/(1 + exp(-" + gamma + "* (e - " + step + "))))"
      case _ => throw new DMLRuntimeException("The lr policy is not supported:" + lr_policy)
    }
    dmlScript.append("lr = " + new_lr + "\n")
  }
}

/**
 * lambda: regularization parameter
 * momentum: Momentum value. Typical values are in the range of [0.5, 0.99], usually started at the lower end and annealed towards the higher end.
 */
class SGD(lambda:Double=5e-04, momentum:Double=0.9) extends CaffeSolver {
  def update(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    l2reg_update(lambda, dmlScript, layer)
    if(momentum == 0) {
      // Use sgd
      if(layer.shouldUpdateWeight) dmlScript.append("\t").append(layer.weight + " = sgd::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer)) + ")\n")
      if(layer.shouldUpdateBias) dmlScript.append("\t").append(layer.bias + " = sgd::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer)) + ")\n")
    }
    else {
      // Use sgd_momentum
      if(layer.shouldUpdateWeight) dmlScript.append("\t").append("["+ commaSep(layer.weight, layer.weight+"_v") + "] " + 
          "= sgd_momentum::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), momentum.toString, layer.weight+"_v") + ")\n")
      if(layer.shouldUpdateBias) dmlScript.append("\t").append("["+ commaSep(layer.bias, layer.bias+"_v") + "] " + 
          "= sgd_momentum::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), momentum.toString, layer.bias+"_v") + ")\n")
    }
  }
  def init(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    if(momentum != 0) {
      if(layer.shouldUpdateWeight) dmlScript.append(layer.weight+"_v = sgd_momentum::init(" + layer.weight + ")\n")
      if(layer.shouldUpdateBias) dmlScript.append(layer.bias+"_v = sgd_momentum::init(" + layer.bias + ")\n")
    }
  }
  def sourceFileName:String = if(momentum == 0) "sgd" else "sgd_momentum" 
}

/**
 * lambda: regularization parameter
 * epsilon: Smoothing term to avoid divide by zero errors. Typical values are in the range of [1e-8, 1e-4].
 * 
 * See Adaptive Subgradient Methods for Online Learning and Stochastic Optimization, Duchi et al.
 */
class AdaGrad(lambda:Double=5e-04, epsilon:Double=1e-6) extends CaffeSolver {
  def update(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    l2reg_update(lambda, dmlScript, layer)
    if(layer.shouldUpdateWeight) dmlScript.append("\t").append("["+ commaSep(layer.weight, layer.weight+"_cache") + "] " + 
        "= adagrad::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), epsilon.toString, layer.weight+"_cache") + ")\n")
    if(layer.shouldUpdateBias) dmlScript.append("\t").append("["+ commaSep(layer.bias, layer.bias+"_cache") + "] " + 
        "= adagrad::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), epsilon.toString, layer.bias+"_cache") + ")\n")
  }
  def init(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    if(layer.shouldUpdateWeight) dmlScript.append(layer.weight+"_cache = adagrad::init(" + layer.weight + ")\n")
    if(layer.shouldUpdateBias) dmlScript.append(layer.bias+"_cache = adagrad::init(" + layer.bias + ")\n")
  }
  def sourceFileName:String = "adagrad"
}

/**
 * lambda: regularization parameter
 * momentum: Momentum value. Typical values are in the range of [0.5, 0.99], usually started at the lower end and annealed towards the higher end.
 */
class Nesterov(lambda:Double=5e-04, momentum:Double=0.9) extends CaffeSolver {
  def update(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    l2reg_update(lambda, dmlScript, layer)
    val fn = if(Caffe2DML.USE_NESTEROV_UDF) "update_nesterov" else "sgd_nesterov::update"
    if(layer.shouldUpdateWeight) dmlScript.append("\t").append("["+ commaSep(layer.weight, layer.weight+"_v") + "] " + 
        "= " + fn + "(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), momentum.toString, layer.weight+"_v") + ")\n")
    if(layer.shouldUpdateBias) dmlScript.append("\t").append("["+ commaSep(layer.bias, layer.bias+"_v") + "] " + 
        "= " + fn + "(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), momentum.toString, layer.bias+"_v") + ")\n")
  }
  def init(dmlScript:StringBuilder, layer:CaffeLayer):Unit = {
    if(layer.shouldUpdateWeight) dmlScript.append(layer.weight+"_v = sgd_nesterov::init(" + layer.weight + ")\n")
    if(layer.shouldUpdateBias) dmlScript.append(layer.bias+"_v = sgd_nesterov::init(" + layer.bias + ")\n")
  }
  def sourceFileName:String = "sgd_nesterov"
}