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
  def sourceFileName: String;
  def update(dmlScript: StringBuilder, layer: CaffeLayer): Unit;
  def init(dmlScript: StringBuilder, layer: CaffeLayer): Unit;

  // ----------------------------------------------------------------
  // Used for Fine-tuning
  private def getLayerLr(layer: CaffeLayer, paramIndex: Int): String = {
    val param = layer.param.getParamList
    if (param == null || param.size() <= paramIndex)
      return "lr"
    else
      // TODO: Ignoring param.get(index).getDecayMult for now
      return "(lr * " + param.get(paramIndex).getLrMult + ")"
  }
  // the first param { } is for the weights and the second is for the biases.
  def getWeightLr(layer: CaffeLayer): String = getLayerLr(layer, 0)
  def getBiasLr(layer: CaffeLayer): String   = getLayerLr(layer, 1)
  // ----------------------------------------------------------------

  def commaSep(arr: String*): String =
    if (arr.length == 1) arr(0)
    else {
      var ret = arr(0)
      for (i <- 1 until arr.length) {
        ret = ret + "," + arr(i)
      }
      ret
    }

  def regularization_update(regularizationType:String, lambda: Double, dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    // val donotRegularizeLayers:Boolean = layer.isInstanceOf[BatchNorm] || layer.isInstanceOf[Scale];
    val regularizationSource = 
      if(regularizationType.toLowerCase.equals("l2")) "l2_reg"
      else if(regularizationType.toLowerCase.equals("l1")) "l1_reg"
      else null
    if(regularizationSource == null) {
      throw new DMLRuntimeException("Unsupported regularization_type:" + regularizationType + ". Please use either L2 or L1.")
    }
    
    if (lambda != 0 && layer.shouldUpdateWeight) {
      // Use layer-specific decay multiplier, if param { lr_mult: 1 decay_mult: 1 } is specified in the network file
      val hasDecayMult = layer.param.getParamList != null && layer.param.getParamList.size >= 1 && layer.param.getParamList.get(0).hasDecayMult
      val newLambda = if(hasDecayMult) layer.param.getParamList.get(0).getDecayMult * lambda else lambda
      
      dmlScript.append("\t").append(layer.dWeight + "_reg = " + regularizationSource + "::backward(" + layer.weight + ", " + newLambda + ")\n")
      dmlScript.append("\t").append(layer.dWeight + " = " + layer.dWeight + " + " + layer.dWeight + "_reg\n")
      if(layer.shouldUpdateExtraWeight) {
        dmlScript.append("\t").append(layer.dExtraWeight + "_reg = " + regularizationSource + "::backward(" + layer.extraWeight + ", " + newLambda + ")\n")
        dmlScript.append("\t").append(layer.dExtraWeight + " = " + layer.dExtraWeight + " + " + layer.dExtraWeight + "_reg\n")
      }
    }
  }
}

class LearningRatePolicy(lr_policy: String = "exp", base_lr: Double = 0.01) {
  def this(solver: Caffe.SolverParameter) {
    this(solver.getLrPolicy, solver.getBaseLr)
    if (solver.hasGamma) setGamma(solver.getGamma)
    if (solver.hasStepsize) setStepsize(solver.getStepsize)
    if (solver.hasPower()) setPower(solver.getPower)
  }
  var gamma: Double                    = 0.95
  var step: Double                     = 100000
  var power: Double                    = 0.75
  def setGamma(gamma1: Double): Unit   = gamma = gamma1
  def setStepsize(step1: Double): Unit = step = step1
  def setPower(power1: Double): Unit   = power = power1
  def updateLearningRate(dmlScript: StringBuilder): Unit = {
    val new_lr = lr_policy.toLowerCase match {
      case "fixed"   => base_lr.toString
      case "step"    => "(" + base_lr + " * " + gamma + " ^ " + " floor(e/" + step + "))"
      case "exp"     => "(" + base_lr + " * " + gamma + "^e)"
      case "inv"     => "(" + base_lr + "* (1 + " + gamma + " * e) ^ (-" + power + "))"
      case "poly"    => "(" + base_lr + " * (1 - e/ max_epochs) ^ " + power + ")"
      case "sigmoid" => "(" + base_lr + "( 1/(1 + exp(-" + gamma + "* (e - " + step + "))))"
      case _         => throw new DMLRuntimeException("The lr policy is not supported:" + lr_policy)
    }
    dmlScript.append("lr = " + new_lr + "\n")
  }
}

class SGD(regularizationType:String = "L2", lambda: Double = 5e-04, momentum: Double = 0.9) extends CaffeSolver {
  /*
   * Performs an SGD update with momentum.
   *
   * In SGD with momentum, we assume that the parameters have a velocity
   * that continues with some momentum, and that is influenced by the
   * gradient.
   *
   * Inputs:
   *  - X: Parameters to update, of shape (any, any).
   *  - dX: Gradient wrt `X` of a loss function being optimized, of
   *      same shape as `X`.
   *  - lr: Learning rate.
   *  - mu: Momentum value.
   *      Typical values are in the range of [0.5, 0.99], usually
   *      started at the lower end and annealed towards the higher end.
   *  - v: State maintaining the velocity of the parameters `X`, of same
   *      shape as `X`.
   *
   * Outputs:
   *  - X: Updated parameters `X`, of same shape as input `X`.
   *  - v: Updated velocity of the parameters `X`, of same shape as
   *      input `X`.
   */
  def update(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    regularization_update(regularizationType, lambda, dmlScript, layer)
    if (momentum == 0) {
      // Use sgd
      if (layer.shouldUpdateWeight) dmlScript.append("\t").append(layer.weight + " = sgd::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer)) + ")\n")
      if (layer.shouldUpdateExtraWeight) dmlScript.append("\t").append(layer.extraWeight + " = sgd::update(" + commaSep(layer.extraWeight, layer.dExtraWeight, getWeightLr(layer)) + ")\n")
      if (layer.shouldUpdateBias) dmlScript.append("\t").append(layer.bias + " = sgd::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer)) + ")\n")
    } else {
      // Use sgd_momentum
      if (layer.shouldUpdateWeight)
        dmlScript
          .append("\t")
          .append(
            "[" + commaSep(layer.weight, layer.weight + "_v") + "] " +
            "= sgd_momentum::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), momentum.toString, layer.weight + "_v") + ")\n"
          )
      if (layer.shouldUpdateExtraWeight)
        dmlScript
          .append("\t")
          .append(
            "[" + commaSep(layer.extraWeight, layer.extraWeight + "_v") + "] " +
            "= sgd_momentum::update(" + commaSep(layer.extraWeight, layer.dExtraWeight, getWeightLr(layer), momentum.toString, layer.extraWeight + "_v") + ")\n"
          )
      if (layer.shouldUpdateBias)
        dmlScript
          .append("\t")
          .append(
            "[" + commaSep(layer.bias, layer.bias + "_v") + "] " +
            "= sgd_momentum::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), momentum.toString, layer.bias + "_v") + ")\n"
          )
    }
  }
  def init(dmlScript: StringBuilder, layer: CaffeLayer): Unit =
    if (momentum != 0) {
      if (layer.shouldUpdateWeight) dmlScript.append(layer.weight + "_v = sgd_momentum::init(" + layer.weight + ")\n")
      if (layer.shouldUpdateExtraWeight) dmlScript.append(layer.extraWeight + "_v = sgd_momentum::init(" + layer.extraWeight + ")\n")
      if (layer.shouldUpdateBias) dmlScript.append(layer.bias + "_v = sgd_momentum::init(" + layer.bias + ")\n")
    }
  def sourceFileName: String = if (momentum == 0) "sgd" else "sgd_momentum"
}

class AdaGrad(regularizationType:String = "L2", lambda: Double = 5e-04, epsilon: Double = 1e-6) extends CaffeSolver {
  /*
   * Performs an Adagrad update.
   *
   * This is an adaptive learning rate optimizer that maintains the
   * sum of squared gradients to automatically adjust the effective
   * learning rate.
   *
   * Reference:
   *  - Adaptive Subgradient Methods for Online Learning and Stochastic
   *    Optimization, Duchi et al.
   *      - http://jmlr.org/papers/v12/duchi11a.html
   *
   * Inputs:
   *  - X: Parameters to update, of shape (any, any).
   *  - dX: Gradient wrt `X` of a loss function being optimized, of
   *      same shape as `X`.
   *  - lr: Learning rate.
   *  - epsilon: Smoothing term to avoid divide by zero errors.
   *      Typical values are in the range of [1e-8, 1e-4].
   *  - cache: State that maintains per-parameter sum of squared
   *      gradients, of same shape as `X`.
   *
   * Outputs:
   *  - X: Updated parameters `X`, of same shape as input `X`.
   *  - cache: State that maintains per-parameter sum of squared
   *      gradients, of same shape as `X`.
   */
  def update(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    regularization_update(regularizationType, lambda, dmlScript, layer)
    if (layer.shouldUpdateWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.weight, layer.weight + "_cache") + "] " +
          "= adagrad::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), epsilon.toString, layer.weight + "_cache") + ")\n"
        )
    if (layer.shouldUpdateExtraWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.extraWeight, layer.extraWeight + "_cache") + "] " +
          "= adagrad::update(" + commaSep(layer.extraWeight, layer.dExtraWeight, getWeightLr(layer), epsilon.toString, layer.extraWeight + "_cache") + ")\n"
        )
    if (layer.shouldUpdateBias)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.bias, layer.bias + "_cache") + "] " +
          "= adagrad::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), epsilon.toString, layer.bias + "_cache") + ")\n"
        )
  }
  def init(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    if (layer.shouldUpdateWeight) dmlScript.append(layer.weight + "_cache = adagrad::init(" + layer.weight + ")\n")
    if (layer.shouldUpdateExtraWeight) dmlScript.append(layer.extraWeight + "_cache = adagrad::init(" + layer.extraWeight + ")\n")
    if (layer.shouldUpdateBias) dmlScript.append(layer.bias + "_cache = adagrad::init(" + layer.bias + ")\n")
  }
  def sourceFileName: String = "adagrad"
}

class Adam(regularizationType:String = "L2", lambda: Double = 5e-04, momentum:Double = 0.9, momentum2:Double = 0.999, delta:Double = 1e-8) extends CaffeSolver {
  /*
   * Performs an Adam update.
   *
   * Reference:
   *  - Adam: A Method for Stochastic Optimization, Kingma, Ba.
   *    - http://arxiv.org/abs/1412.6980
   *
   * Inputs:
   *  - X: Parameters to update, of shape (any, any).
   *  - dX: Gradient wrt `X` of a loss function being optimized, of
   *      same shape as `X`.
   *  - lr: Learning rate.  Recommended value is 0.001.
   *  - beta1: Exponential decay rate for the 1st moment estimates.
   *      Recommended value is 0.9.
   *  - beta2: Exponential decay rate for the 2nd moment estimates.
   *      Recommended value is 0.999.
   *  - epsilon: Smoothing term to avoid divide by zero errors.
   *      Recommended value is 1e-8.
   *  - t: Timestep, starting at 0.
   *  - m: State containing the 1st moment (mean) estimate by
   *      maintaining exponential moving averages of the gradients, of
   *      same shape as `X`.
   *  - v: State containing the 2nd raw moment (uncentered variance)
   *      estimate by maintaining exponential moving averages of the
   *      squared gradients, of same shape as `X`.
   *
   * Outputs:
   *  - X: Updated parameters `X`, of same shape as input `X`.
   *  - m: Updated state containing the 1st moment (mean) estimate by
   *      maintaining exponential moving averages of the gradients, of
   *      same shape as `X`.
   *  - v: Updated state containing the 2nd raw moment (uncentered
   *      variance) estimate by maintaining exponential moving averages
   *      of the squared gradients, of same shape as `X`.
   */
  def update(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    regularization_update(regularizationType, lambda, dmlScript, layer)
    val t:String = "iter - 1" // since iter starts with 0
    // X, dX, double lr, double beta1, double beta2, epsilon, int t, matrix[double] m, matrix[double] v
    if (layer.shouldUpdateWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.weight, layer.weight + "_m", layer.weight + "_v") + "] " +
          "= adam::update(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), 
              momentum.toString, momentum2.toString, delta.toString,  t,
              layer.weight + "_m", layer.weight + "_v") + ")\n"
        )
    if (layer.shouldUpdateExtraWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.extraWeight, layer.extraWeight + "_m", layer.extraWeight + "_v") + "] " +
          "= adam::update(" + commaSep(layer.extraWeight, layer.dExtraWeight, getWeightLr(layer), 
              momentum.toString, momentum2.toString, delta.toString,  t,
              layer.extraWeight + "_m", layer.extraWeight + "_v") + ")\n"
        )
    if (layer.shouldUpdateBias)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.bias, layer.bias + "_m", layer.bias + "_v") + "] " +
          "= adam::update(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), 
              momentum.toString, momentum2.toString, delta.toString,  t, 
              layer.weight + "_m", layer.weight + "_v") + ")\n"
        )
  }
  def init(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    if (layer.shouldUpdateWeight) dmlScript.append("[ " + layer.weight + "_m, " + layer.weight + "_v ] = adam::init(" + layer.weight + ")\n")
    if (layer.shouldUpdateExtraWeight) dmlScript.append("[ " + layer.extraWeight + "_m, " + layer.extraWeight + "_v ] = adam::init(" + layer.extraWeight + ")\n")
    if (layer.shouldUpdateBias) dmlScript.append("[ " + layer.bias + "_m, " + layer.bias + "_v ] = adam::init(" + layer.bias + ")\n")
  }
  def sourceFileName: String = "adam"
}

class Nesterov(regularizationType:String = "L2", lambda: Double = 5e-04, momentum: Double = 0.9) extends CaffeSolver {
  /*
   * Performs an SGD update with Nesterov momentum.
   *
   * As with regular SGD with momentum, in SGD with Nesterov momentum,
   * we assume that the parameters have a velocity that continues
   * with some momentum, and that is influenced by the gradient.
   * In this view specifically, we perform the position update from the
   * position that the momentum is about to carry the parameters to,
   * rather than from the previous position.  Additionally, we always
   * store the parameters in their position after momentum.
   *
   * Reference:
   *  - Advances in optimizing Recurrent Networks, Bengio et al.,
   *    section 3.5.
   *    - http://arxiv.org/abs/1212.0901
   *
   * Inputs:
   *  - X: Parameters to update, of shape (any, any).
   *  - dX: Gradient wrt `X` of a loss function being optimized, of
   *      same shape as `X`.
   *  - lr: Learning rate.
   *  - mu: Momentum value.
   *      Typical values are in the range of [0.5, 0.99], usually
   *      started at the lower end and annealed towards the higher end.
   *  - v: State maintaining the velocity of the parameters `X`, of same
   *      shape as `X`.
   *
   * Outputs:
   *  - X: Updated parameters X, of same shape as input X.
   *  - v: Updated velocity of the parameters X, of same shape as
   *      input v.
   */
  def update(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    val fn            = if (Caffe2DML.USE_NESTEROV_UDF) "update_nesterov" else "sgd_nesterov::update"
    val lastParameter = if (Caffe2DML.USE_NESTEROV_UDF) (", " + lambda) else ""
    if (!Caffe2DML.USE_NESTEROV_UDF) {
      regularization_update(regularizationType, lambda, dmlScript, layer)
    }
    if (layer.shouldUpdateWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.weight, layer.weight + "_v") + "] " +
          "= " + fn + "(" + commaSep(layer.weight, layer.dWeight, getWeightLr(layer), momentum.toString, layer.weight + "_v") + lastParameter + ")\n"
        )
    if (layer.shouldUpdateExtraWeight)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.extraWeight, layer.extraWeight + "_v") + "] " +
          "= " + fn + "(" + commaSep(layer.extraWeight, layer.dExtraWeight, getWeightLr(layer), momentum.toString, layer.extraWeight + "_v") + lastParameter + ")\n"
        )
    if (layer.shouldUpdateBias)
      dmlScript
        .append("\t")
        .append(
          "[" + commaSep(layer.bias, layer.bias + "_v") + "] " +
          "= " + fn + "(" + commaSep(layer.bias, layer.dBias, getBiasLr(layer), momentum.toString, layer.bias + "_v") + lastParameter + ")\n"
        )
  }
  def init(dmlScript: StringBuilder, layer: CaffeLayer): Unit = {
    if (layer.shouldUpdateWeight) dmlScript.append(layer.weight + "_v = sgd_nesterov::init(" + layer.weight + ")\n")
    if (layer.shouldUpdateExtraWeight) dmlScript.append(layer.extraWeight + "_v = sgd_nesterov::init(" + layer.extraWeight + ")\n")
    if (layer.shouldUpdateBias) dmlScript.append(layer.bias + "_v = sgd_nesterov::init(" + layer.bias + ")\n")
  }
  def sourceFileName: String = "sgd_nesterov"
}
