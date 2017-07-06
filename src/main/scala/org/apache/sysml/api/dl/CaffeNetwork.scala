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

import org.apache.sysml.runtime.DMLRuntimeException
import scala.collection.JavaConversions._
import caffe.Caffe.NetParameter
import caffe.Caffe.LayerParameter
import caffe.Caffe.Phase
import java.util.ArrayList
import java.util.HashSet
import scala.collection.mutable.Stack
import org.apache.sysml.parser.LanguageException;
import java.util.HashMap
import caffe.Caffe.PoolingParameter
import org.apache.commons.logging.LogFactory

trait Network {
  def getLayers(): List[String]
  def getCaffeLayer(layerName:String):CaffeLayer
  def getBottomLayers(layerName:String): Set[String]
  def getTopLayers(layerName:String): Set[String]
  def getLayerID(layerName:String): Int
}

object CaffeNetwork {
  val LOG = LogFactory.getLog(classOf[CaffeNetwork].getName)
}

class CaffeNetwork(netFilePath:String, val currentPhase:Phase, 
     var numChannels:String, var height:String, var width:String
    ) extends Network {
  private def isIncludedInCurrentPhase(l:LayerParameter): Boolean = {
    if(currentPhase == null) return true // while deployment
    else if(l.getIncludeCount == 0) true 
    else l.getIncludeList.filter(r => r.hasPhase() && r.getPhase != currentPhase).length == 0
  }
  private var id = 1
  def this(deployFilePath:String) {
    this(deployFilePath, null, null, null, null)
  }
  // --------------------------------------------------------------------------------
  private var _net:NetParameter = Utils.readCaffeNet(netFilePath)
  private var _caffeLayerParams:List[LayerParameter] = _net.getLayerList.filter(l => isIncludedInCurrentPhase(l)).toList
  // This method is used if the user doesnot provide number of channels, height and width
  private def setCHW(inputShapes:java.util.List[caffe.Caffe.BlobShape]):Unit = {
    if(inputShapes.size != 1)
        throw new DMLRuntimeException("Expected only one input shape")
    val inputShape = inputShapes.get(0)
    if(inputShape.getDimCount != 4)
      throw new DMLRuntimeException("Expected the input shape of dimension 4")
    numChannels = inputShape.getDim(1).toString
    height = inputShape.getDim(2).toString
    width = inputShape.getDim(3).toString
  }
  if(numChannels == null && height == null && width == null) {
    val inputLayer:List[LayerParameter] = _caffeLayerParams.filter(_.getType.toLowerCase.equals("input"))
    if(inputLayer.size == 1) {
      setCHW(inputLayer(0).getInputParam.getShapeList)
    }
    else if(inputLayer.size == 0) {
      throw new DMLRuntimeException("Input shape (number of channels, height, width) is unknown. Hint: If you are using deprecated input/input_shape API, we recommend you use Input layer.")
    }
    else {
      throw new DMLRuntimeException("Multiple Input layer is not supported")
    }
  }
  // --------------------------------------------------------------------------------
  
  private var _layerNames: List[String] = _caffeLayerParams.map(l => l.getName).toList
  CaffeNetwork.LOG.debug("Layers in current phase:" + _layerNames)
  
  // Condition 1: assert that each name is unique
  private val _duplicateLayerNames = _layerNames.diff(_layerNames.distinct)
  if(_duplicateLayerNames.size != 0) throw new LanguageException("Duplicate layer names is not supported:" + _duplicateLayerNames)
  
  // Condition 2: only 1 top name, except Data layer
  private val _condition2Exceptions = Set("data")
  _caffeLayerParams.filter(l => !_condition2Exceptions.contains(l.getType.toLowerCase)).map(l => if(l.getTopCount != 1) throw new LanguageException("Multiple top layers is not supported for " + l.getName))

  // Condition 3: Replace top layer names referring to a Data layer with its name
  // Example: layer{ name: mnist, top: data, top: label, ... }
  private val _topToNameMappingForDataLayer = new HashMap[String, String]()
  private def containsOnly(list:java.util.List[String], v:String): Boolean = list.toSet.diff(Set(v)).size() == 0
  private def isData(l:LayerParameter):Boolean = l.getType.equalsIgnoreCase("data")
  private def replaceTopWithNameOfDataLayer(l:LayerParameter):LayerParameter =  {
    if(containsOnly(l.getTopList,l.getName))
      return l
    else {
      val builder = l.toBuilder(); 
      for(i <- 0 until l.getTopCount) {
        if(! l.getTop(i).equals(l.getName)) { _topToNameMappingForDataLayer.put(l.getTop(i), l.getName) }
        builder.setTop(i, l.getName) 
      }
      return builder.build() 
    }
  }
  // 3a: Replace top of DataLayer with its names
  // Example: layer{ name: mnist, top: mnist, top: mnist, ... }
  _caffeLayerParams = _caffeLayerParams.map(l => if(isData(l)) replaceTopWithNameOfDataLayer(l) else l)
  private def replaceBottomOfNonDataLayers(l:LayerParameter):LayerParameter = {
    val builder = l.toBuilder();
    // Note: Top will never be Data layer
    for(i <- 0 until l.getBottomCount) {
      if(_topToNameMappingForDataLayer.containsKey(l.getBottom(i))) 
        builder.setBottom(i, _topToNameMappingForDataLayer.get(l.getBottom(i)))
    }
    return builder.build()
  }
  // 3a: If top/bottom of other layers refer DataLayer, then replace them
  // layer { name: "conv1_1", type: "Convolution", bottom: "data"
  _caffeLayerParams = if(_topToNameMappingForDataLayer.size == 0) _caffeLayerParams else _caffeLayerParams.map(l => if(isData(l)) l else replaceBottomOfNonDataLayers(l))
  
  // Condition 4: Deal with fused layer
  // Example: layer { name: conv1, top: conv1, ... } layer { name: foo, bottom: conv1, top: conv1 }
  private def isFusedLayer(l:LayerParameter): Boolean = l.getTopCount == 1 && l.getBottomCount == 1 && l.getTop(0).equalsIgnoreCase(l.getBottom(0))
  private def containsReferencesToFusedLayer(l:LayerParameter):Boolean = l.getBottomList.foldLeft(false)((prev, bLayer) => prev || _fusedTopLayer.containsKey(bLayer))
  private val _fusedTopLayer = new HashMap[String, String]()
  _caffeLayerParams = _caffeLayerParams.map(l => {
    if(isFusedLayer(l)) {
      val builder = l.toBuilder();
      if(_fusedTopLayer.containsKey(l.getBottom(0))) {
        builder.setBottom(0, _fusedTopLayer.get(l.getBottom(0)))
      }
      builder.setTop(0, l.getName)
      _fusedTopLayer.put(l.getBottom(0), l.getName)
      builder.build()
    }
    else if(containsReferencesToFusedLayer(l)) {
      val builder = l.toBuilder();
      for(i <- 0 until l.getBottomCount) {
        if(_fusedTopLayer.containsKey(l.getBottomList.get(i))) {
          builder.setBottom(i, _fusedTopLayer.get(l.getBottomList.get(i)))
        }
      }
      builder.build()
    }
    else l
  })
  
  // Used while reading caffemodel
  val replacedLayerNames = new HashMap[String, String]();
  
  // Condition 5: Deal with incorrect naming
  // Example: layer { name: foo, bottom: arbitrary, top: bar } ... Rename the layer to bar
  private def isIncorrectNamingLayer(l:LayerParameter): Boolean = l.getTopCount == 1 && !l.getTop(0).equalsIgnoreCase(l.getName)
  _caffeLayerParams = _caffeLayerParams.map(l => {
    if(isIncorrectNamingLayer(l)) {
      val builder = l.toBuilder();
      replacedLayerNames.put(l.getName, l.getTop(0))
      builder.setName(l.getTop(0))
      builder.build()
    }
    else l
  })

  // --------------------------------------------------------------------------------
  
  // Helper functions to extract bottom and top layers
  private def convertTupleListToMap(m:List[(String, String)]):Map[String, Set[String]] = m.groupBy(_._1).map(x => (x._1, x._2.map(y => y._2).toSet)).toMap
  private def flipKeyValues(t:List[(String, Set[String])]): List[(String, String)] = t.flatMap(x => x._2.map(b => b -> x._1))
  private def expandBottomList(layerName:String, bottomList:java.util.List[String]): List[(String, String)] = bottomList.filter(b => !b.equals(layerName)).map(b => layerName -> b).toList 
  
  // The bottom layers are the layers available in the getBottomList (from Caffe .proto files)
  private val _bottomLayers:Map[String, Set[String]] = convertTupleListToMap(
      _caffeLayerParams.flatMap(l => expandBottomList(l.getName, l.getBottomList)))
  CaffeNetwork.LOG.debug("Bottom layers:" + _bottomLayers)
  
  // Find the top layers by reversing the bottom list
  private val _topLayers:Map[String, Set[String]] = convertTupleListToMap(flipKeyValues(_bottomLayers.toList))
  CaffeNetwork.LOG.debug("Top layers:" + _topLayers)
  
  private val _layers: Map[String, CaffeLayer] = _caffeLayerParams.map(l => l.getName -> convertLayerParameterToCaffeLayer(l)).toMap
  CaffeNetwork.LOG.debug("Layers:" + _layers)
  private val _layerIDs: Map[String, Int] = _layers.entrySet().map(x => x.getKey -> x.getValue.id).toMap
  
  
  private def throwException(layerName:String) = throw new LanguageException("Layer with name " + layerName + " not found")                              
  def getLayers(): List[String] =  _layerNames
  def getCaffeLayer(layerName:String):CaffeLayer = {
    if(checkKey(_layers, layerName)) _layers.get(layerName).get
    else {
      if(replacedLayerNames.contains(layerName) && checkKey(_layers, replacedLayerNames.get(layerName))) {
        _layers.get(replacedLayerNames.get(layerName)).get
      }
      else throwException(layerName)
    }
  }
  def getBottomLayers(layerName:String): Set[String] =  if(checkKey(_bottomLayers, layerName)) _bottomLayers.get(layerName).get else throwException(layerName)
  def getTopLayers(layerName:String): Set[String] = if(checkKey(_topLayers, layerName)) _topLayers.get(layerName).get else throwException(layerName)
  def getLayerID(layerName:String): Int = if(checkKey(_layerIDs, layerName))  _layerIDs.get(layerName).get else throwException(layerName)
  
  // Helper functions
  private def checkKey(m:Map[String, Any], key:String): Boolean = {
    if(m == null) throw new LanguageException("Map is null (key=" + key + ")")
    else if(key == null) throw new LanguageException("key is null (map=" + m + ")")
    else m.containsKey(key)
  }
  private def convertLayerParameterToCaffeLayer(param:LayerParameter):CaffeLayer = {
    id = id + 1
    param.getType.toLowerCase() match {
      case "convolution" => new Convolution(param, id, this)
      case "pooling" => if(param.getPoolingParam.getPool == PoolingParameter.PoolMethod.MAX)  new MaxPooling(param, id, this)
                        else throw new LanguageException("Only maxpooling is supported:" + param.getPoolingParam.getPool.name)
      case "innerproduct" => new InnerProduct(param, id, this)
      case "relu" => new ReLU(param, id, this)
      case "softmaxwithloss" => new SoftmaxWithLoss(param, id, this)
      case "dropout" => new Dropout(param, id, this)
      case "data" => new Data(param, id, this, numChannels, height, width)
      case "input" => new Data(param, id, this, numChannels, height, width)
      case "batchnorm" => new BatchNorm(param, id, this)
      case "scale" => new Scale(param, id, this)
      case "eltwise" => new Elementwise(param, id, this)
      case "concat" => new Concat(param, id, this)
      case "deconvolution" => new DeConvolution(param, id, this)
      case "threshold" => new Threshold(param, id, this)
      case "softmax" => new Softmax(param, id, this)
      case _ => throw new LanguageException("Layer of type " + param.getType + " is not supported")
    }
  }
}