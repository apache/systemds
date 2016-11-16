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

// Wrapper on top of Caffe Network to simplify usage
// 
class CaffeNetwork(netFilePath:String, val currentPhase:Phase, val numChannels:Int, val inputHeight:Int, val inputWidth:Int) {
  // Returns names of layers in sorted order
  def getLayers(): List[String] = layerNameList
  def getCaffeLayer(layerName:String):CaffeLayer = {
    val ret = layerNameMap.get(layerName)
    if(ret == null) throw new LanguageException("Layer with name " + layerName + " is not available for current phase: " + currentPhase.name() + ".")
    else ret
  }
  def getBottomLayers(layerName:String): HashSet[String] = layerNameBottomMap.get(layerName)
  def getTopLayers(layerName:String): HashSet[String] = layerNameTopMap.get(layerName)
  def getLayerID(layerName:String): Int = layerNameIDMap.get(layerName)
  
  private def getCaffeLayer(param:LayerParameter, id:Int) = {
    param.getType.toLowerCase() match {
      case "convolution" => new Convolution(param, id, this)
      case "pooling" => if(param.getPoolingParam.getPool == PoolingParameter.PoolMethod.MAX)  new MaxPooling(param, id, this)
                        else throw new LanguageException("Only maxpooling is supported:" + param.getPoolingParam.getPool.name)
      case "innerproduct" => new InnerProduct(param, id, this)
      case "relu" => new ReLU(param, id, this)
      case "softmaxwithloss" => new SoftmaxWithLoss(param, id, this)
      case "dropout" => new Dropout(param, id, this)
      case "data" => new Data(param, id, this, numChannels, inputHeight, inputWidth)
      case "batchnorm" => new BatchNorm(param, id, this)
      case "scale" => new Scale(param, id, this)
      case "eltwise" => new Elementwise(param, id, this)
      case _ => throw new LanguageException("Layer of type " + param.getType + " is not supported")
    }
  }
  // ------------------------------------------------------------------
  private def getLayersFromCurrentPhase(net:NetParameter) = {
    var ret = net.getLayerList.filter(l =>
	    if(l.getIncludeCount == 0) true else l.getIncludeList.filter(r => r.hasPhase() && r.getPhase != currentPhase).length == 0
	    // (l.getPhase == currentPhase)
	  )
	  val squashedNodes = ret.filter(l => l.getTopCount == 1 && l.getBottomCount == 1 && l.getTop(0).equals(l.getBottom(0))).map(_.getName).toSet
	  
	  // This logic handles cases where activation's top and bottom layer name points to the previous primary layer
	  // Example:
//	  layer {
//    	bottom: "conv1"
//    	top: "conv1"
//    	name: "relu1"
//    	type: "ReLU"
//    }
	  val layerNameMapping:HashMap[String, String] = new HashMap[String, String]()
	  ret = ret.map(l => {
	     if(!squashedNodes.contains(l.getName)) {
	       if(l.getBottomList.filter(x => layerNameMapping.containsKey(x)).length > 0) {
	         // Next primary layer (eg: pooling) pointing to previous primary layer (eg: conv1)
	         val builder = l.toBuilder()
	         for(i <- 0 until l.getBottomCount) {
	           if(layerNameMapping.containsKey(l.getBottom(i))) builder.setBottom(i, layerNameMapping.get(l.getBottom(i)))
	         }
	         builder.build()
	       }
	       else l
	     }
	     else {
	      // Usually an activation layer (eg: relu1) with top and bottom layer referring to primary layer (eg: conv1)
	      val oldBottom = l.getBottom(0)
	      val newBottom = if(layerNameMapping.containsKey(oldBottom)) layerNameMapping.get(oldBottom) else oldBottom
	      layerNameMapping.put(oldBottom, l.getName)
	      l.toBuilder().setTop(0, l.getName).setBottom(0, newBottom).build()
	     }
	    }
	  )
	  // uncomment for debugging
    // ret.map(l => System.out.println(l.getBottomList + " -> " + l.getName + " -> " + l.getTopList))
	  ret
  }
	private var layerNameMap:HashMap[String, CaffeLayer] = new HashMap[String, CaffeLayer]
  private var layerNameBottomMap:HashMap[String, HashSet[String]] = new HashMap[String, HashSet[String]]
  private var layerNameTopMap:HashMap[String, HashSet[String]] = new HashMap[String, HashSet[String]]
  private var layerNameIDMap:HashMap[String, Int] = new HashMap[String, Int]
  private var layerNameList:List[String] = null
  
  populateLayerNameList(Utils.readCaffeNet(netFilePath))
  
  private def getTopLayers(l:LayerParameter, currentLayers:List[LayerParameter]) = 
    (l.getTopList ++ currentLayers.filter(_.getBottomList.toSet.contains(l.getName)).map(_.getName)).filter(!_.equals(l.getName))
  
    private def getBottomLayers(l:LayerParameter, currentLayers:List[LayerParameter]) = 
    (l.getBottomList ++ currentLayers.filter(_.getTopList.toSet.contains(l.getName)).map(_.getName)).filter(!_.equals(l.getName))
  
  private def populateLayerNameList(net:NetParameter):Unit = {
    // TODO: getTopologicalSortedLayers
    val currentLayers = getLayersFromCurrentPhase(net)
    val currentLayerNames = currentLayers.map(_.getName).distinct.toSet
    
    // Append top/bottom layers
    currentLayers.map(l => {
      getBottomLayers(l, currentLayers.toList).filter(currentLayerNames.contains(_)).map(b => appendToHM(layerNameBottomMap, l.getName, b))
      getTopLayers(l, currentLayers.toList).filter(currentLayerNames.contains(_)).map(t => appendToHM(layerNameTopMap, l.getName, t))
    })
    
    var id = 1
    // Then append all layerNameMap
    currentLayers.map(l => { 
      layerNameMap.put(l.getName, getCaffeLayer(l, id))
      layerNameIDMap.put(l.getName, id)
      id = id + 1
     })
    
    layerNameList = currentLayers.map(_.getName).toList
  }
  private def appendToHM(hm:HashMap[String, HashSet[String]], key:String, value:String) = {
    if(key == null) throw new DMLRuntimeException("Cannot append null key")
    if(value == null) throw new DMLRuntimeException("Cannot append null key")
    if(!hm.containsKey(key)) hm.put(key, new HashSet[String]())
    hm.get(key).add(value)
  }
  private def shouldVisit(layerName:String, visited:HashSet[String]):Boolean = {
    val iter = getBottomLayers(layerName).iterator()
    while(iter.hasNext()) {
      val bottomLayer = iter.next() 
      if(!bottomLayer.equals(layerName) && !visited.contains(bottomLayer)) {
        return false
      }
    }
    return true
	}
	private def getTopologicalSortedLayers(netLayersList:List[CaffeLayer]): List[CaffeLayer] = {
	  val visited:HashSet[String] = new HashSet[String]()
	  val ret:ArrayList[CaffeLayer] = new ArrayList[CaffeLayer]()
	  while(visited.size < netLayersList.size) {
	    var atleastOneVisited = false
	    for(l <- netLayersList) {
	      val isAlreadyVisited = visited.contains(l.param.getName)
	      if(!isAlreadyVisited && shouldVisit(l.param.getName, visited)) {
	        visited.add(l.param.getName)
	        ret.add(l)
	        atleastOneVisited = true
	      }
	    }
	    if(!atleastOneVisited && visited.size < netLayersList.size) {
	      throw new LanguageException("Possible cycle")
	    }
	  }
	  ret.toList
	}
}