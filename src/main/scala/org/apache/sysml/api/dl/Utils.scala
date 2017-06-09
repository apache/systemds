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
import scala.collection.JavaConversions._
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import org.apache.sysml.parser.LanguageException;
import com.google.protobuf.TextFormat;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import caffe.Caffe.SolverParameter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import org.apache.sysml.runtime.DMLRuntimeException
import java.io.StringReader
import java.io.BufferedReader
import com.google.protobuf.CodedInputStream
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.api.mlcontext.MLContext
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaSparkContext

object Utils {
  // ---------------------------------------------------------------------------------------------
  // Helper methods for DML generation
  
  // Returns number of classes if inferred from the network
  def numClasses(net:CaffeNetwork):String = {
  	try {
  		return "" + net.getCaffeLayer(net.getLayers().last).outputShape._1.toLong
  	} catch {
  		case _:Throwable => {
  			Caffe2DML.LOG.warn("Cannot infer the number of classes from network definition. User needs to pass it via set(num_classes=...) method.")
  			return "$num_classes" // Expect users to provide it
  		}
  	}
  }
  def prettyPrintDMLScript(script:String) {
	  val bufReader = new BufferedReader(new StringReader(script))
	  var line = bufReader.readLine();
	  var lineNum = 1
	  while( line != null ) {
		  System.out.println( "%03d".format(lineNum) + "|" + line)
		  lineNum = lineNum + 1
		  line = bufReader.readLine()
	  }
  }
  
  // ---------------------------------------------------------------------------------------------
  def parseSolver(solverFilePath:String): CaffeSolver = parseSolver(readCaffeSolver(solverFilePath))
	def parseSolver(solver:SolverParameter): CaffeSolver = {
	  val momentum = if(solver.hasMomentum) solver.getMomentum else 0.0
	  val lambda = if(solver.hasWeightDecay) solver.getWeightDecay else 0.0
	  val delta = if(solver.hasDelta) solver.getDelta else 0.0
	  
	  solver.getType.toLowerCase match {
	    case "sgd" => new SGD(lambda, momentum)
	    case "adagrad" => new AdaGrad(lambda, delta)
	    case "nesterov" => new Nesterov(lambda, momentum)
	    case _ => throw new DMLRuntimeException("The solver type is not supported: " + solver.getType + ". Try: SGD, AdaGrad or Nesterov.")
	  }
    
  }
  
	// --------------------------------------------------------------
	// Caffe utility functions
	def readCaffeNet(netFilePath:String):NetParameter = {
	  // Load network
		val reader:InputStreamReader = getInputStreamReader(netFilePath); 
  	val builder:NetParameter.Builder =  NetParameter.newBuilder();
  	TextFormat.merge(reader, builder);
  	return builder.build();
	}
	
	class CopyFloatToDoubleArray(data:java.util.List[java.lang.Float], rows:Int, cols:Int, transpose:Boolean, arr:Array[Double]) extends Thread {
	  override def run(): Unit = {
	    if(transpose) {
        var iter = 0
        for(i <- 0 until cols) {
          for(j <- 0 until rows) {
            arr(j*cols + i) = data.get(iter).doubleValue()
            iter += 1
          }
        }
      }
      else {
        for(i <- 0 until data.size()) {
          arr(i) = data.get(i).doubleValue()
        }
      }
	  }
	}
	
	def allocateMatrixBlock(data:java.util.List[java.lang.Float], rows:Int, cols:Int, transpose:Boolean):(MatrixBlock,CopyFloatToDoubleArray) = {
	  val mb =  new MatrixBlock(rows, cols, false)
    mb.allocateDenseBlock()
    val arr = mb.getDenseBlock
    val thread = new CopyFloatToDoubleArray(data, rows, cols, transpose, arr)
	  thread.start
	  return (mb, thread)
	}
	def validateShape(shape:Array[Int], data:java.util.List[java.lang.Float], layerName:String): Unit = {
	  if(shape == null) 
      throw new DMLRuntimeException("Unexpected weight for layer: " + layerName)
    else if(shape.length != 2) 
      throw new DMLRuntimeException("Expected shape to be of length 2:" + layerName)
    else if(shape(0)*shape(1) != data.size())
      throw new DMLRuntimeException("Incorrect size of blob from caffemodel for the layer " + layerName + ". Expected of size " + shape(0)*shape(1) + ", but found " + data.size())
	}
	
	def saveCaffeModelFile(sc:JavaSparkContext, deployFilePath:String, 
	    caffeModelFilePath:String, outputDirectory:String, format:String):Unit = {
	  saveCaffeModelFile(sc.sc, deployFilePath, caffeModelFilePath, outputDirectory, format)
	}
	
	def saveCaffeModelFile(sc:SparkContext, deployFilePath:String, caffeModelFilePath:String, outputDirectory:String, format:String):Unit = {
	  val inputVariables = new java.util.HashMap[String, MatrixBlock]()
	  readCaffeNet(new CaffeNetwork(deployFilePath), deployFilePath, caffeModelFilePath, inputVariables)
	  val ml = new MLContext(sc)
	  val dmlScript = new StringBuilder
	  if(inputVariables.keys.size == 0)
	    throw new DMLRuntimeException("No weights found in the file " + caffeModelFilePath)
	  for(input <- inputVariables.keys) {
	    dmlScript.append("write(" + input + ", \"" + input + ".mtx\", format=\"" + format + "\");\n")
	  }
	  if(Caffe2DML.LOG.isDebugEnabled())
	    Caffe2DML.LOG.debug("Executing the script:" + dmlScript.toString)
	  val script = org.apache.sysml.api.mlcontext.ScriptFactory.dml(dmlScript.toString()).in(inputVariables)
	  ml.execute(script)
	}
	
	def readCaffeNet(net:CaffeNetwork, netFilePath:String, weightsFilePath:String, inputVariables:java.util.HashMap[String, MatrixBlock]):NetParameter = {
	  // Load network
		val reader:InputStreamReader = getInputStreamReader(netFilePath); 
  	val builder:NetParameter.Builder =  NetParameter.newBuilder();
  	TextFormat.merge(reader, builder);
  	// Load weights
	  val inputStream = CodedInputStream.newInstance(new FileInputStream(weightsFilePath))
	  inputStream.setSizeLimit(Integer.MAX_VALUE)
	  builder.mergeFrom(inputStream)
	  val net1 = builder.build();
	  
	  val asyncThreads = new java.util.ArrayList[CopyFloatToDoubleArray]()
	  for(layer <- net1.getLayerList) {
	    if(layer.getBlobsCount == 0) {
	      // No weight or bias
	      Caffe2DML.LOG.debug("The layer:" + layer.getName + " has no blobs")
	    }
	    else if(layer.getBlobsCount == 2) {
	      // Both weight and bias
	      val caffe2DMLLayer = net.getCaffeLayer(layer.getName)
	      val transpose = caffe2DMLLayer.isInstanceOf[InnerProduct]
	      
	      // weight
	      val data = layer.getBlobs(0).getDataList
	      val shape = caffe2DMLLayer.weightShape()
	      if(shape == null)
	        throw new DMLRuntimeException("Didnot expect weights for the layer " + layer.getName)
	      validateShape(shape, data, layer.getName)
	      val ret1 = allocateMatrixBlock(data, shape(0), shape(1), transpose)
	      asyncThreads.add(ret1._2)
	      inputVariables.put(caffe2DMLLayer.weight, ret1._1)
	      
	      // bias
	      val biasData = layer.getBlobs(1).getDataList
	      val biasShape = caffe2DMLLayer.biasShape()
	      if(biasShape == null)
	        throw new DMLRuntimeException("Didnot expect bias for the layer " + layer.getName)
	      validateShape(biasShape, biasData, layer.getName)
	      val ret2 = allocateMatrixBlock(biasData, biasShape(0), biasShape(1), transpose)
	      asyncThreads.add(ret2._2)
	      inputVariables.put(caffe2DMLLayer.bias, ret2._1)
	      Caffe2DML.LOG.debug("Read weights/bias for layer:" + layer.getName)
	    }
	    else if(layer.getBlobsCount == 1) {
	      // Special case: convolution/deconvolution without bias
	      // TODO: Extend nn layers to handle this situation + Generalize this to other layers, for example: InnerProduct
	      val caffe2DMLLayer = net.getCaffeLayer(layer.getName)
	      val convParam = if((caffe2DMLLayer.isInstanceOf[Convolution] || caffe2DMLLayer.isInstanceOf[DeConvolution]) && caffe2DMLLayer.param.hasConvolutionParam())  caffe2DMLLayer.param.getConvolutionParam else null  
	      if(convParam == null)
	        throw new DMLRuntimeException("Layer with blob count " + layer.getBlobsCount + " is not supported for the layer " + layer.getName)
	     
	      val data = layer.getBlobs(0).getDataList
	      val shape = caffe2DMLLayer.weightShape()
	      validateShape(shape, data, layer.getName)
	      val ret1 = allocateMatrixBlock(data, shape(0), shape(1), false)
	      asyncThreads.add(ret1._2)
	      inputVariables.put(caffe2DMLLayer.weight, ret1._1)
	      inputVariables.put(caffe2DMLLayer.bias, new MatrixBlock(convParam.getNumOutput, 1, false))
	      Caffe2DML.LOG.debug("Read only weight for layer:" + layer.getName)
	    }
	    else {
	      throw new DMLRuntimeException("Layer with blob count " + layer.getBlobsCount + " is not supported for the layer " + layer.getName)
	    }
	  }
	  
	  // Wait for the copy to be finished
	  for(t <- asyncThreads) {
	    t.join()
	  }
	  
	  // Return the NetParameter without
	  return readCaffeNet(netFilePath)
	}
	
	def readCaffeSolver(solverFilePath:String):SolverParameter = {
		val reader = getInputStreamReader(solverFilePath);
		val builder =  SolverParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}
	
	// --------------------------------------------------------------
	// File IO utility functions
	def getInputStreamReader(filePath:String ):InputStreamReader = {
		//read solver script from file
		if(filePath == null)
			throw new LanguageException("file path was not specified!");
		if(filePath.startsWith("hdfs:")  || filePath.startsWith("gpfs:")) { 
			val fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			return new InputStreamReader(fs.open(new Path(filePath)));
		}
		else { 
			return new InputStreamReader(new FileInputStream(new File(filePath)), "ASCII");
		}
	}
	// --------------------------------------------------------------
}

class Utils {
  def saveCaffeModelFile(sc:JavaSparkContext, deployFilePath:String, 
	    caffeModelFilePath:String, outputDirectory:String, format:String):Unit = {
    Utils.saveCaffeModelFile(sc, deployFilePath, caffeModelFilePath, outputDirectory, format)
  }
  
}