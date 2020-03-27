/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 package org.tugraz.sysds.runtime.instructions.gpu.context;

import static jcuda.runtime.JCuda.cudaMemcpy;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.Sizeof;

public class ShadowBuffer {
	private static final Log LOG = LogFactory.getLog(ShadowBuffer.class.getName());
	
	GPUObject gpuObj;
	float[] shadowPointer = null;
	private static boolean _warnedAboutShadowBuffer = false;
	
	public ShadowBuffer(GPUObject gpuObj) {
		this.gpuObj = gpuObj;
	}
	
	/**
	 * Check if the gpu object is shadow buffered
	 * 
	 * @return true if the gpu object is shadow buffered
	 */
	public boolean isBuffered() {
		return shadowPointer != null;
	}
	
	/**
	 * Move the data from GPU to shadow buffer 
	 * @param instName name of the instruction
	 */
	public void moveFromDevice(String instName) {
		long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
		int numElems = GPUObject.toIntExact(gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns());
		shadowPointer = new float[numElems];
		DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES += shadowPointer.length*Sizeof.FLOAT;
		cudaMemcpy(Pointer.to(shadowPointer), gpuObj.jcudaDenseMatrixPtr, numElems*LibMatrixCUDA.sizeOfDataType, jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost);
		gpuObj.getGPUContext().cudaFreeHelper(instName, gpuObj.jcudaDenseMatrixPtr, true);
		gpuObj.jcudaDenseMatrixPtr = null;
		if (DMLScript.STATISTICS) {
			// Eviction time measure in malloc
			long totalTime = System.nanoTime() - start;
			GPUStatistics.cudaFromDevToShadowTime.add(totalTime);
			GPUStatistics.cudaFromDevToShadowCount.increment();
			
		}
	}
	
	/**
	 * Move the data from shadow buffer to Matrix object
	 */
	public void moveToHost() {
		long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
		MatrixBlock tmp = new MatrixBlock(GPUObject.toIntExact(gpuObj.mat.getNumRows()), GPUObject.toIntExact(gpuObj.mat.getNumColumns()), false);
		tmp.allocateDenseBlock();
		double [] tmpArr = tmp.getDenseBlockValues();
		for(int i = 0; i < shadowPointer.length; i++) {
			tmpArr[i] = shadowPointer[i];
		}
		gpuObj.mat.acquireModify(tmp);
		gpuObj.mat.release();
		clearShadowPointer();
		gpuObj.dirty = false;
		if (DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - start;
			GPUStatistics.cudaFromShadowToHostTime.add(totalTime);
			GPUStatistics.cudaFromShadowToHostCount.increment();
			// Part of dev -> host, not eviction
			GPUStatistics.cudaFromDevTime.add(totalTime);
			GPUStatistics.cudaFromDevCount.increment();
		}
	}
	
	/**
	 * Move the data from shadow buffer to GPU
	 */
	public void moveToDevice() {
		long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
		long numBytes = shadowPointer.length*LibMatrixCUDA.sizeOfDataType;
		gpuObj.jcudaDenseMatrixPtr = gpuObj.getGPUContext().allocate(null, numBytes);
		cudaMemcpy(gpuObj.jcudaDenseMatrixPtr, Pointer.to(shadowPointer), numBytes, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice);
		clearShadowPointer();
		if (DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - start;
			GPUStatistics.cudaFromShadowToDevTime.add(totalTime);
			GPUStatistics.cudaFromShadowToDevCount.increment();
		}
	}
	
	/**
	 * Checks if the GPU object is eligible for shadow buffering
	 * 
	 * @param isEviction true if this method is called during eviction
	 * @param eagerDelete true if the data on device has to be eagerly deleted
	 * @return true if the given GPU object is eligible to be shadow buffered
	 */
	public boolean isEligibleForBuffering(boolean isEviction, boolean eagerDelete) {
		if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.FLOAT && isEviction && eagerDelete && !gpuObj.isDensePointerNull()) {
			int numBytes = GPUObject.toIntExact(gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns())*Sizeof.FLOAT;
			boolean ret = DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES + numBytes <= DMLScript.EVICTION_SHADOW_BUFFER_MAX_BYTES;
			if(!ret && !_warnedAboutShadowBuffer) {
				LOG.warn("Shadow buffer is full, so using CP bufferpool instead. Consider increasing sysds.gpu.eviction.shadow.bufferSize.");
				_warnedAboutShadowBuffer = true;
			}
			return ret;
		}
		else {
			return false;
		}
	}
	
	/**
	 * Removes the content from shadow buffer
	 */
	public void clearShadowPointer() {
		if(shadowPointer != null) {
			DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES -= shadowPointer.length*Sizeof.FLOAT;
		}
		shadowPointer = null;
	}
}
