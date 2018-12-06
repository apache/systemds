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
package org.apache.sysml.runtime.instructions.gpu.context;

import static jcuda.runtime.JCuda.cudaMemcpy;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.Sizeof;

public class ShadowBuffer {
	private static final Log LOG = LogFactory.getLog(ShadowBuffer.class.getName());
	
	private GPUObject gpuObj;
	// shadowPointer can be double[], float[] or short[].
	private Object shadowPointer = null;
	private static boolean _warnedAboutShadowBuffer = false;
	private static long EVICTION_SHADOW_BUFFER_CURR_BYTES = 0;
	private static long EVICTION_SHADOW_BUFFER_MAX_BYTES;
	static {
		double shadowBufferSize = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
		if(shadowBufferSize < 0 || shadowBufferSize > 1) 
			throw new RuntimeException("Incorrect value (" + shadowBufferSize + ") for the configuration:" + DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
		EVICTION_SHADOW_BUFFER_MAX_BYTES = (long) (((double)InfrastructureAnalyzer.getLocalMaxMemory())*shadowBufferSize);
	}
	
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
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		int numElems = GPUObject.toIntExact(gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns());
		if(LibMatrixCUDA.sizeOfDataType == Sizeof.DOUBLE) {
			shadowPointer = new double[numElems];
		}
		else if(LibMatrixCUDA.sizeOfDataType == Sizeof.FLOAT) {
			shadowPointer = new float[numElems];
		}
		else if(LibMatrixCUDA.sizeOfDataType == Sizeof.SHORT) {
			shadowPointer = new short[numElems];
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype");
		}
		long numBytes = getNumBytesOfShadowBuffer();
		EVICTION_SHADOW_BUFFER_CURR_BYTES += numBytes;
		cudaMemcpy(getHostShadowPointer(), gpuObj.jcudaDenseMatrixPtr, numBytes, jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost);
		gpuObj.getGPUContext().cudaFreeHelper(instName, gpuObj.jcudaDenseMatrixPtr, true);
		gpuObj.jcudaDenseMatrixPtr = null;
		if (ConfigurationManager.isStatistics()) {
			// Eviction time measure in malloc
			long totalTime = System.nanoTime() - start;
			GPUStatistics.cudaFromDevToShadowTime.add(totalTime);
			GPUStatistics.cudaFromDevToShadowCount.increment();
			
		}
	}
	
	private long getNumBytesOfShadowBuffer() {
		long numElems = 0;
		switch(LibMatrixCUDA.sizeOfDataType) {
			case Sizeof.DOUBLE:
				numElems = ((double[])shadowPointer).length;
				break;
			case Sizeof.FLOAT:
				numElems = ((float[])shadowPointer).length;
				break;
			case Sizeof.SHORT:
				numElems = ((short[])shadowPointer).length;
				break;
			default:
				throw new DMLRuntimeException("Unsupported datatype of size:" + LibMatrixCUDA.sizeOfDataType);	
		}
		return numElems*LibMatrixCUDA.sizeOfDataType;
	}
	
	private Pointer getHostShadowPointer() {
		switch(LibMatrixCUDA.sizeOfDataType) {
			case Sizeof.DOUBLE:
				return Pointer.to((double[])shadowPointer);
			case Sizeof.FLOAT:
				return Pointer.to((float[])shadowPointer);
			case Sizeof.SHORT:
				return Pointer.to((short[])shadowPointer);
			default:
				throw new DMLRuntimeException("Unsupported datatype of size:" + LibMatrixCUDA.sizeOfDataType);	
		}
	}
	
	/**
	 * Move the data from shadow buffer to Matrix object
	 */
	public void moveToHost() {
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		MatrixBlock tmp = new MatrixBlock(GPUObject.toIntExact(gpuObj.mat.getNumRows()), GPUObject.toIntExact(gpuObj.mat.getNumColumns()), false);
		tmp.allocateDenseBlock();
		double [] tmpArr = tmp.getDenseBlockValues();
		if(LibMatrixCUDA.sizeOfDataType == Sizeof.DOUBLE) {
			double[] sArr = ((double[])shadowPointer);
			System.arraycopy(sArr, 0, tmpArr, 0, sArr.length);
		}
		else if(LibMatrixCUDA.sizeOfDataType == Sizeof.FLOAT) {
			float[] sArr = ((float[])shadowPointer);
			for(int i = 0; i < sArr.length; i++) {
				tmpArr[i] = sArr[i];
			}
		}
		else if(LibMatrixCUDA.sizeOfDataType == Sizeof.SHORT) {
			// short[] sArr = ((short[])shadowPointer);
			throw new DMLRuntimeException("Unsupported operation: moveToHost for half precision");
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype of size:" + LibMatrixCUDA.sizeOfDataType);
		}
		
		gpuObj.mat.acquireModify(tmp);
		gpuObj.mat.release();
		clearShadowPointer();
		gpuObj.dirty = false;
		if (ConfigurationManager.isStatistics()) {
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
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		long numBytes = getNumBytesOfShadowBuffer();
		gpuObj.jcudaDenseMatrixPtr = gpuObj.getGPUContext().allocate(null, numBytes);
		cudaMemcpy(gpuObj.jcudaDenseMatrixPtr, getHostShadowPointer(), numBytes, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice);
		clearShadowPointer();
		if (ConfigurationManager.isStatistics()) {
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
		if(isEviction && eagerDelete && !gpuObj.isDensePointerNull()) {
			long numBytes = gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns()*LibMatrixCUDA.sizeOfDataType;
			boolean ret = EVICTION_SHADOW_BUFFER_CURR_BYTES + numBytes <= EVICTION_SHADOW_BUFFER_MAX_BYTES;
			if(!ret && !_warnedAboutShadowBuffer) {
				LOG.warn("Shadow buffer is full, so using CP bufferpool instead. Consider increasing sysml.gpu.eviction.shadow.bufferSize.");
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
			EVICTION_SHADOW_BUFFER_CURR_BYTES -= getNumBytesOfShadowBuffer();
		}
		shadowPointer = null;
	}
}
