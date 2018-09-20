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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.PersistentLRUCache;

import jcuda.Pointer;

/**
 * Shadow buffer is a temporary staging area used during eviction.
 * It is eagerly deleted and backed using the local filesystem in case of Garbage Collection
 * or if the staging memory size exceeds the user-specified size.
 * This is needed to respect SystemML's memory estimates, while still allowing
 * for caching in case of GPU plans.
 */
public class ShadowBuffer {
	private static final Log LOG = LogFactory.getLog(ShadowBuffer.class.getName());
	private static PersistentLRUCache CACHE;
	private static AtomicLong UNIQUE_ID = new AtomicLong();
	private static long EVICTION_SHADOW_BUFFER_MAX_BYTES; 
	final GPUObject gpuObj;
	boolean isBuffered = false;
	String fileName;
	
	public static boolean isEnabled() {
		if(CACHE == null && EVICTION_SHADOW_BUFFER_MAX_BYTES >= 0) {
			double shadowBufferSize = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
			if(shadowBufferSize <= 0) {
				EVICTION_SHADOW_BUFFER_MAX_BYTES = -1; // Minor optimization to avoid unnecessary invoking configuration manager.
			}
			else {
				if(shadowBufferSize > 1) 
					throw new RuntimeException("Incorrect value (" + shadowBufferSize + ") for the configuration:" + DMLConfig.EVICTION_SHADOW_BUFFERSIZE);
				EVICTION_SHADOW_BUFFER_MAX_BYTES = (long) (((double)InfrastructureAnalyzer.getLocalMaxMemory())*shadowBufferSize);
				try {
					CACHE = new PersistentLRUCache(EVICTION_SHADOW_BUFFER_MAX_BYTES);
				} catch(IOException e) {
					LOG.warn("Unable to create a temporary directory for shadow buffering on the local filesystem; disabling shadow buffering:" + e.getMessage());
					EVICTION_SHADOW_BUFFER_MAX_BYTES = -1; // Minor optimization to avoid checking for file permission.
				}
			}
		}
		return CACHE != null;
	}
	
	public ShadowBuffer(GPUObject gpuObj) {
		if(isEnabled())
			fileName = "shadow_" + UNIQUE_ID.incrementAndGet();
		this.gpuObj = gpuObj;
		
	}
	
	/**
	 * Check if the gpu object is shadow buffered
	 * 
	 * @return true if the gpu object is shadow buffered
	 */
	public boolean isBuffered() {
		return isBuffered;
	}
	
	private static long getSizeOfDataType(long numElems) {
		return numElems * ((long) LibMatrixCUDA.sizeOfDataType);
	}
	
	/**
	 * Move the data from GPU to shadow buffer 
	 * @param instName name of the instruction
	 * @throws IOException if error 
	 * @throws FileNotFoundException  if error
	 */
	public void moveFromDevice(String instName) throws FileNotFoundException, IOException {
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		int numElems = GPUObject.toIntExact(gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns());
	
		if(isDoublePrecision()) {
			double [] shadowPointer = new double[numElems];
			cudaMemcpy(Pointer.to(shadowPointer), gpuObj.jcudaDenseMatrixPtr, getSizeOfDataType(numElems), jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost);
			CACHE.put(fileName, shadowPointer);
			isBuffered = true;
		}
		else if(isSinglePrecision()) {
			float [] shadowPointer = new float[numElems];
			cudaMemcpy(Pointer.to(shadowPointer), gpuObj.jcudaDenseMatrixPtr, getSizeOfDataType(numElems), jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost);
			CACHE.put(fileName, shadowPointer);
			isBuffered = true;
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype");
		}
		
		gpuObj.getGPUContext().cudaFreeHelper(instName, gpuObj.jcudaDenseMatrixPtr, true);
		gpuObj.jcudaDenseMatrixPtr = null;
		if (ConfigurationManager.isStatistics()) {
			// Eviction time measure in malloc
			long totalTime = System.nanoTime() - start;
			GPUStatistics.cudaFromDevToShadowTime.add(totalTime);
			GPUStatistics.cudaFromDevToShadowCount.increment();
			
		}
	}
	

	private static boolean isDoublePrecision() {
		return LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.DOUBLE;
	}
	
	private static boolean isSinglePrecision() {
		return LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.FLOAT;
	}
	
	/**
	 * Move the data from shadow buffer to Matrix object
	 * @throws IOException if error 
	 * @throws FileNotFoundException if error
	 */
	public void moveToHost() throws FileNotFoundException, IOException {
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		MatrixBlock tmp = new MatrixBlock(GPUObject.toIntExact(gpuObj.mat.getNumRows()), GPUObject.toIntExact(gpuObj.mat.getNumColumns()), false);
		tmp.allocateDenseBlock();
		double [] tmpArr = tmp.getDenseBlockValues();
		if(isDoublePrecision()) {
			System.arraycopy(CACHE.getAsDoubleArray(fileName), 0, tmpArr, 0, tmpArr.length);
		}
		else if(isSinglePrecision()) {
			float [] shadowPointer = CACHE.getAsFloatArray(fileName);
			for(int i = 0; i < shadowPointer.length; i++) {
				tmpArr[i] = shadowPointer[i];
			}
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype");
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
	 * @throws IOException if error
	 * @throws FileNotFoundException if error
	 */
	public void moveToDevice() throws FileNotFoundException, IOException {
		long start = ConfigurationManager.isStatistics() ? System.nanoTime() : 0;
		int length; Pointer shadowDevicePointer;
		if(isDoublePrecision()) {
			double [] shadowPointer = CACHE.getAsDoubleArray(fileName);
			length = shadowPointer.length;
			shadowDevicePointer = Pointer.to(shadowPointer);
		}
		else if(isSinglePrecision()) {
			float [] shadowPointer = CACHE.getAsFloatArray(fileName);
			length = shadowPointer.length;
			shadowDevicePointer = Pointer.to(shadowPointer);
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype");
		}
		long numBytes = getSizeOfDataType(length);
		gpuObj.jcudaDenseMatrixPtr = gpuObj.getGPUContext().allocate(null, numBytes);
		cudaMemcpy(gpuObj.jcudaDenseMatrixPtr, shadowDevicePointer, numBytes, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice);
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
		if(isEnabled() && isEviction && eagerDelete && !gpuObj.isDensePointerNull()) {
			long numBytes = getSizeOfDataType(gpuObj.mat.getNumRows()*gpuObj.mat.getNumColumns());
			if(EVICTION_SHADOW_BUFFER_MAX_BYTES <= numBytes) {
				return false; // Don't attempt to cache very large GPU objects.
			}
			else {
				return true; // Dense GPU objects is eligible for shadow buffering when called during eviction and is being eagerly deleted.
			}
		}
		else {
			return false;
		}
	}
	
	/**
	 * Removes the content from shadow buffer
	 */
	public void clearShadowPointer() {
		if(CACHE.containsKey(fileName)) {
			CACHE.remove(fileName);
			isBuffered = false;
		}
	}
}
