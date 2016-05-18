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
package org.apache.sysml.runtime.controlprogram.context;

import java.util.Collections;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;

import java.util.Comparator;

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.runtime.JCuda;
import jcuda.jcudnn.cudnnHandle;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.cudaError.cudaSuccess;

/**
 * Setup:
 * 1. Install CUDA 7.5
 * 2. Install CuDNN v4 from http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-win-x64-v4.0-prod.zip
 * 3. Download JCuda binaries version 0.7.5b and JCudnn version 0.7.5. Copy the DLLs into C:\lib (or /lib) directory. Link: http://www.jcuda.org/downloads/downloads.html
 *
 */
public class JCudaContext extends GPUContext {
	
	private static final Log LOG = LogFactory.getLog(JCudaContext.class.getName());
	
	public static boolean DEBUG = true;
	
	public static long totalNumBytes = 0;
	public static long availableNumBytesWithoutUtilFactor = 0;
	// Fraction of available memory to use. The available memory is computer when the JCudaContext is created
	// to handle the tradeoff on calling cudaMemGetInfo too often. 
	public static double GPU_MEMORY_UTILIZATION_FACTOR = 0.9; 
	public static boolean REFRESH_AVAILABLE_MEMORY_EVERY_TIME = true;
	
	static {
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0); // Initialize the driver
		// Obtain the number of devices
        int deviceCountArray[] = { 0 };
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        LOG.info("Total number of GPUs on the machine: " + deviceCount);
	}
	
	public long getAvailableMemory() {
		if(REFRESH_AVAILABLE_MEMORY_EVERY_TIME) {
			long free [] = { 0 };
	        long total [] = { 0 };
	        if(cudaMemGetInfo(free, total) == cudaSuccess) {
	        	totalNumBytes = total[0];
	        	availableNumBytesWithoutUtilFactor = free[0];
	        }
	        else {
	        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
	        }
		}
		return (long) (availableNumBytesWithoutUtilFactor*GPU_MEMORY_UTILIZATION_FACTOR);
	}
	
	
	public JCudaContext() {
		if(GPUContext.currContext != null) {
			throw new RuntimeException("Cannot create multiple JCudaContext");
		}
		GPUContext.currContext = this;
		LibMatrixCUDA.cudnnHandle = new cudnnHandle();
		cudnnCreate(LibMatrixCUDA.cudnnHandle);
		LibMatrixCUDA.cublasHandle = new cublasHandle();
		cublasCreate(LibMatrixCUDA.cublasHandle);
		
		long free [] = { 0 };
        long total [] = { 0 };
        if(cudaMemGetInfo(free, total) == cudaSuccess) {
        	totalNumBytes = total[0];
        	availableNumBytesWithoutUtilFactor = free[0];
        }
        else {
        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
        }
        LOG.info("Total GPU memory: " + (totalNumBytes*(1e-6)) + " MB");
        LOG.info("Available GPU memory: " + (availableNumBytesWithoutUtilFactor*(1e-6)) + " MB");
	}

	@Override
	public void destroy() throws DMLRuntimeException {
		if(currContext != null) {
			currContext = null;
			cudnnDestroy(LibMatrixCUDA.cudnnHandle);
			cublasDestroy(LibMatrixCUDA.cublasHandle);
		}
		else if(LibMatrixCUDA.cudnnHandle != null || LibMatrixCUDA.cublasHandle != null) {
			throw new DMLRuntimeException("Error while destroying the GPUContext");
		}
	}
	
	
	@Override
	void acquireRead(MatrixObject mat) throws DMLRuntimeException {
		prepare(mat, true);
	}
	
	@Override
	void acquireModify(MatrixObject mat) throws DMLRuntimeException {
		prepare(mat, false);
		mat._gpuHandle.isDeviceCopyModified = true;
	}
	
	private void prepare(MatrixObject mat, boolean isInput) throws DMLRuntimeException {
		if(mat._gpuHandle == null) {
			mat._gpuHandle = GPUObject.createGPUObject(mat, this);
			long GPUSize = mat._gpuHandle.getSizeOnDevice();
			
			// Ensure enough memory while allocating the matrix
			if(GPUSize > getAvailableMemory()) {
				if(DEBUG)
					LOG.info("There is not enough memory on device. Eviction is issued!");
				evict(GPUSize);
			}
			
			mat._gpuHandle.allocateMemoryOnDevice();
			synchronized(evictionLock) {
				allocatedPointers.add(mat._gpuHandle);
			}
			if(isInput)
				mat._gpuHandle.copyFromHostToDevice();
		}
		mat._gpuHandle.isLocked = true;
	}

	
	Boolean evictionLock = new Boolean(true);

	@Override
	public void release(MatrixObject mat, boolean isGPUCopyModified) {
		mat._gpuHandle.isLocked = false;
		mat._gpuHandle.isDeviceCopyModified = isGPUCopyModified;
	}
	
	
	/**
	 * It finds matrix toBeRemoved such that toBeRemoved.GPUSize >= size
	 * // TODO: it is the smallest matrix size that satisfy the above condition. For now just evicting the largest pointer.
	 * Then returns toBeRemoved. 
	 * 
	 */
	protected void evict(long GPUSize) throws DMLRuntimeException {
		if(allocatedPointers.size() == 0) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
		}
		
		synchronized(evictionLock) {
			Collections.sort(allocatedPointers, new Comparator<GPUObject>() {
	
				@Override
				public int compare(GPUObject p1, GPUObject p2) {
					if(p1.isLocked && p2.isLocked) {
						return 0;
					}
					else if(p1.isLocked && !p2.isLocked) {
						// p2 by default is considered larger
						return 1;
					}
					else if(!p1.isLocked && p2.isLocked) {
						return -1;
					}
					long p1Size = 0; long p2Size = 0;
					try {
						p1Size = p1.getSizeOnDevice();
						p2Size = p2.getSizeOnDevice();
					} catch (DMLRuntimeException e) {
						throw new RuntimeException(e);
					}
					if(p1Size == p2Size) {
						return 0;
					}
					else if(p1Size < p2Size) {
						return 1;
					}
					else {
						return -1;
					}
				}
			});
			
			
			while(GPUSize > getAvailableMemory() && allocatedPointers.size() > 0) {
				GPUObject toBeRemoved = allocatedPointers.get(allocatedPointers.size() - 1);
				if(toBeRemoved.isLocked) {
					throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
				}
				if(toBeRemoved.isDeviceCopyModified) {
					toBeRemoved.copyFromDeviceToHost();
				}
				remove(toBeRemoved.mat);
			}
		}
	}


	@Override
	public void remove(MatrixObject mat) throws DMLRuntimeException {
		if(mat != null && mat._gpuHandle != null) {
			if(mat._gpuHandle.numReferences <= 1) {
				synchronized(evictionLock) {
					allocatedPointers.remove(mat._gpuHandle);
				}
				mat._gpuHandle.deallocateMemoryOnDevice();
				mat._gpuHandle = null;
			}
			else {
				mat._gpuHandle.numReferences--;
			}
			
		}
	}


	
	
}