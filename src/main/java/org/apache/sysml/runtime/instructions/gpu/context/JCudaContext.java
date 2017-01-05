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

import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.utils.Statistics;

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.runtime.JCuda;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.cudaDeviceProp;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
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

	// The minimum CUDA Compute capability needed for SystemML.
	// After compute capability 3.0, 2^31 - 1 blocks and 1024 threads per block are supported.
	// If SystemML needs to run on an older card, this logic can be revisited.
	final int MAJOR_REQUIRED = 3;
	final int MINOR_REQUIRED = 0;

	private static final Log LOG = LogFactory.getLog(JCudaContext.class.getName());
	
	public static boolean DEBUG = false;
	
	public static long totalNumBytes = 0;
	public static AtomicLong availableNumBytesWithoutUtilFactor = new AtomicLong(0);
	// Fraction of available memory to use. The available memory is computer when the JCudaContext is created
	// to handle the tradeoff on calling cudaMemGetInfo too often.
	public boolean REFRESH_AVAILABLE_MEMORY_EVERY_TIME = ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.REFRESH_AVAILABLE_MEMORY_EVERY_TIME);
	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application.
	public double GPU_MEMORY_UTILIZATION_FACTOR = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
	static {
		long start = System.nanoTime();
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCusparse.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0); // Initialize the driver
		// Obtain the number of devices
        int deviceCountArray[] = { 0 };
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        LOG.info("Total number of GPUs on the machine: " + deviceCount);
        Statistics.cudaInitTime = System.nanoTime() - start;
	}

	@Override
	public long getAvailableMemory() {
		if(REFRESH_AVAILABLE_MEMORY_EVERY_TIME) {
			long free [] = { 0 };
	        long total [] = { 0 };
	        if(cudaMemGetInfo(free, total) == cudaSuccess) {
	        	totalNumBytes = total[0];
	        	availableNumBytesWithoutUtilFactor.set(free[0]);
	        }
	        else {
	        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
	        }
		}
		return (long) (availableNumBytesWithoutUtilFactor.get()*GPU_MEMORY_UTILIZATION_FACTOR);
	}

	@Override
	public void ensureComputeCapability() throws DMLRuntimeException {
		int[] devices =  {-1};
		cudaGetDeviceCount(devices);
		if (devices[0] == -1){
			throw new DMLRuntimeException("Call to cudaGetDeviceCount returned 0 devices");
		}
		boolean isComputeCapable = true;
		for (int i=0; i<devices[0]; i++) {
			cudaDeviceProp properties = new cudaDeviceProp();
			cudaGetDeviceProperties(properties, i);
			int major = properties.major;
			int minor = properties.minor;
			if (major < MAJOR_REQUIRED) {
				isComputeCapable = false;
			} else if (major == MAJOR_REQUIRED && minor < MINOR_REQUIRED) {
				isComputeCapable = false;
			}
		}
		if (!isComputeCapable) {
			throw new DMLRuntimeException("One of the CUDA cards on the system has compute capability lower than " + MAJOR_REQUIRED + "." + MINOR_REQUIRED);
		}
	}
	
	
	public JCudaContext() throws DMLRuntimeException {
		if(isGPUContextCreated) {
			// Wait until it is deleted. This case happens during multi-threaded testing.
			// This also allows for multi-threaded execute calls
			long startTime = System.currentTimeMillis();
			do {
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {}
			} while(isGPUContextCreated && (System.currentTimeMillis() - startTime) < 60000);
			synchronized(isGPUContextCreated) {
				if(GPUContext.currContext != null) {
					throw new RuntimeException("Cannot create multiple JCudaContext. Waited for 10 min to close previous GPUContext");
				}
			}
		}
		GPUContext.currContext = this;
		
		long start = System.nanoTime();
		LibMatrixCUDA.cudnnHandle = new cudnnHandle();
		cudnnCreate(LibMatrixCUDA.cudnnHandle);
		LibMatrixCUDA.cublasHandle = new cublasHandle();
		cublasCreate(LibMatrixCUDA.cublasHandle);
		// For cublas v2, cublasSetPointerMode tells Cublas whether to expect scalar arguments on device or on host
		// This applies to arguments like "alpha" in Dgemm, and "y" in Ddot.
		// cublasSetPointerMode(LibMatrixCUDA.cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE); 
		LibMatrixCUDA.cusparseHandle = new cusparseHandle();
		cusparseCreate(LibMatrixCUDA.cusparseHandle);
		Statistics.cudaLibrariesInitTime = System.nanoTime() - start;
		
		long free [] = { 0 };
        long total [] = { 0 };
        if(cudaMemGetInfo(free, total) == cudaSuccess) {
        	totalNumBytes = total[0];
        	availableNumBytesWithoutUtilFactor.set(free[0]);
        }
        else {
        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
        }
        LOG.info("Total GPU memory: " + (totalNumBytes*(1e-6)) + " MB");
        LOG.info("Available GPU memory: " + (availableNumBytesWithoutUtilFactor.get()*(1e-6)) + " MB");
        
        LibMatrixCUDA.kernels = new JCudaKernels();
	}

	@Override
	public void destroy() throws DMLRuntimeException {
		if(currContext != null) {
			synchronized(isGPUContextCreated) {
				cudnnDestroy(LibMatrixCUDA.cudnnHandle);
				cublasDestroy(LibMatrixCUDA.cublasHandle);
				cusparseDestroy(LibMatrixCUDA.cusparseHandle);
				currContext = null;
				isGPUContextCreated = false;
			}
		}
		else if(LibMatrixCUDA.cudnnHandle != null || LibMatrixCUDA.cublasHandle != null) {
			throw new DMLRuntimeException("Error while destroying the GPUContext");
		}
	}

}