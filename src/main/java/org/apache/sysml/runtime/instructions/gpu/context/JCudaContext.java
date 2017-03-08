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

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.utils.GPUStatistics;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaError.cudaSuccess;


public class JCudaContext extends GPUContext {

	/** Synchronization object to make sure no allocations happen when something is being evicted from memory */
	public static final Object syncObj = new Object();
	private static final Log LOG = LogFactory.getLog(JCudaContext.class.getName());

	/** Global list of allocated {@link GPUObject} instances. This list must be accessed in a synchronized way */
	public static ArrayList<GPUObject> allocatedPointers = new ArrayList<GPUObject>();

	// The minimum CUDA Compute capability needed for SystemML.
	// After compute capability 3.0, 2^31 - 1 blocks and 1024 threads per block are supported.
	// If SystemML needs to run on an older card, this logic can be revisited.
	final int MAJOR_REQUIRED = 3;
	final int MINOR_REQUIRED = 0;

	/** The total number of cuda devices on this machine */
	public static int deviceCount = -1;

	/** enable this to print debug information before code pertaining to the GPU is executed  */
	public static boolean DEBUG = false;

	/** total bytes available on currently active cude device, please be careful with its bookkeeping */
	AtomicLong deviceMemBytes = new AtomicLong(0);

	/** Stores the cached deviceProperties */
	private static cudaDeviceProp[] deviceProperties;

	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application.
	public double GPU_MEMORY_UTILIZATION_FACTOR = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
	// Whether to invoke cudaMemGetInfo for available memory or rely on internal bookkeeping for memory info.
	public boolean REFRESH_AVAILABLE_MEMORY_EVERY_TIME = ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.REFRESH_AVAILABLE_MEMORY_EVERY_TIME);
	static {
		long start = System.nanoTime();
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCusparse.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0); // Initialize the driver

		int deviceCountArray[] = { 0 };
		cuDeviceGetCount(deviceCountArray);				// Obtain the number of devices
		deviceCount = deviceCountArray[0];
		deviceProperties = new cudaDeviceProp[deviceCount];

		LOG.info("Total number of GPUs on the machine: " + deviceCount);
		int maxBlocks = getMaxBlocks();
		int maxThreadsPerBlock = getMaxThreadsPerBlock();
		long sharedMemPerBlock = getMaxSharedMemory();
		int[] device = {-1};
		cudaGetDevice(device);
		LOG.info("Active CUDA device number : " + device[0]);
		LOG.info("Max Blocks/Threads/SharedMem : " + maxBlocks + "/" + maxThreadsPerBlock + "/" + sharedMemPerBlock);


		GPUStatistics.cudaInitTime = System.nanoTime() - start;
	}

	@Override
	public long getAvailableMemory() {
		if (REFRESH_AVAILABLE_MEMORY_EVERY_TIME) {
			long free[] = {0};
			long total[] = {0};
			if (cudaMemGetInfo(free, total) == cudaSuccess) {
				//long totalNumBytes = total[0];
				deviceMemBytes.set(free[0]);
			} else {
				throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
			}
		}
		return (long) (deviceMemBytes.get()*GPU_MEMORY_UTILIZATION_FACTOR);
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
			cudaDeviceProp properties = getGPUProperties(i);
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

	/**
	 * Gets the device properties for the active GPU (set with cudaSetDevice())
	 * @return the device properties
	 */
	public static cudaDeviceProp getGPUProperties() {
		int[] device = {-1};
		cudaGetDevice(device);	// Get currently active device
		return getGPUProperties(device[0]);
	}

	/**
	 * Gets the device properties
	 * @param device the device number (on a machine with more than 1 GPU)
	 * @return the device properties
	 */
	public static cudaDeviceProp getGPUProperties(int device){
		if (deviceProperties[device] == null) {
			cudaDeviceProp properties = new cudaDeviceProp();
			cudaGetDeviceProperties(properties, device);
			deviceProperties[device] = properties;
		}
		return deviceProperties[device];
	}


	/**
	 * Gets the maximum number of threads per block for "active" GPU
	 * @return the maximum number of threads per block
	 */
	public static int getMaxThreadsPerBlock() {
		cudaDeviceProp deviceProps = getGPUProperties();
		return deviceProps.maxThreadsPerBlock;
	}

	/**
	 * Gets the maximum number of blocks supported by the active cuda device
	 * @return the maximum number of blocks supported
	 */
	public static int getMaxBlocks() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.maxGridSize[0];
	}

	/**
	 * Gets the shared memory per block supported by the active cuda device
	 * @return the shared memory per block
	 */
	public static long getMaxSharedMemory() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.sharedMemPerBlock;
	}

	/**
	 * Gets the warp size supported by the active cuda device
	 * @return the warp size
	 */
	public static int getWarpSize() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.warpSize;
	}

	/**
	 * Gets the available memory and then adds value to it
	 * @param v the value to add
	 * @return the current available memory before adding value to it
	 */
	public long getAndAddAvailableMemory(long v){
		return deviceMemBytes.getAndAdd(v);
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
		synchronized (isGPUContextCreated){
			GPUContext.currContext = this;
		}

		long free [] = { 0 };
		long total [] = { 0 };
		long totalNumBytes = 0;
		if(cudaMemGetInfo(free, total) == cudaSuccess) {
			totalNumBytes = total[0];
			deviceMemBytes.set(free[0]);
		}
		else {
			throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
		}
		LOG.info("Total GPU memory: " + (totalNumBytes*(1e-6)) + " MB");
		LOG.info("Available GPU memory: " + (deviceMemBytes.get()*(1e-6)) + " MB");

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
		try {
			LibMatrixCUDA.kernels = new JCudaKernels();
		} catch (DMLRuntimeException e) {
			System.err.println("ERROR - Unable to initialize JCudaKernels. System in an inconsistent state");
			LibMatrixCUDA.kernels = null;
		}
		GPUStatistics.cudaLibrariesInitTime = System.nanoTime() - start;
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
