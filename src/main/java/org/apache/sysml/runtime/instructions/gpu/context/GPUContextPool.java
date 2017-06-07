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

import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class GPUContextPool {

	protected static final Log LOG = LogFactory.getLog(GPUContextPool.class.getName());

	/**
	 * Maximum number of gpus to use, -1 for all
	 */
	public static int PER_PROCESS_MAX_GPUS = -1;

	/**
	 * Whether cuda has been initialized
	 */
	static boolean initialized = false;

	/**
	 * The total number of cuda devices on this machine
	 */
	static int deviceCount = -1;

	/**
	 * Stores the cached deviceProperties
	 */
	static cudaDeviceProp[] deviceProperties;

	/**
	 * Set of free GPUContexts
	 */
	static Queue<GPUContext> freePool = new LinkedList<>();

	ArrayList<GPUContextPool> gpuContextPools = new ArrayList<>();

	/**
	 * Static initialization of the number of devices
	 * Also sets behaviour for J{Cuda, Cudnn, Cublas, Cusparse} in case of error
	 * Initializes the CUDA driver
	 * All these need be done once, and not per GPU
	 *
	 * @throws DMLRuntimeException ?
	 */
	public synchronized static void initializeGPU() throws DMLRuntimeException {
		GPUContext.LOG.info("Initializing CUDA");
		long start = System.nanoTime();
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCusparse.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0); // Initialize the driver

		int deviceCountArray[] = { 0 };
		cuDeviceGetCount(deviceCountArray);        // Obtain the number of devices
		deviceCount = deviceCountArray[0];
		deviceProperties = new cudaDeviceProp[deviceCount];

		if (PER_PROCESS_MAX_GPUS > 0)
			deviceCount = Math.min(PER_PROCESS_MAX_GPUS, deviceCount);

		// Initialize the list of devices
		for (int i = 0; i < deviceCount; i++) {
			cudaDeviceProp properties = new cudaDeviceProp();
			cudaGetDeviceProperties(properties, i);
			deviceProperties[i] = properties;
		}

		// Initialize the pool of GPUContexts
		for (int i = 0; i < deviceCount; i++) {
			GPUContext gCtx = new GPUContext(i);
			freePool.add(gCtx);
		}

		GPUContext.LOG.info("Total number of GPUs on the machine: " + deviceCount);
		//int[] device = {-1};
		//cudaGetDevice(device);
		//cudaDeviceProp prop = getGPUProperties(device[0]);
		//int maxBlocks = prop.maxGridSize[0];
		//int maxThreadsPerBlock = prop.maxThreadsPerBlock;
		//long sharedMemPerBlock = prop.sharedMemPerBlock;
		//LOG.debug("Active CUDA device number : " + device[0]);
		//LOG.debug("Max Blocks/Threads/SharedMem on active device: " + maxBlocks + "/" + maxThreadsPerBlock + "/" + sharedMemPerBlock);
		initialized = true;
		GPUStatistics.cudaInitTime = System.nanoTime() - start;
	}

	/**
	 * Gets an initialized GPUContext from a pool of GPUContexts, each linked to a GPU
	 *
	 * @return null if not more GPUContexts in pool, a valid GPUContext otherwise
	 * @throws DMLRuntimeException ?
	 */
	public static synchronized GPUContext getFromPool() throws DMLRuntimeException {
		if (!initialized)
			initializeGPU();
		GPUContext gCtx = freePool.poll();
		LOG.trace("GPU : got GPUContext (" + gCtx + ") from freePool. New sizes - FreePool[" + freePool.size() + "]");
		return gCtx;
	}

	/**
	 * Get the number of free GPUContexts
	 *
	 * @return number of free GPUContexts
	 */
	public static synchronized int getAvailableCount() {
		return freePool.size();
	}

	/**
	 * Gets the device properties
	 *
	 * @param device the device number (on a machine with more than 1 GPU)
	 * @return the device properties
	 * @throws DMLRuntimeException if there is problem initializing the GPUContexts
	 */
	static cudaDeviceProp getGPUProperties(int device) throws DMLRuntimeException {
		// do once - initialization of GPU
		if (!initialized)
			initializeGPU();
		return deviceProperties[device];
	}

	public static int getDeviceCount() throws DMLRuntimeException {
		if (!initialized)
			initializeGPU();
		return deviceCount;
	}

	/**
	 * Returns a {@link GPUContext} back to the pool of {@link GPUContext}s
	 *
	 * @param gCtx the GPUContext instance to return. If null, nothing happens
	 * @throws DMLRuntimeException if error
	 */
	public static synchronized void returnToPool(GPUContext gCtx) throws DMLRuntimeException {
		if (gCtx == null)
			return;
		freePool.add(gCtx);
		LOG.trace(
				"GPU : returned GPUContext (" + gCtx + ") to freePool. New sizes - FreePool[" + freePool.size() + "]");

	}

}
