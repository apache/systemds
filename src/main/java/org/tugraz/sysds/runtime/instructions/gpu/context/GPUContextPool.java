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
package org.tugraz.sysds.runtime.instructions.gpu.context;

import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.utils.GPUStatistics;

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class GPUContextPool {

	protected static final Log LOG = LogFactory.getLog(GPUContextPool.class.getName());

	/**
	 * GPUs to use, can specify -1 to use all, comma separated list of GPU numbers, a specific GPU or a range
	 */
	public static String AVAILABLE_GPUS;


	private static long INITIAL_GPU_MEMORY_BUDGET = -1;

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
	static List<GPUContext> pool = new LinkedList<>();

	/**
	 * Whether the pool of GPUs is reserved or not
	 */
	static boolean reserved = false;

	/**
	 * Static initialization of the number of devices
	 * Also sets behaviour for J{Cuda, Cudnn, Cublas, Cusparse} in case of error
	 * Initializes the CUDA driver
	 * All these need be done once, and not per GPU
	 */
	public synchronized static void initializeGPU() {
		initialized = true;
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

		try {
			ArrayList<Integer> listOfGPUs = parseListString(AVAILABLE_GPUS, deviceCount);

			// Initialize the list of devices & the pool of GPUContexts
			for (int i : listOfGPUs) {
				cudaDeviceProp properties = new cudaDeviceProp();
				cudaGetDeviceProperties(properties, i);
				deviceProperties[i] = properties;
				GPUContext gCtx = new GPUContext(i);
				pool.add(gCtx);
			}

		} catch (IllegalArgumentException e) {
			LOG.warn("Invalid setting for setting systemds.gpu.availableGPUs, defaulting to use ALL GPUs");

			// Initialize the list of devices & the pool of GPUContexts
			for (int i = 0; i < deviceCount; i++) {
				cudaDeviceProp properties = new cudaDeviceProp();
				cudaGetDeviceProperties(properties, i);
				deviceProperties[i] = properties;
				GPUContext gCtx = new GPUContext(i);
				pool.add(gCtx);
			}
		}


		// Initialize the initial memory budget
		// If there are heterogeneous GPUs on the machine (different memory sizes)
		// initially available memory is set to the GPU with the lowest memory
		// This is because at runtime, we wouldn't know which GPU a certain
		// operation gets scheduled on
		long minAvailableMemory = Long.MAX_VALUE;
		for (GPUContext gCtx : pool) {
			gCtx.initializeThread();
			minAvailableMemory = Math.min(minAvailableMemory, gCtx.getAvailableMemory());
		}
		INITIAL_GPU_MEMORY_BUDGET = minAvailableMemory;


		GPUContext.LOG.info("Total number of GPUs on the machine: " + deviceCount);
		GPUContext.LOG.info("GPUs being used: " + AVAILABLE_GPUS);
		GPUContext.LOG.info("Initial GPU memory: " + initialGPUMemBudget());

		//int[] device = {-1};
		//cudaGetDevice(device);
		//cudaDeviceProp prop = getGPUProperties(device[0]);
		//int maxBlocks = prop.maxGridSize[0];
		//int maxThreadsPerBlock = prop.maxThreadsPerBlock;
		//long sharedMemPerBlock = prop.sharedMemPerBlock;
		//LOG.debug("Active CUDA device number : " + device[0]);
		//LOG.debug("Max Blocks/Threads/SharedMem on active device: " + maxBlocks + "/" + maxThreadsPerBlock + "/" + sharedMemPerBlock);
		GPUStatistics.cudaInitTime = System.nanoTime() - start;
	}

	/**
	 * Parses a string into a list. The string can be of these forms:
	 * 1. "-1" : all integers from range 0 to max - [0,1,2,3....max]
	 * 2. "2,3,0" : comma separated list of integers - [0,2,3]
	 * 3. "4" : a specific integer - [4]
	 * 4. "0-4" : a range of integers - [0,1,2,3,4]
	 * In ranges and comma separated lists, all values must be positive. Anything else is invalid.
	 * @param str input string
	 * @param max maximum range of integers
	 * @return the list of integers in the parsed string
	 */
	public static ArrayList<Integer> parseListString(String str, int max) {
		ArrayList<Integer> result = new ArrayList<>();
		str = str.trim();
		if (str.equalsIgnoreCase("-1")) {  // all
			for (int i=0; i<max; i++){
				result.add(i);
			}
		} else if (str.contains("-")){  // range
			String[] numbersStr = str.split("-");
			if (numbersStr.length != 2) {
				throw new IllegalArgumentException("Invalid string to parse to a list of numbers : " + str);
			}
			String beginStr = numbersStr[0];
			String endStr = numbersStr[1];
			int begin = Integer.parseInt(beginStr);
			int end = Integer.parseInt(endStr);

			for (int i=begin; i<=end; i++){
				result.add(i);
			}
		} else if (str.contains(",")) { // comma separated list
			String[] numbers = str.split(",");
			for (int i = 0; i < numbers.length; i++) {
				int n = Integer.parseInt(numbers[i].trim());
				result.add(n);
			}
		} else {  // single number
			int number = Integer.parseInt(str);
			result.add(number);
		}
		// Check if all numbers between 0 and max
		for (int n : result){
			if (n < 0 || n >= max) {
				throw new IllegalArgumentException("Invalid string (" + str + ") parsed to a list of numbers (" + result + ") which exceeds the maximum range : ");
			}
		}
		return result;
	}

	/**
	 * Reserves and gets an initialized list of GPUContexts
	 *
	 * @return null if no GPUContexts in pool, otherwise a valid list of GPUContext
	 */
	public static synchronized List<GPUContext> reserveAllGPUContexts() {
		if (reserved)
			throw new DMLRuntimeException("Trying to re-reserve GPUs");
		if (!initialized)
			initializeGPU();
		reserved = true;
		LOG.trace("GPU : Reserved all GPUs");
		return pool;
	}

	/**
	 * Get the number of free GPUContexts
	 *
	 * @return number of free GPUContexts
	 */
	public static synchronized int getAvailableCount() {
		return pool.size();
	}

	/**
	 * Gets the device properties
	 *
	 * @param device the device number (on a machine with more than 1 GPU)
	 * @return the device properties
	 */
	static cudaDeviceProp getGPUProperties(int device) {
		// do once - initialization of GPU
		if (!initialized)
			initializeGPU();
		return deviceProperties[device];
	}

	/**
	 * Number of available devices on this machine
	 *
	 * @return number of available GPUs on this machine
	 */
	public static int getDeviceCount() {
		if (!initialized)
			initializeGPU();
		return deviceCount;
	}

	/**
	 * Unreserves all GPUContexts
	 */
	public static synchronized void freeAllGPUContexts() {
		if (!reserved)
			throw new DMLRuntimeException("Trying to free unreserved GPUs");
		reserved = false;
		LOG.trace("GPU : Unreserved all GPUs");

	}

	/**
	 * Gets the initial GPU memory budget. This is the minimum of the
	 * available memories across all the GPUs on the machine(s)
	 * @return minimum available memory
	 * @throws RuntimeException if error initializing the GPUs
	 */
	public static synchronized long initialGPUMemBudget() throws RuntimeException {
		try {
			if (!initialized)
				initializeGPU();
			return INITIAL_GPU_MEMORY_BUDGET;
		} catch (DMLRuntimeException e){
			throw new RuntimeException(e);
		}
	}
}
