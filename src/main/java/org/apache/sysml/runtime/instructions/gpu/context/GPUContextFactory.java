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

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class GPUContextFactory {

  /** Whether cuda has been initialized */
  static boolean initialized = false;

  /** The total number of cuda devices on this machine */
  static int deviceCount = -1;

  /** The list of free devices */
  static Queue<Integer> freeDevices = new LinkedList<>();

  /**
   * Maintiains a mapping of threads assigned to GPUContexts. Each thread  is given a new GPUContext
   * as long as there are enough GPUs to accommodate them
   * The size of this map also represents the number of GPUs that are in use
   */
  static HashMap<Thread, GPUContext> assignedGPUContextsMap = new HashMap<>();

  /** Stores the cached deviceProperties */
  static cudaDeviceProp[] deviceProperties;

  /**
   * Static initialization of the number of devices
   * Also sets behaviour for J{Cuda, Cudnn, Cublas, Cusparse} in case of error
   * Initializes the CUDA driver
   * All these need be done once, and not per GPU
   */
  public synchronized static void initializeGPU() {
    GPUContext.LOG.info("Initializing CUDA");
    long start = System.nanoTime();
    JCuda.setExceptionsEnabled(true);
    JCudnn.setExceptionsEnabled(true);
    JCublas2.setExceptionsEnabled(true);
    JCusparse.setExceptionsEnabled(true);
    JCudaDriver.setExceptionsEnabled(true);
    cuInit(0); // Initialize the driver

    int deviceCountArray[] = {0};
    cuDeviceGetCount(deviceCountArray);        // Obtain the number of devices
    deviceCount = deviceCountArray[0];
    deviceProperties = new cudaDeviceProp[deviceCount];

    // Initialize the list of devices
    for (int i = 0; i < deviceCount; i++) {
      cudaDeviceProp properties = new cudaDeviceProp();
      cudaGetDeviceProperties(properties, i);
      deviceProperties[i] = properties;
    }

    // Populate the list of free devices
    for (int i = 0; i < deviceCount; i++) {
      freeDevices.add(i);
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
   * Singleton Factory method for creation/retrieval of a {@link GPUContext} for the current thread
   * This method is threadsafe
   * Each {@link GPUContext} instance is attached to a thread and a physical GPU
   *
   * @return a valid {@link GPUContext} instance or null if no more GPUs available
   * @throws DMLRuntimeException if DMLRuntimeException occurs
   */
  public synchronized static GPUContext createGPUContext() throws DMLRuntimeException {
    // do once - initialization of GPU
    if (!initialized) initializeGPU();
    // If the singleton for the current thread has already been created, well and
    // good. Other wise create a new GPUContext object, provided there are enough number
    // of GPUs left on the system
    Thread thisThread = Thread.currentThread();
    GPUContext activeGPUContext = assignedGPUContextsMap.get(thisThread);
    if (activeGPUContext == null) {
      Integer deviceNum = freeDevices.poll();
      if (deviceNum == null) { // no more devices to allocate
        return null;
      }
      activeGPUContext = new GPUContext(deviceNum);
      activeGPUContext.ensureComputeCapability();
      OptimizerUtils.GPU_MEMORY_BUDGET = activeGPUContext.getAvailableMemory();
      assignedGPUContextsMap.put(thisThread, activeGPUContext);
      GPUContext.LOG.trace("Created context for thread " + thisThread + ", with GPU " + deviceNum);
    }
    return activeGPUContext;
  }

  /**
   * Gets the device properties
   * @param device the device number (on a machine with more than 1 GPU)
   * @return the device properties
   */
  static cudaDeviceProp getGPUProperties(int device){
    // do once - initialization of GPU
    if (!initialized) initializeGPU();
    return deviceProperties[device];
  }
}
