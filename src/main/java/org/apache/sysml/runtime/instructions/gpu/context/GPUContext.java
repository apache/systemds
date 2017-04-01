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

import jcuda.Pointer;
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
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.LRUCacheMap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.JCuda.cudaMemset;
import static jcuda.runtime.JCuda.cudaSetDevice;

/**
 * Represents a context per GPU accessible through the same JVM
 * Each context holds cublas, cusparse, cudnn... handles which are separate for each GPU
 */
public class GPUContext {

	protected static final Log LOG = LogFactory.getLog(GPUContext.class.getName());

	/** Eviction policies for {@link GPUContext#evict(long)} */
	public enum EvictionPolicy {
		LRU, LFU, MIN_EVICT
	}

	/** currently employed eviction policy */
	public final EvictionPolicy evictionPolicy = EvictionPolicy.LRU;

	/** Map of free blocks allocate on GPU. maps size_of_block -> pointer on GPU */
	private LRUCacheMap<Long, LinkedList<Pointer>> freeCUDASpaceMap = new LRUCacheMap<>();

	/** To record size of allocated blocks */
	private HashMap<Pointer, Long> cudaBlockSizeMap = new HashMap<>();

	/** Whether cuda has been initialized */
  private static boolean initialized = false;

  /** The total number of cuda devices on this machine */
  private static int deviceCount = -1;

  /** The list of free devices */
  private static Queue<Integer> freeDevices = new LinkedList<>();

  /** Stores the cached deviceProperties */
  private static cudaDeviceProp[] deviceProperties;

  /** Maintiains a mapping of threads assigned to GPUContexts. Each thread  is given a new GPUContext
   * as long as there are enough GPUs to accommodate them
   * The size of this map also represents the number of GPUs that are in use */
  private static HashMap<Thread, GPUContext> assignedGPUContextsMap = new HashMap<>();

  /** active device assigned to this GPUContext instance */
  private int deviceNum = -1;

  /** list of allocated {@link GPUObject} instances allocated on {@link GPUContext#deviceNum} GPU
   * These are matrices allocated on the GPU on which rmvar hasn't been called yet.
   * If a {@link GPUObject} has more than one lock on it, it cannot be freed
   * If it has zero locks on it, it can be freed, but it is preferrable to keep it around
   * so that an extraneous host to dev transfer can be avoided */
  private ArrayList<GPUObject> allocatedGPUObjects = new ArrayList<>();

  /** cudnnHandle specific to the active GPU for this GPUContext */
  private cudnnHandle cudnnHandle;

  /** cublasHandle specific to the active GPU for this GPUContext */
  private cublasHandle cublasHandle;

  /** cusparseHandle specific to the active GPU for this GPUContext */
  private cusparseHandle cusparseHandle;

  /** to launch custom CUDA kernel, specific to the active GPU for this GPUContext */
  private JCudaKernels kernels;

  /**
   * Static initialization of the number of devices
   * Also sets behaviour for J{Cuda, Cudnn, Cublas, Cusparse} in case of error
   * Initializes the CUDA driver
   * All these need be done once, and not per GPU
   */
  public synchronized static void initializeGPU() {
    LOG.info("Initializing CUDA");
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

    // Initialize the list of devices
    for (int i=0; i<deviceCount; i++){
			cudaDeviceProp properties = new cudaDeviceProp();
			cudaGetDeviceProperties(properties, i);
			deviceProperties[i] = properties;
		}

    // Populate the list of free devices
    for (int i=0; i<deviceCount; i++){
      freeDevices.add(i);
    }

    LOG.info("Total number of GPUs on the machine: " + deviceCount);
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
	 * The minimum CUDA Compute capability needed for SystemML.
	 * After compute capability 3.0, 2^31 - 1 blocks and 1024 threads per block are supported.
	 * If SystemML needs to run on an older card, this logic can be revisited.
	 */
	final int MAJOR_REQUIRED = 3;
	final int MINOR_REQUIRED = 0;

	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application.
	public double GPU_MEMORY_UTILIZATION_FACTOR = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);

	/**
	 * Convenience method for {@link #allocate(String, long, int)}, defaults statsCount to 1.
	 * @param size size of data (in bytes) to allocate
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public Pointer allocate(long size) throws DMLRuntimeException {
		return allocate(null, size, 1);
	}

	/**
	 * Convenience method for {@link #allocate(String, long, int)}, defaults statsCount to 1.
	 * @param instructionName name of instruction for which to record per instruction performance statistics, null if don't want to record
	 * @param size size of data (in bytes) to allocate
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public Pointer allocate(String instructionName, long size) throws DMLRuntimeException {
		return allocate(instructionName, size, 1);
	}

	/**
	 * Allocates temporary space on the device.
	 * Does not update bookkeeping.
	 * The caller is responsible for freeing up after usage.
	 * @param instructionName name of instruction for which to record per instruction performance statistics, null if don't want to record
	 * @param size   			Size of data (in bytes) to allocate
	 * @param statsCount	amount to increment the cudaAllocCount by
	 * @return jcuda Pointer
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public Pointer allocate(String instructionName, long size, int statsCount) throws DMLRuntimeException{
		long t0=0, t1=0, end=0;
		Pointer A;
		if (freeCUDASpaceMap.containsKey(size)) {
			LOG.trace("GPU : in allocate from instruction " + instructionName + ", found free block of size " + (size/1024.0) + " Kbytes from previously allocated block on " + this);
			if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) t0 = System.nanoTime();
			LinkedList<Pointer> freeList = freeCUDASpaceMap.get(size);
			A = freeList.pop();
			if (freeList.isEmpty())
				freeCUDASpaceMap.remove(size);
			if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_REUSE, System.nanoTime() - t0);
		} else {
			LOG.trace("GPU : in allocate from instruction " + instructionName + ", allocating new block of size " + (size/1024.0) + " Kbytes on " + this);
			if (DMLScript.STATISTICS) t0 = System.nanoTime();
			GPUObject.ensureFreeSpace(instructionName, size);
			A = new Pointer();
			cudaMalloc(A, size);
			if (DMLScript.STATISTICS) GPUStatistics.cudaAllocTime.getAndAdd(System.nanoTime() - t0);
			if (DMLScript.STATISTICS) GPUStatistics.cudaAllocCount.getAndAdd(statsCount);
			if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_ALLOCATE, System.nanoTime() - t0);
		}
		// Set all elements to 0 since newly allocated space will contain garbage
		if (DMLScript.STATISTICS) t1 = System.nanoTime();
		LOG.trace("GPU : in allocate from instruction " + instructionName + ", setting block of size " + (size/1024.0) + " Kbytes to zero on " + this);
		cudaMemset(A, 0, size);
		if (DMLScript.STATISTICS) end = System.nanoTime();
		if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_SET_ZERO, end - t1);
		if (DMLScript.STATISTICS) GPUStatistics.cudaMemSet0Time.getAndAdd(end - t1);
		if (DMLScript.STATISTICS) GPUStatistics.cudaMemSet0Count.getAndAdd(1);
		cudaBlockSizeMap.put(A, size);
		return A;

	}

	/**
	 * Does lazy cudaFree calls
	 * @param toFree {@link Pointer} instance to be freed
	 */
	public void cudaFreeHelper(final Pointer toFree) {
		cudaFreeHelper(null, toFree, false);
	}

	/**
	 * does lazy/eager cudaFree calls
	 * @param toFree {@link Pointer} instance to be freed
	 * @param eager true if to be done eagerly
	 */
	public void cudaFreeHelper(final Pointer toFree, boolean eager) {
		cudaFreeHelper(null, toFree, eager);
	}

	/**
	 * Does lazy cudaFree calls
	 * @param instructionName name of the instruction for which to record per instruction free time, null if do not want to record
	 * @param toFree {@link Pointer} instance to be freed
	 */
	public void cudaFreeHelper(String instructionName, final Pointer toFree) {
		cudaFreeHelper(instructionName, toFree, false);
	}

	/**
	 * Does cudaFree calls, lazily
	 * @param instructionName name of the instruction for which to record per instruction free time, null if do not want to record
	 * @param toFree {@link Pointer} instance to be freed
	 * @param eager true if to be done eagerly
	 */
	public void cudaFreeHelper(String instructionName, final Pointer toFree, boolean eager){
		long t0 = 0;
		assert cudaBlockSizeMap.containsKey(toFree) : "ERROR : Internal state corrupted, cache block size map is not aware of a block it trying to free up";
		long size = cudaBlockSizeMap.get(toFree);
		if (eager) {
			LOG.trace("GPU : eagerly freeing cuda memory [ " + toFree + " ] for instruction " + instructionName + " on " + this);
			if (DMLScript.STATISTICS) t0 = System.nanoTime();
			cudaFree(toFree);
			cudaBlockSizeMap.remove(toFree);
			if (DMLScript.STATISTICS) GPUStatistics.cudaDeAllocTime.addAndGet(System.nanoTime() - t0);
			if (DMLScript.STATISTICS) GPUStatistics.cudaDeAllocCount.addAndGet(1);
			if (instructionName != null && GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instructionName, GPUInstruction.MISC_TIMER_CUDA_FREE, System.nanoTime() - t0);
		} else {
			LOG.trace("GPU : lazily freeing cuda memory for instruction " + instructionName + " on " + this);
			LinkedList<Pointer> freeList = freeCUDASpaceMap.get(size);
			if (freeList == null) {
				freeList = new LinkedList<Pointer>();
				freeCUDASpaceMap.put(size, freeList);
			}
			freeList.add(toFree);
		}
	}

	/**
	 * Convenience wrapper over {@link GPUContext#evict(String, long)}
	 * @param GPUSize Desired size to be freed up on the GPU
	 * @throws DMLRuntimeException If no blocks to free up or if not enough blocks with zero locks on them.
	 */
	protected void evict(final long GPUSize) throws DMLRuntimeException {
		evict(null, GPUSize);
	}

	/**
	 * Memory on the GPU is tried to be freed up until either a chunk of needed size is freed up
	 * or it fails.
	 * First the set of reusable blocks is freed up. If that isn't enough, the set of allocated matrix
	 * blocks with zero locks on them is freed up.
	 * The process cycles through the sorted list of allocated {@link GPUObject} instances. Sorting is based on
	 * number of (read) locks that have been obtained on it (reverse order). It repeatedly frees up
	 * blocks on which there are zero locks until the required size has been freed up.
	 * // TODO: update it with hybrid policy
	 * @param instructionName name of the instruction for which performance measurements are made
	 * @param neededSize desired size to be freed up on the GPU
	 * @throws DMLRuntimeException If no reusable memory blocks to free up or if not enough matrix blocks with zero locks on them.
	 */
	protected void evict(String instructionName, final long neededSize) throws DMLRuntimeException {
		LOG.trace("GPU : evict called from " + instructionName + " for size " + neededSize + " on " + this);
		GPUStatistics.cudaEvictionCount.addAndGet(1);
		// Release the set of free blocks maintained in a GPUObject.freeCUDASpaceMap
		// to free up space
		LRUCacheMap<Long, LinkedList<Pointer>> lruCacheMap = freeCUDASpaceMap;
		while (lruCacheMap.size() > 0) {
			if (neededSize <= getAvailableMemory())
				break;
			Map.Entry<Long, LinkedList<Pointer>> toFreeListPair = lruCacheMap.removeAndGetLRUEntry();
			LinkedList<Pointer> toFreeList = toFreeListPair.getValue();
			Long size = toFreeListPair.getKey();
			Pointer toFree = toFreeList.pop();
			if (toFreeList.isEmpty())
				lruCacheMap.remove(size);
			cudaFreeHelper(instructionName, toFree, true);
		}

		if (neededSize <= getAvailableMemory())
			return;

		if (allocatedGPUObjects.size() == 0) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
		}

		Collections.sort(allocatedGPUObjects, new Comparator<GPUObject>() {
			@Override
			public int compare(GPUObject p1, GPUObject p2) {
				long p1Val = p1.readLocks.get();
				long p2Val = p2.readLocks.get();

				if (p1Val > 0 && p2Val > 0) {
					// Both are locked, so don't sort
					return 0;
				} else if (p1Val > 0 || p2Val > 0) {
					// Put the unlocked one to RHS
					return Long.compare(p2Val, p1Val);
				} else {
					// Both are unlocked

					if (evictionPolicy == EvictionPolicy.MIN_EVICT) {
						long p1Size = 0;
						long p2Size = 0;
						try {
							p1Size = p1.getSizeOnDevice() - neededSize;
							p2Size = p2.getSizeOnDevice() - neededSize;
						} catch (DMLRuntimeException e) {
							throw new RuntimeException(e);
						}

						if (p1Size >= 0 && p2Size >= 0) {
							return Long.compare(p2Size, p1Size);
						} else {
							return Long.compare(p1Size, p2Size);
						}
					} else if (evictionPolicy == EvictionPolicy.LRU || evictionPolicy == EvictionPolicy.LFU) {
						return Long.compare(p2.timestamp.get(), p1.timestamp.get());
					} else {
						throw new RuntimeException("Unsupported eviction policy:" + evictionPolicy.name());
					}
				}
			}
		});

		while (neededSize > getAvailableMemory() && allocatedGPUObjects.size() > 0) {
			GPUObject toBeRemoved = allocatedGPUObjects.get(allocatedGPUObjects.size() - 1);
			if (toBeRemoved.readLocks.get() > 0) {
				throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
			}
			if (toBeRemoved.dirty) {
				toBeRemoved.copyFromDeviceToHost();
			}

			toBeRemoved.clearData(true);
		}
	}

	/**
	 * @see GPUContext#allocatedGPUObjects
	 * Records the usage of a matrix block
	 * @param o {@link GPUObject} instance to record
	 */
	public void recordBlockUsage(GPUObject o) {
		allocatedGPUObjects.add(o);
	}

	/**
	 * @see GPUContext#allocatedGPUObjects
	 * Records that a block is not used anymore
	 * @param o {@link GPUObject} instance to remove from the list of allocated GPU objects
	 */
	public void removeRecordedUsage(GPUObject o){
		allocatedGPUObjects.remove(o);
	}

	/**
	 * Gets the available memory on GPU that SystemML can use
	 * @return the available memory in bytes
	 */
	public long getAvailableMemory() {
		long free[] = {0};
		long total[] = {0};
		cudaMemGetInfo(free, total);
		return (long) (free[0] * GPU_MEMORY_UTILIZATION_FACTOR);
	}

	/**
	 * Makes sure that GPU that SystemML is trying to use has the minimum compute capability needed
	 * @throws DMLRuntimeException if the compute capability is less than what is required
	 */
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

	protected GPUContext(int deviceNum) throws DMLRuntimeException {
		this.deviceNum = deviceNum;
		cudaSetDevice(deviceNum);

		long free[] = {0};
		long total[] = {0};
		cudaMemGetInfo(free, total);
		LOG.info("Total GPU memory: " + (total[0] * (1e-6)) + " MB");
		LOG.info("Available GPU memory: " + (free[0] * (1e-6)) + " MB");

		long start = System.nanoTime();
		cudnnHandle = new cudnnHandle();
		cudnnCreate(cudnnHandle);
		cublasHandle = new cublasHandle();
		cublasCreate(cublasHandle);
		// For cublas v2, cublasSetPointerMode tells Cublas whether to expect scalar arguments on device or on host
		// This applies to arguments like "alpha" in Dgemm, and "y" in Ddot.
		// cublasSetPointerMode(LibMatrixCUDA.cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE);
		cusparseHandle = new cusparseHandle();
		cusparseCreate(cusparseHandle);
    kernels = new JCudaKernels();

		GPUStatistics.cudaLibrariesInitTime = System.nanoTime() - start;
	}

	/**
	 * Singleton Factory method for creation/retrieval of a {@link GPUContext} for the current thread
   * This method is threadsafe
   * Each {@link GPUContext} instance is attached to a thread and a physical GPU
	 * @return a valid {@link GPUContext} instance or null if no more GPUs available
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public synchronized static GPUContext getGPUContext() throws DMLRuntimeException {
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
		  LOG.trace("Created context for thread " + thisThread + ", with GPU " + deviceNum);
    }
    return activeGPUContext;
	}

	public GPUObject createGPUObject(MatrixObject mo) {
    return new GPUObject(mo);
	}

	/**
	 * Gets the device properties for the active GPU (set with cudaSetDevice())
	 * @return the device properties
	 */
	public cudaDeviceProp getGPUProperties() {
		return getGPUProperties(deviceNum);
	}

	/**
	 * Gets the device properties
	 * @param device the device number (on a machine with more than 1 GPU)
	 * @return the device properties
	 */
	private static cudaDeviceProp getGPUProperties(int device){
		// do once - initialization of GPU
		if (!initialized) initializeGPU();
		return deviceProperties[device];
	}

	/**
	 * Gets the maximum number of threads per block for "active" GPU
	 * @return the maximum number of threads per block
	 */
	public int getMaxThreadsPerBlock() {
		cudaDeviceProp deviceProps = getGPUProperties();
		return deviceProps.maxThreadsPerBlock;
	}

	/**
	 * Gets the maximum number of blocks supported by the active cuda device
	 * @return the maximum number of blocks supported
	 */
	public int getMaxBlocks() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.maxGridSize[0];
	}

	/**
	 * Gets the shared memory per block supported by the active cuda device
	 * @return the shared memory per block
	 */
	public long getMaxSharedMemory() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.sharedMemPerBlock;
	}

	/**
	 * Gets the warp size supported by the active cuda device
	 * @return the warp size
	 */
	public int getWarpSize() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.warpSize;
	}


  public cudnnHandle getCudnnHandle() {
    return cudnnHandle;
  }

  public cublasHandle getCublasHandle() {
    return cublasHandle;
  }

  public cusparseHandle getCusparseHandle() {
    return cusparseHandle;
  }

  public JCudaKernels getKernels() {
    return kernels;
  }

  public static int getDeviceCount() {
	  if (!initialized) initializeGPU();
	  return deviceCount;
  }

  @SuppressWarnings("unused")
  public static int cudaGetDevice() {
		int[] device = new int[1];
		JCuda.cudaGetDevice(device);
		return device[0];
	}

  /**
   * Destroys this GPUContext object
   * This method MUST BE called so that the GPU is available to be used again
   * @throws DMLRuntimeException
   */
	public void destroy() throws DMLRuntimeException {
    LOG.trace("GPU : this context was destroyed, this = " + this.toString());
	  synchronized (GPUContext.class) {
      assignedGPUContextsMap.entrySet().removeIf(e -> e.getValue().equals(this));
      freeDevices.add(deviceNum);
			LOG.trace("GPU : removed this from contexts map (size : " + assignedGPUContextsMap.size() + ") and added freed up device back into freeDevices (size : " + freeDevices.size()+ ")");

		}
    cudnnDestroy(cudnnHandle);
    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
	}

	@Override
	public String toString() {
		return "GPUContext{" +
						"deviceNum=" + deviceNum +
						", cudnnHandle=" + cudnnHandle +
						", cublasHandle=" + cublasHandle +
						", cusparseHandle=" + cusparseHandle +
						'}';
	}
}
