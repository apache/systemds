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

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcusolver.JCusolverDn.cusolverDnCreate;
import static jcuda.jcusolver.JCusolverDn.cusolverDnDestroy;
import static jcuda.jcusparse.JCusparse.cusparseCreate;
import static jcuda.jcusparse.JCusparse.cusparseDestroy;
import static jcuda.runtime.JCuda.cudaDeviceScheduleBlockingSync;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;
import static jcuda.runtime.JCuda.cudaSetDeviceFlags;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.utils.GPUStatistics;

import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * Represents a context per GPU accessible through the same JVM.
 * Each context holds cublas, cusparse, cudnn... handles which are separate for each GPU.
 */
public class GPUContext {

	protected static final Log LOG = LogFactory.getLog(GPUContext.class.getName());
	/**
	 * The minimum CUDA Compute capability needed for SystemDS.
	 * After compute capability 3.0, 2^31 - 1 blocks and 1024 threads per block are supported.
	 * If SystemDS needs to run on an older card, this logic can be revisited.
	 */
	final int MAJOR_REQUIRED = 3;
	final int MINOR_REQUIRED = 0;
	/**
	 * active device assigned to this GPUContext instance
	 */
	private final int deviceNum;
	
	/**
	 * cudnnHandle for Deep Neural Network operations on the GPU
	 */
	private cudnnHandle cudnnHandle;
	/**
	 * cublasHandle for BLAS operations on the GPU
	 */
	private cublasHandle cublasHandle;
	/**
	 * cusparseHandle for certain sparse BLAS operations on the GPU
	 */
	private cusparseHandle cusparseHandle;
	/**
	 * cusolverDnHandle for invoking solve() function on dense matrices on the GPU
	 */
	private cusolverDnHandle cusolverDnHandle;
	/**
	 * to launch custom CUDA kernel, specific to the active GPU for this GPUContext
	 */
	private JCudaKernels kernels;
	
	private GPUMemoryManager memoryManager;
	
	public GPUMemoryManager getMemoryManager() {
		return memoryManager;
	}

	protected GPUContext(int deviceNum) {
		this.deviceNum = deviceNum;
		cudaSetDevice(deviceNum);

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		long start = -1;
		if (DMLScript.STATISTICS)
			start = System.nanoTime();
		initializeCudaLibraryHandles();
		

		if (DMLScript.STATISTICS)
			GPUStatistics.cudaLibrariesInitTime = System.nanoTime() - start;

		memoryManager = new GPUMemoryManager(this);
	}

	/**
	 * Returns which device is currently being used.
	 *
	 * @return the current device for the calling host thread
	 */
	public static int cudaGetDevice() {
		int[] device = new int[1];
		JCuda.cudaGetDevice(device);
		return device[0];
	}

	/**
	 * Print information of memory usage.
	 *
	 * @param opcode opcode of caller
	 */
	public void printMemoryInfo(String opcode) {
		if (LOG.isDebugEnabled()) {
			LOG.debug(opcode + ": " + memoryManager.toString());
		}
	}

	private void initializeCudaLibraryHandles() throws DMLRuntimeException {
		// We don't need to explicitly delete the handles if we are planning to create them again. 
		// This has a huge performance impact on scripts that has large number of layers (i.e. FunctionCallCP) for example ResNet.
		// If this is absolutely required for parfor, please add appropriate safeguard for non-parfor scripts. 
		// deleteCudaLibraryHandles();
		if (cudnnHandle == null) {
			cudnnHandle = new cudnnHandle();
			cudnnCreate(cudnnHandle);
		}

		if (cublasHandle == null) {
			cublasHandle = new cublasHandle();
			cublasCreate(cublasHandle);
		}
		// For cublas v2, cublasSetPointerMode tells Cublas whether to expect scalar arguments on device or on host
		// This applies to arguments like "alpha" in Dgemm, and "y" in Ddot.
		// cublasSetPointerMode(LibMatrixCUDA.cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE);

		if (cusparseHandle == null) {
			cusparseHandle = new cusparseHandle();
			cusparseCreate(cusparseHandle);
		}
		
		if (kernels == null) {
			kernels = new JCudaKernels();
		}
	}

	/**
	 * Returns which device is assigned to this GPUContext instance.
	 *
	 * @return active device assigned to this GPUContext instance
	 */
	public int getDeviceNum() {
		return deviceNum;
	}

	/**
	 * Sets the device for the calling thread.
	 * This method must be called after
	 * {@link org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext#getGPUContext(int)}
	 * If in a multi-threaded environment like parfor, this method must be called when in the
	 * appropriate thread.
	 *
	 */
	public void initializeThread() {
		cudaSetDevice(deviceNum);
		initializeCudaLibraryHandles();
	}

	/**
	 * Invokes memory manager's malloc method
	 *
	 * @param instructionName name of instruction for which to record per instruction performance statistics, null if don't want to record
	 * @param size            size of data (in bytes) to allocate
	 * @return jcuda pointer
	 */
	public Pointer allocate(String instructionName, long size) {
		return memoryManager.malloc(instructionName, size);
	}

	/**
	 * Does cudaFree calls, lazily.
	 *
	 * @param instructionName name of the instruction for which to record per instruction free time, null if do not want to record
	 * @param toFree          {@link Pointer} instance to be freed
	 * @param eager           true if to be done eagerly
	 */
	public void cudaFreeHelper(String instructionName, final Pointer toFree, boolean eager) {
		memoryManager.free(instructionName, toFree, eager);
	}


	/**
	 * Gets the available memory on GPU that SystemDS can use.
	 *
	 * @return the available memory in bytes
	 */
	public long getAvailableMemory() {
		return memoryManager.allocator.getAvailableMemory();
	}

	/**
	 * Makes sure that GPU that SystemDS is trying to use has the minimum compute capability needed.
	 */
	public void ensureComputeCapability() {
		int[] devices = { -1 };
		cudaGetDeviceCount(devices);
		if (devices[0] == -1) {
			throw new DMLRuntimeException("Call to cudaGetDeviceCount returned 0 devices");
		}
		boolean isComputeCapable = true;
		for (int i = 0; i < devices[0]; i++) {
			cudaDeviceProp properties = GPUContextPool.getGPUProperties(i);
			int major = properties.major;
			int minor = properties.minor;
			if (major < MAJOR_REQUIRED) {
				isComputeCapable = false;
			} else if (major == MAJOR_REQUIRED && minor < MINOR_REQUIRED) {
				isComputeCapable = false;
			}
		}
		if (!isComputeCapable) {
			throw new DMLRuntimeException(
					"One of the CUDA cards on the system has compute capability lower than " + MAJOR_REQUIRED + "."
							+ MINOR_REQUIRED);
		}
	}

	/**
	 * Instantiates a new {@link GPUObject} initialized with the given {@link org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject MatrixObject}.
	 *
	 * @param mo a {@link org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject MatrixObject} that represents a matrix
	 * @return a new {@link GPUObject} instance
	 */
	public GPUObject createGPUObject(MatrixObject mo) {
		GPUObject ret = new GPUObject(this, mo);
		getMemoryManager().getGPUMatrixMemoryManager().addGPUObject(ret);
		return ret;
	}

	/**
	 * Gets the device properties for the active GPU (set with cudaSetDevice()).
	 *
	 * @return the device properties
	 */
	public cudaDeviceProp getGPUProperties() {
		return GPUContextPool.getGPUProperties(deviceNum);
	}

	/**
	 * Gets the maximum number of threads per block for "active" GPU.
	 *
	 * @return the maximum number of threads per block
	 */
	public int getMaxThreadsPerBlock() {
		cudaDeviceProp deviceProps = getGPUProperties();
		return deviceProps.maxThreadsPerBlock;
	}

	/**
	 * Gets the maximum number of blocks supported by the active cuda device.
	 *
	 * @return the maximum number of blocks supported
	 */
	public int getMaxBlocks() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.maxGridSize[0];
	}

	/**
	 * Gets the shared memory per block supported by the active cuda device.
	 *
	 * @return the shared memory per block
	 */
	public long getMaxSharedMemory() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.sharedMemPerBlock;
	}

	/**
	 * Gets the warp size supported by the active cuda device.
	 *
	 * @return the warp size
	 */
	public int getWarpSize() {
		cudaDeviceProp deviceProp = getGPUProperties();
		return deviceProp.warpSize;
	}

	/**
	 * Returns the cudnnHandle for Deep Neural Network operations on the GPU.
	 *
	 * @return cudnnHandle for current thread
	 */
	public cudnnHandle getCudnnHandle() {
		return cudnnHandle;
	}

	/**
	 * Returns cublasHandle for BLAS operations on the GPU.
	 *
	 * @return cublasHandle for current thread
	 */
	public cublasHandle getCublasHandle() {
		return cublasHandle;
	}

	/**
	 * Returns cusparseHandle for certain sparse BLAS operations on the GPU.
	 *
	 * @return cusparseHandle for current thread
	 */
	public cusparseHandle getCusparseHandle() {
		return cusparseHandle;
	}

	/**
	 * Returns cusolverDnHandle for invoking solve() function on dense matrices on the GPU.
	 *
	 * @return cusolverDnHandle for current thread
	 */
	public cusolverDnHandle getCusolverDnHandle() {
		if (cusolverDnHandle == null) {
			synchronized(this) {
				if (cusolverDnHandle == null) {
					// Since cusolverDnHandle handle is rarely used and occupies unnecessary memory, it is only initialized when needed.
					cusolverDnHandle = new cusolverDnHandle();
					cusolverDnCreate(cusolverDnHandle);
				}
			}
		}
		return cusolverDnHandle;
	}

	/**
	 * Returns utility class used to launch custom CUDA kernel, specific to the active GPU for this GPUContext.
	 *
	 * @return {@link JCudaKernels} for current thread
	 */
	public JCudaKernels getKernels() {
		return kernels;
	}

	/**
	 * Destroys this GPUContext object.
	 *
	 */
	public void destroy() {
		if (LOG.isTraceEnabled()) {
			LOG.trace("GPU : this context was destroyed, this = " + this.toString());
		}
		clearMemory();

		deleteCudaLibraryHandles();
	}

	/**
	 *	Deletes CUDA library handles
	 */
	private void deleteCudaLibraryHandles() {
		if (cudnnHandle != null)
			cudnnDestroy(cudnnHandle);

		if (cublasHandle != null)
			cublasDestroy(cublasHandle);

		if (cusparseHandle != null)
			cusparseDestroy(cusparseHandle);

		if (cusolverDnHandle != null)
			cusolverDnDestroy(cusolverDnHandle);

		cudnnHandle = null;
		cublasHandle = null;
		cusparseHandle = null;
		cusolverDnHandle = null;
	}

	/**
	 * Clears all memory used by this {@link GPUContext}.
	 * Be careful to ensure that no memory is currently being used in the temporary memory before invoking this.
	 * If memory is being used between MLContext invocations, they are pointed to by a {@link GPUObject} instance
	 * which would be part of the {@link MatrixObject}. The cleanup of that {@link MatrixObject} instance will
	 * cause the memory associated with that block on the GPU to be freed up.
	 */
	public void clearMemory() {
		memoryManager.clearMemory();
	}
	
	public void clearTemporaryMemory() {
		memoryManager.clearTemporaryMemory();
	}

	@Override
	public String toString() {
		return "GPUContext{" + "deviceNum=" + deviceNum + '}';
	}
}
