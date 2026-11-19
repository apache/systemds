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

package org.apache.sysds.cujava.runtime;

/**
 * This class replicates the CUDA device properties (cudaDeviceProp).
 * The descriptions are directly taken from the Documentation:
 * https://docs.nvidia.com/cuda/archive/12.8.0/pdf/CUDA_Runtime_API.pdf
 */
public class CudaDeviceProp {

	/**
	 * The maximum value of cudaAccessPolicyWindow::num_bytes.
	 */
	public int accessPolicyMaxWindowSize;

	/**
	 * Number of asynchronous engines
	 */
	public int asyncEngineCount;

	/**
	 * Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
	 */
	public int canMapHostMemory;

	/**
	 * Device can access host registered memory at the same virtual address as the CPU
	 */
	public int canUseHostPointerForRegisteredMem;

	/**
	 * @deprecated in CUDA 12 Clock frequency in kilohertz
	 */
	public int clockRate;

	/**
	 * Indicates device supports cluster launch
	 */
	public int clusterLaunch;

	/**
	 * @deprecated Compute mode (See cudaComputeMode)
	 */
	public int computeMode;

	/**
	 * Device supports Compute Preemption
	 */
	public int computePreemptionSupported;

	/**
	 * Device can possibly execute multiple kernels concurrently
	 */
	public int concurrentKernels;

	/**
	 * Device can coherently access managed memory concurrently with the CPU
	 */
	public int concurrentManagedAccess;

	/**
	 * Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
	 */
	public int cooperativeLaunch;

	/**
	 * @deprecated cudaLaunchCooperativeKernelMultiDevice is deprecated.
	 */
	public int cooperativeMultiDeviceLaunch;

	/**
	 * 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays
	 */
	public int deferredMappingCudaArraySupported;

	/**
	 * Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.
	 */
	public int deviceOverlap;

	/**
	 * Host can directly access managed memory on the device without migration.
	 */
	public int directManagedMemAccessFromHost;

	/**
	 * Device has ECC support enabled
	 */
	public int ECCEnabled;

	/**
	 * Device supports caching globals in L1
	 */
	public int globalL1CacheSupported;

	/**
	 * Bitmask to be interpreted according to the cudaFlushGPUDirectRDMAWritesOptions enum
	 */
	public int gpuDirectRDMAFlushWritesOptions;

	/**
	 * 1 if the device supports GPUDirect RDMA APIs, 0 otherwise
	 */
	public int gpuDirectRDMASupported;

	/**
	 * See the cudaGPUDirectRDMAWritesOrdering enum for numerical values
	 */
	public int gpuDirectRDMAWritesOrdering;

	/**
	 * Link between the device and the host supports native atomic operations
	 */
	public int hostNativeAtomicSupported;

	/**
	 * Device supports using the cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped
	 * as read-only to the GPU
	 */
	public int hostRegisterReadOnlySupported;

	/**
	 * Device supports host memory registration via cudaHostRegister.
	 */
	public int hostRegisterSupported;

	/**
	 * Device is integrated as opposed to discrete
	 */
	public int integrated;

	/**
	 * Device supports IPC Events.
	 */
	public int ipcEventSupported;

	/**
	 * Device is on a multi-GPU board
	 */
	public int isMultiGpuBoard;

	/**
	 * @deprecated Specified whether there is a run time limit on kernels
	 */
	public int kernelExecTimeoutEnabled;

	/**
	 * Size of L2 cache in bytes
	 */
	public int l2CacheSize;

	/**
	 * Device supports caching locals in L1
	 */
	public int localL1CacheSupported;

	/**
	 * 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms
	 */
	public byte[] luid = new byte[8];

	/**
	 * LUID device node mask. Value is undefined on TCC and non-Windows platforms
	 */
	public int luidDeviceNodeMask;

	/**
	 * Major compute capability
	 */
	public int major;

	/**
	 * Device supports allocating managed memory on this system
	 */
	public int managedMemory;

	/**
	 * Maximum number of resident blocks per multiprocessor
	 */
	public int maxBlocksPerMultiProcessor;

	/**
	 * Maximum size of each dimension of a grid
	 */
	public int[] maxGridSize = new int[3];

	/**
	 * Maximum 1D surface size
	 */
	public int maxSurface1D;

	/**
	 * Maximum 1D layered surface dimensions
	 */
	public int[] maxSurface1DLayered = new int[2];

	/**
	 * Maximum 2D surface dimensions
	 */
	public int[] maxSurface2D = new int[2];

	/**
	 * Maximum 2D layered surface dimensions
	 */
	public int[] maxSurface2DLayered = new int[3];

	/**
	 * Maximum 3D surface dimensions
	 */
	public int[] maxSurface3D = new int[3];

	/**
	 * Maximum Cubemap surface dimensions
	 */
	public int maxSurfaceCubemap;

	/**
	 * Maximum Cubemap layered surface dimensions
	 */
	public int[] maxSurfaceCubemapLayered = new int[2];

	/**
	 * Maximum 1D texture size
	 */
	public int maxTexture1D;

	/**
	 * Maximum 1D layered texture dimensions
	 */
	public int[] maxTexture1DLayered = new int[2];

	/**
	 * @deprecated Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
	 */
	public int maxTexture1DLinear;

	/**
	 * Maximum 1D mipmapped texture size
	 */
	public int maxTexture1DMipmap;

	/**
	 * Maximum 2D texture dimensions
	 */
	public int[] maxTexture2D = new int[2];

	/**
	 * Maximum 2D texture dimensions if texture gather operations have to be performed
	 */
	public int[] maxTexture2DGather = new int[2];

	/**
	 * Maximum 2D layered texture dimensions
	 */
	public int[] maxTexture2DLayered = new int[3];

	/**
	 * Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
	 */
	public int[] maxTexture2DLinear = new int[3];

	/**
	 * Maximum 2D mipmapped texture dimensions
	 */
	public int[] maxTexture2DMipmap = new int[2];

	/**
	 * Maximum 3D texture dimensions
	 */
	public int[] maxTexture3D = new int[3];

	/**
	 * Contains the maximum alternate 3D texture dimensions
	 */
	public int[] maxTexture3DAlt = new int[3];

	/**
	 * Maximum Cubemap texture dimensions
	 */
	public int maxTextureCubemap;

	/**
	 * Maximum Cubemap layered texture dimensions
	 */
	public int[] maxTextureCubemapLayered = new int[2];

	/**
	 * The maximum sizes of each dimension of a block;
	 */
	public int[] maxThreadsDim = new int[3];

	/**
	 * The maximum number of threads per block;
	 */
	public int maxThreadsPerBlock;

	/**
	 * The number of maximum resident threads per multiprocessor.
	 */
	public int maxThreadsPerMultiProcessor;

	/**
	 * The memory bus width in bits
	 */
	public int memoryBusWidth;

	/**
	 * @deprecated The peak memory clock frequency in kilohertz.
	 */
	public int memoryClockRate;

	/**
	 * 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise
	 */
	public int memoryPoolsSupported;

	/**
	 * Bitmask of handle types supported with mempool-based IPC
	 */
	public int memoryPoolSupportedHandleTypes;

	/**
	 * The maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through
	 * cudaMallocPitch();
	 */
	public long memPitch;

	/**
	 * Minor compute capability
	 */
	public int minor;

	/**
	 * Unique identifier for a group of devices on the same multi-GPU board
	 */
	public int multiGpuBoardGroupID;

	/**
	 * Number of multiprocessors on device
	 */
	public int multiProcessorCount;

	/**
	 * An ASCII string identifying the device;
	 */
	public byte[] name = new byte[256];

	/**
	 * Device supports coherently accessing pageable memory without calling cudaHostRegister on it
	 */
	public int pageableMemoryAccess;

	/**
	 * Device accesses pageable memory via the host's page tables
	 */
	public int pageableMemoryAccessUsesHostPageTables;

	/**
	 * PCI bus ID of the device
	 */
	public int pciBusID;

	/**
	 * PCI device ID of the device
	 */
	public int pciDeviceID;

	/**
	 * PCI domain ID of the device
	 */
	public int pciDomainID;

	/**
	 * Device's maximum l2 persisting lines capacity setting in bytes
	 */
	public int persistingL2CacheMaxSize;

	/**
	 * The maximum number of 32-bit registers available to a thread block; this number is shared by all thread blocks
	 * simultaneously resident on a multiprocessor;
	 */
	public int regsPerBlock;

	/**
	 * 32-bit registers available per multiprocessor
	 */
	public int regsPerMultiprocessor;

	/**
	 * Reserved for future use
	 */
	public int reserved;

	/**
	 * Shared memory reserved by CUDA driver per block in bytes
	 */
	public long reservedSharedMemPerBlock;

	/**
	 * The maximum amount of shared memory available to a thread block in bytes; this amount is shared by all thread
	 * blocks simultaneously resident on a multiprocessor;
	 */
	public long sharedMemPerBlock;

	/**
	 * Per device maximum shared memory per block usable by special opt in
	 */
	public long sharedMemPerBlockOptin;

	/**
	 * Shared memory available per multiprocessor in bytes
	 */
	public long sharedMemPerMultiprocessor;

	/**
	 * @deprecated Ratio of single precision performance (in floating-point operations per second) to double precision
	 * performance
	 */
	public int singleToDoublePrecisionPerfRatio;

	/**
	 * 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise
	 */
	public int sparseCudaArraySupported;

	/**
	 * Is 1 if the device supports stream priorities, or 0 if it is not supported
	 */
	public int streamPrioritiesSupported;

	/**
	 * Alignment requirements for surfaces
	 */
	public long surfaceAlignment;

	/**
	 * 1 if device is a Tesla device using TCC driver, 0 otherwise
	 */
	public int tccDriver;

	/**
	 * The alignment requirement; texture base addresses that are aligned to textureAlignment bytes do not need an
	 * offset applied to texture fetches;
	 */
	public long textureAlignment;

	/**
	 * Pitch alignment requirement for texture references bound to pitched memory
	 */
	public long texturePitchAlignment;

	/**
	 * External timeline semaphore interop is supported on the device
	 */
	public int timelineSemaphoreInteropSupported;

	/**
	 * The total amount of constant memory available on the device in bytes;
	 */
	public long totalConstMem;

	/**
	 * The total amount of global memory available on the device in bytes;
	 */
	public long totalGlobalMem;

	/**
	 * 1 if the device shares a unified address space with the host and 0 otherwise.
	 */
	public int unifiedAddressing;

	/**
	 * Indicates device supports unified pointers
	 */
	public int unifiedFunctionPointers;

	/**
	 * The warp size in threads;
	 */
	public int warpSize;

	// Uninitialized CudaDeviceProp object
	public CudaDeviceProp() {
	}
}
