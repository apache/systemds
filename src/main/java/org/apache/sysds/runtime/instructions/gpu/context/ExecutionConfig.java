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

package org.apache.sysds.runtime.instructions.gpu.context;

import java.util.HashMap;

import org.apache.sysds.runtime.DMLRuntimeException;

import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUstream;

/**
 * Java Wrapper to specify CUDA execution configuration for launching custom kernels
 */
public class ExecutionConfig {
	public int gridDimX;
	public int gridDimY = 1;
	public int gridDimZ = 1;
	public int blockDimX;
	public int blockDimY = 1;
	public int blockDimZ = 1;
	public int sharedMemBytes = 0;
	public CUstream stream = null;

	private static HashMap<Integer, Integer> maxBlockDimForDevice = new HashMap<>();

	/**
	 * Convenience constructor for setting the number of blocks, number of threads and the
	 * shared memory size
	 *
	 * @param gridDimX       Number of blocks on the horizontal axis of the grid (for CUDA Kernel)
	 * @param blockDimX      Number of threads on the horizontal axis of a block (for CUDA Kernel)
	 * @param sharedMemBytes Amount of Shared memory (for CUDA Kernel)
	 */
	public ExecutionConfig(int gridDimX, int blockDimX, int sharedMemBytes) {
		this.gridDimX = gridDimX;
		this.blockDimX = blockDimX;
		this.sharedMemBytes = sharedMemBytes;
	}

	/**
	 * Convenience constructor for setting the number of blocks, number of threads and the
	 * shared memory size
	 *
	 * @param gridDimX       Number of blocks on the horizontal axis of the grid (for CUDA Kernel)
	 * @param blockDimX      Number of threads on the horizontal axis of a block (for CUDA Kernel)
	 */
	public ExecutionConfig(int gridDimX, int blockDimX) {
		this.gridDimX = gridDimX;
		this.blockDimX = blockDimX;
	}

	/**
	 * Convenience constructor for setting the number of blocks, number of threads and the
	 * shared memory size
	 *
	 * @param gridDimX       Number of blocks on the horizontal axis of the grid (for CUDA Kernel)
	 * @param gridDimY       Number of blocks on the vertical axis of the grid (for CUDA Kernel)
	 * @param blockDimX      Number of threads on the horizontal axis of a block (for CUDA Kernel)
	 * @param blockDimY      Number of threads on the vertical axis of a block (for CUDA Kernel)=
	 */
	public ExecutionConfig(int gridDimX, int gridDimY, int blockDimX, int blockDimY) {
		this.gridDimX = gridDimX;
		this.gridDimY = gridDimY;
		this.blockDimX = blockDimX;
		this.blockDimY = blockDimY;
	}

	/**
	 * Convenience constructor for setting the number of blocks, number of threads and the
	 * shared memory size
	 *
	 * @param gridDimX       Number of blocks on the horizontal axis of the grid (for CUDA Kernel)
	 * @param gridDimY       Number of blocks on the vertical axis of the grid (for CUDA Kernel)
	 * @param blockDimX      Number of threads on the horizontal axis of a block (for CUDA Kernel)
	 * @param blockDimY      Number of threads on the vertical axis of a block (for CUDA Kernel)
	 * @param sharedMemBytes Amount of Shared memory (for CUDA Kernel)
	 */
	public ExecutionConfig(int gridDimX, int gridDimY, int blockDimX, int blockDimY, int sharedMemBytes) {
		this.gridDimX = gridDimX;
		this.gridDimY = gridDimY;
		this.blockDimX = blockDimX;
		this.blockDimY = blockDimY;
		this.sharedMemBytes = sharedMemBytes;
	}

	/**
	 * Use this for simple vector operations and use following in the kernel
	 * <code>
	 * int index = blockIdx.x * blockDim.x + threadIdx.x
	 * </code>
	 * <p>
	 * This tries to schedule as minimum grids as possible.
	 *
	 * @param numCells number of cells
	 * @return execution configuration
	 */
	public static ExecutionConfig getConfigForSimpleVectorOperations(int numCells) {
		if(numCells == 0)
			throw new DMLRuntimeException("Attempting to invoke a kernel with 0 threads");
		int deviceNumber = 0;
		int blockDimX = getMaxBlockDim(deviceNumber);
		int gridDimX = (int) Math.ceil((double) numCells / blockDimX);
		return new ExecutionConfig(gridDimX, blockDimX);
	}

	/**
	 * Use this for simple vector operations and use following in the kernel
	 * <code>
	 * int index = blockIdx.x * blockDim.x + threadIdx.x
	 * </code>
	 * <p>
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @return execution configuration
	 */
	public static ExecutionConfig getConfigForSimpleMatrixOperations(int rlen, int clen) {
		return getConfigForSimpleVectorOperations(rlen * clen);
	}

	/**
	 * Get the CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X of the given device
	 *
	 * @param deviceNumber device number of the given device
	 * @return The maximum block dimension, in x-direction
	 */
	private static int getMaxBlockDim(int deviceNumber) {
		//    	return 32;
		// TODO: Use JCudaDriver.cuOccupancyMaxPotentialBlockSize to chose the block size that maximizes occupancy
		Integer ret = maxBlockDimForDevice.get(deviceNumber);
		if (ret == null) {
			CUdevice device = new CUdevice();
			JCudaKernels.checkResult(jcuda.driver.JCudaDriver.cuDeviceGet(device, deviceNumber));
			int maxBlockDimX[] = { 0 };
			jcuda.driver.JCudaDriver
					.cuDeviceGetAttribute(maxBlockDimX, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
			maxBlockDimForDevice.put(deviceNumber, maxBlockDimX[0]);
			return maxBlockDimX[0];
		}
		return ret;
	}

	@Override
	public String toString() {
		return "ExecutionConfig{" + "gridDimX=" + gridDimX + ", gridDimY=" + gridDimY + ", gridDimZ=" + gridDimZ
				+ ", blockDimX=" + blockDimX + ", blockDimY=" + blockDimY + ", blockDimZ=" + blockDimZ
				+ ", sharedMemBytes=" + sharedMemBytes + '}';
	}
}
