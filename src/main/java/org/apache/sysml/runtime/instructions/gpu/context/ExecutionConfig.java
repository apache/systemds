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

import org.apache.sysml.runtime.DMLRuntimeException;

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
	
//	private static HashMap<Integer, Integer> maxBlockDimXForDevice = new HashMap<Integer, Integer>();
//	private static HashMap<Integer, Integer> maxBlockDimYForDevice = new HashMap<Integer, Integer>();
	
	/**
	 * Use this for simple vector operations and use following in the kernel 
	 * <code> 
	 * int index = blockIdx.x * blockDim.x + threadIdx.x 
	 * </code>
	 * 
	 * This tries to schedule as minimum grids as possible.
	 * 
	 * @param numCells
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ExecutionConfig getConfigForSimpleVectorOperations(int numCells) throws DMLRuntimeException {
		int deviceNumber = 0;
		int blockDimX = getMaxBlockDimX(deviceNumber);
		int gridDimX = (int)Math.ceil((double)numCells / blockDimX);
		return new ExecutionConfig(gridDimX, blockDimX);
	}
	
	/**
	 * Use this for simple matrix operations and use following in the kernel 
	 * <code> 
	 * int ix = blockIdx.x * blockDim.x + threadIdx.x;
	 * int iy = blockIdx.y * blockDim.y + threadIdx.y;
	 * </code>
	 * 
	 * This tries to schedule as minimum grids as possible.
	 * 
	 * @param numCells
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ExecutionConfig getConfigForSimpleMatrixOperations(int rlen, int clen) throws DMLRuntimeException {
		int deviceNumber = 0;
		int blockDimX = (int) Math.min(getMaxBlockDimX(deviceNumber), rlen);
		int gridDimX = (int)Math.ceil((double)rlen / blockDimX);
		int blockDimY = (int)Math.min(getMaxBlockDimY(deviceNumber), clen);
		int gridDimY = (int)Math.ceil((double)clen / blockDimY);
		return new ExecutionConfig(gridDimX, gridDimY, blockDimX, blockDimY);
	}
	
	public ExecutionConfig(int gridDimX, int blockDimX) {
		this.gridDimX = gridDimX;
		this.blockDimX = blockDimX;
	}
	
	public ExecutionConfig(int gridDimX, int gridDimY, int blockDimX, int blockDimY) {
		this.gridDimX = gridDimX;
		this.gridDimY = gridDimY;
		this.blockDimX = blockDimX;
		this.blockDimY = blockDimY;
	}
	
	
	/**
     * Get the CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X of the given device
     * 
     * @return The maximum block dimension, in x-direction
	 * @throws DMLRuntimeException 
     */
    private static int getMaxBlockDimX(int deviceNumber) throws DMLRuntimeException {
    	return 32;
    	// TODO: Use JCudaDriver.cuOccupancyMaxPotentialBlockSize to chose the block size that maximizes occupancy
//    	Integer ret = maxBlockDimXForDevice.get(deviceNumber);
//    	if(ret == null) {
//    		CUdevice device = new CUdevice();
//            JCudaKernels.checkResult(cuDeviceGet(device, deviceNumber));
//            int maxBlockDimX[] =  {0};
//            cuDeviceGetAttribute(maxBlockDimX, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
//            maxBlockDimXForDevice.put(deviceNumber, maxBlockDimX[0]);
//            return maxBlockDimX[0];
//    	}
//        return ret;
    }
    
    /**
     * Get the CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y of the given device
     * 
     * @return The maximum block dimension, in y-direction
	 * @throws DMLRuntimeException 
     */
    private static int getMaxBlockDimY(int deviceNumber) throws DMLRuntimeException {
    	return 32;
    	// TODO: Use JCudaDriver.cuOccupancyMaxPotentialBlockSize to chose the block size that maximizes occupancy
//    	Integer ret = maxBlockDimYForDevice.get(deviceNumber);
//    	if(ret == null) {
//    		CUdevice device = new CUdevice();
//            JCudaKernels.checkResult(cuDeviceGet(device, deviceNumber));
//            int maxBlockDimY[] =  {0};
//            cuDeviceGetAttribute(maxBlockDimY, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
//            maxBlockDimYForDevice.put(deviceNumber, maxBlockDimY[0]);
//            return maxBlockDimY[0];
//    	}
//        return ret;
    }
}
