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

import static jcuda.runtime.JCuda.cudaMemGetInfo;

import jcuda.CudaException;
import jcuda.Pointer;
import static jcuda.runtime.cudaError.cudaSuccess;

import org.tugraz.sysds.api.DMLScript;

import static jcuda.runtime.JCuda.cudaMalloc;
import jcuda.runtime.cudaError;
import static jcuda.runtime.JCuda.cudaFree;

public class CudaMemoryAllocator implements GPUMemoryAllocator {
	
	/**
	 * Allocate memory on the device. 
	 * 
	 * @param devPtr Pointer to allocated device memory
	 * @param size size in bytes
	 * @throws jcuda.CudaException if unable to allocate
	 */
	public void allocate(Pointer devPtr, long size) throws CudaException {
		int status = cudaMalloc(devPtr, size);
		if(status != cudaSuccess) {
			throw new jcuda.CudaException("cudaMalloc failed:" + cudaError.stringFor(status));
		}
	}

	/**
	 * Frees memory on the device
	 * 
	 * @param devPtr Device pointer to memory to free
	 * @throws jcuda.CudaException if error occurs
	 */
	public void free(Pointer devPtr) throws CudaException {
		int status = cudaFree(devPtr);
		if(status != cudaSuccess) {
			throw new jcuda.CudaException("cudaFree failed:" + cudaError.stringFor(status));
		}
	}

	/**
	 * Check if there is enough memory to allocate a pointer of given size 
	 * 
	 * @param size size in bytes
	 * @return true if there is enough available memory to allocate a pointer of the given size 
	 */
	public boolean canAllocate(long size) {
		return size <= getAvailableMemory();
	}
	
	/**
	 * Gets the available memory on GPU that SystemDS can use.
	 *
	 * @return the available memory in bytes
	 */
	public long getAvailableMemory() {
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		return (long) (free[0] * DMLScript.GPU_MEMORY_UTILIZATION_FACTOR);
	}

}
