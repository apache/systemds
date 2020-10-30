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

#pragma once
#ifndef REDUCTION_CUH
#define REDUCTION_CUH

using uint = unsigned int;
#include <cuda_runtime.h>

#include "utils.cuh"

/**
 * Does a reduce operation over all elements of the array.
 * This method has been adapted from the Reduction sample in the NVIDIA CUDA
 * Samples (v8.0)
 * and the Reduction example available through jcuda.org
 * When invoked initially, all blocks partly compute the reduction operation
 * over the entire array
 * and writes it to the output/temporary array. A second invokation needs to
 * happen to get the
 * reduced value.
 * The number of threads, blocks and amount of shared memory is calculated in a
 * specific way.
 * Please refer to the NVIDIA CUDA Sample or the SystemDS code that invokes this
 * method to see
 * how its done.
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 *
 * @param n		size of the input and temporary/output arrays		
 * @param ReductionOp		Type of the functor object that implements the
 *		reduction operation
 * @param SpoofCellwiseOp		initial value for the reduction variable
 */
template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void FULL_AGG(
		T *g_idata, ///< input data stored in device memory (of size n)
		T *g_odata, ///< output/temporary array stored in device memory (of size n)
		uint m,
		uint n,
		T initialValue, 
		ReductionOp reduction_op, 
	    SpoofCellwiseOp spoof_op)
{
	auto sdata = shared_memory_proxy<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	uint gridSize = blockDim.x * 2 * gridDim.x;
	uint N = m * n;
	T v = initialValue;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < N) {
		v = reduction_op(v, spoof_op(g_idata[i], i));

		if (i + blockDim.x < N)	
		{
			//__syncthreads();
			//printf("loop fetch i(%d)+blockDim.x(%d)=%d, in=%f\n",i, blockDim.x, i + blockDim.x, g_idata[i + blockDim.x]);
			v = reduction_op(v, spoof_op(g_idata[i + blockDim.x], blockDim.x + i));
		}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = v;
	__syncthreads();

	// do reduction in shared mem
	if (blockDim.x >= 1024) {
		if (tid < 512) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 512]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 64]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;
		if (blockDim.x >= 64) {
			smem[tid] = v = reduction_op(v, smem[tid + 32]);
		}
		if (blockDim.x >= 32) {
			smem[tid] = v = reduction_op(v, smem[tid + 16]);
		}
		if (blockDim.x >= 16) {
			smem[tid] = v = reduction_op(v, smem[tid + 8]);
		}
		if (blockDim.x >= 8) {
			smem[tid] = v = reduction_op(v, smem[tid + 4]);
		}
		if (blockDim.x >= 4) {
			smem[tid] = v = reduction_op(v, smem[tid + 2]);
		}
		if (blockDim.x >= 2) {
			smem[tid] = v = reduction_op(v, smem[tid + 1]);
		}
	}

	// write result for this block to global mem
	if (tid == 0) {
		if(gridDim.x < 10)
			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		g_odata[blockIdx.x] = sdata[0];
	}
}

/**
 * Does a reduce (sum) over each row of the array.
 * This kernel must be launched with as many blocks as there are rows.
 * The intuition for this kernel is that each block does a reduction over a
 * single row.
 * The maximum number of blocks that can launched (as of compute capability 3.0)
 * is 2^31 - 1
 * This works out fine for SystemDS, since the maximum elements in a Java array
 * can be 2^31 - c (some small constant)
 * If the matrix is "fat" and "short", i.e. there are small number of rows and a
 * large number of columns,
 * there could be under-utilization of the hardware.
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify
 * the value before writing it to its final location in global memory for each
 * row
 */
template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void ROW_AGG(
		T *g_idata, ///< input data stored in device memory (of size rows*cols)
		T *g_odata,  ///< output/temporary array store in device memory (of size
		/// rows*cols)
		uint rows,  ///< rows in input and temporary/output arrays
		uint cols,  ///< columns in input and temporary/output arrays
		T initialValue,  ///< initial value for the reduction variable
		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
		SpoofCellwiseOp spoof_op) ///< Operation to perform before assigning this
{
	auto sdata = shared_memory_proxy<T>();

	// one block per row
	if (blockIdx.x >= rows) {
		return;
	}

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint i = tid;
	uint block_offset = block * cols;

	T v = initialValue;
	while (i < cols) {
		v = reduction_op(v, spoof_op(g_idata[block_offset + i], i));
		i += blockDim.x;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = v;
	__syncthreads();

	// do reduction in shared mem
	if (blockDim.x >= 1024) {
		if (tid < 512) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 512]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64) {
			sdata[tid] = v = reduction_op(v, sdata[tid + 64]);
		}
		__syncthreads();
	}

	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;
		if (blockDim.x >= 64) {
			smem[tid] = v = reduction_op(v, smem[tid + 32]);
		}
		if (blockDim.x >= 32) {
			smem[tid] = v = reduction_op(v, smem[tid + 16]);
		}
		if (blockDim.x >= 16) {
			smem[tid] = v = reduction_op(v, smem[tid + 8]);
		}
		if (blockDim.x >= 8) {
			smem[tid] = v = reduction_op(v, smem[tid + 4]);
		}
		if (blockDim.x >= 4) {
			smem[tid] = v = reduction_op(v, smem[tid + 2]);
		}
		if (blockDim.x >= 2) {
			smem[tid] = v = reduction_op(v, smem[tid + 1]);
		}
	}

	// write result for this block to global mem, modify it with assignment op
	if (tid == 0)
		g_odata[block] = sdata[0];
}

/**
 * Does a column wise reduction.
 * The intuition is that there are as many global threads as there are columns
 * Each global thread is responsible for a single element in the output vector
 * This of course leads to a under-utilization of the GPU resources.
 * For cases, where the number of columns is small, there can be unused SMs
 *
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify
 * the value before writing it to its final location in global memory for each
 * column
 */
template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void COL_AGG(T *g_idata, ///< input data stored in device memory (of size rows*cols)
		T *g_odata,  ///< output/temporary array store in device memory (of size rows*cols)
		uint rows,  ///< rows in input and temporary/output arrays
		uint cols,  ///< columns in input and temporary/output arrays
		T initialValue,  ///< initial value for the reduction variable
		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
		SpoofCellwiseOp spoof_op) ///< Operation to perform before aggregation
		
{
	uint global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_tid >= cols) {
		return;
	}

	uint i = global_tid;
	uint grid_size = cols;
	T val = initialValue;

	while (i < rows * cols) {
		val = reduction_op(val, spoof_op(g_idata[i], i));
		i += grid_size;
	}
	g_odata[global_tid] = val;
}

template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void NO_AGG(T* g_idata, T* g_odata,  uint rows, uint cols,
	T VT,  ReductionOp reduction_op, SpoofCellwiseOp spoof_op) 
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int first_idx = tid * static_cast<int>(VT);
	int last_idx = min(first_idx + static_cast<int>(VT), spoof_op.m * spoof_op.n);
	#pragma unroll
	for(int i = first_idx; i < last_idx; i++) {
		g_odata[i] = spoof_op(g_idata[i], i);
	}
}

#endif // REDUCTION_CUH
