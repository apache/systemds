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
#include "Matrix.h"

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
__device__ void FULL_AGG(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t N, T VT, ReductionOp reduction_op, SpoofCellwiseOp spoof_op) {
	auto sdata = shared_memory_proxy<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	uint tid = threadIdx.x;
	uint i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	uint gridSize = blockDim.x * 2 * gridDim.x;

	T v = reduction_op.init();

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < N) {
//		printf("tid=%d i=%d N=%d, in->cols()=%d rix=%d\n", threadIdx.x, i, N, in->cols(), i/in->cols());
		v = reduction_op(v, spoof_op(*(in->vals(i)), i, i / in->cols(), i % in->cols()));

		if (i + blockDim.x < N)	{
			//__syncthreads();
			//printf("loop fetch i(%d)+blockDim.x(%d)=%d, in=%f\n",i, blockDim.x, i + blockDim.x, g_idata[i + blockDim.x]);
			v = reduction_op(v, spoof_op(*(in->vals(i+blockDim.x)), blockDim.x + i, (i+blockDim.x) / in->cols(), (i+blockDim.x) % in->cols()));
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
//		if(tid<12)
//			printf("bid=%d tid=%d reduction result: %3.1f\n", blockIdx.x, tid, sdata[tid]);
		
		if (blockDim.x >= 32) {
			smem[tid] = v = reduction_op(v, smem[tid + 16]);
		}
//		if(tid==0)
//			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		if (blockDim.x >= 16) {
			smem[tid] = v = reduction_op(v, smem[tid + 8]);
		}
//		if(tid==0)
//			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		if (blockDim.x >= 8) {
			smem[tid] = v = reduction_op(v, smem[tid + 4]);
		}
//		if(tid==0)
//			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		if (blockDim.x >= 4) {
			smem[tid] = v = reduction_op(v, smem[tid + 2]);
		}
//		if(tid==0)
//			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		if (blockDim.x >= 2) {
			smem[tid] = v = reduction_op(v, smem[tid + 1]);
		}
//		if(tid==0)
//			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
	}

	 // write result for this block to global mem
	 if (tid == 0) {
//	 	if(gridDim.x < 10)
//	 		printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
	 	out->val(0, blockIdx.x) = sdata[0];
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
//template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
//__device__ void ROW_AGG(
//		T *g_idata, ///< input data stored in device memory (of size rows*cols)
//		T *g_odata,  ///< output/temporary array store in device memory (of size
//		/// rows*cols)
//		uint rows,  ///< rows in input and temporary/output arrays
//		uint cols,  ///< columns in input and temporary/output arrays
//		T initialValue,  ///< initial value for the reduction variable
//		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
//		SpoofCellwiseOp spoof_op) ///< Operation to perform before assigning this
template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void ROW_AGG(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t N, T VT,  ReductionOp reduction_op,
					   SpoofCellwiseOp spoof_op) {
	auto sdata = shared_memory_proxy<T>();

	// one block per row
	if (blockIdx.x >= in->rows()) {
		return;
	}

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint32_t i = tid;
	uint block_offset = block * in->cols();

//	T v = initialValue;
	T v = reduction_op.init();
	while (i < in->cols()) {
		v = reduction_op(v, spoof_op(in->val(block_offset + i), i, i / in->cols(), i % in->cols()));
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
		out->val(block) = sdata[0];
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
__device__ void COL_AGG(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t N, T VT,  ReductionOp reduction_op,
						SpoofCellwiseOp spoof_op) {
//__device__ void COL_AGG(T *g_idata, ///< input data stored in device memory (of size rows*cols)
//		T *g_odata,  ///< output/temporary array store in device memory (of size rows*cols)
//		uint rows,  ///< rows in input and temporary/output arrays
//		uint cols,  ///< columns in input and temporary/output arrays
//		T initialValue,  ///< initial value for the reduction variable
//		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
//		SpoofCellwiseOp spoof_op) ///< Operation to perform before aggregation
//
//{
	uint global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (global_tid >= in->cols()) {
		return;
	}

	uint i = global_tid;
	uint grid_size = in->cols();
	T val = reduction_op.init();

	while (i < N) {
		val = reduction_op(val, spoof_op(in->val(i), i, i / in->cols(), i % in->cols()));
		i += grid_size;
	}
	out->val(global_tid) = val;
}

template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void NO_AGG(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t N, T VT,  ReductionOp reduction_op, SpoofCellwiseOp spoof_op)
{
	uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t first_idx = gtid * static_cast<uint32_t>(VT);
	uint32_t last_idx = min(first_idx + static_cast<uint32_t>(VT), N);
	#pragma unroll
	for(auto i = first_idx; i < last_idx; i++) {
		T a = in->hasData() ? in->vals(0)[i] : 0;
		T result = spoof_op(a, i, i / in->cols(), i % in->cols());
		out->vals(0)[i] = result;
		//if(i < 4)
		//	printf("tid=%d in=%4.3f res=%4.3f out=%4.3f r=%d\n", i, in->vals(0)[i], result, out->vals(0)[i], i/in->cols());
	}
}

template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void NO_AGG_SPARSE(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t N, T VT,  ReductionOp reduction_op, SpoofCellwiseOp spoof_op)
{
	const uint32_t& rix = blockIdx.x;
	uint32_t tid = threadIdx.x;
	uint32_t row_start = 0;
	uint32_t row_len = 0;
	if(in->hasData()) {
		row_start = in->pos(rix);
		row_len = in->row_len(rix);
	}
	else {
		row_start = rix * in->cols();
		row_len = in->cols();
	}
	while(tid < row_len) {
		uint32_t idx = row_start + tid;
		if(in->hasData()) {
			uint32_t *aix = in->col_idxs(rix);
			uint32_t cix = aix[tid];
			T result = spoof_op(in->val(idx), idx, rix, cix);
			out->set(idx, cix, result);
		}
		else {
			uint32_t cix = tid;
			T result = spoof_op(0, idx, rix, cix);
			out->set(idx, cix, result);
		}
		tid+=blockDim.x;
	}
}

#endif // REDUCTION_CUH
