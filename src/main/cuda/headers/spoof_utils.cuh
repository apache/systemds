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
#ifndef SPOOF_UTILS_CUH
#define SPOOF_UTILS_CUH

#include <math_constants.h>
#include "Matrix.h"
#include "operators.cuh"
// #include "intellisense_cuda_intrinsics.h"

using uint32_t = unsigned int;

__constant__ double DOUBLE_EPS = 1.11022E-16; // 2 ^ -53
__constant__ double FLOAT_EPS = 1.49012E-08; // 2 ^ -26
__constant__ double EPSILON = 1E-11; // margin for comparisons ToDo: make consistent use of it

__device__ long long toInt64(double a) {
	return (signbit(a) == 0 ? 1.0 : -1.0) * abs(floor(a + DOUBLE_EPS));
}

__device__ int toInt32(float a) {
	return (signbit(a) == 0 ? 1.0 : -1.0) * abs(floor(a + FLOAT_EPS));
}

template<typename T>
__device__ T getValue(T* data, int rowIndex) {
	return data[rowIndex];
}

template<typename T>
__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {
	return data[rowIndex * n + colIndex];
}

template<typename T>
__device__ T intDiv(T a, T b);

template<>
__device__ double intDiv(double a, double b) {
	double ret = a / b;
	return (isnan(ret) || isinf(ret)) ? ret : toInt64(ret);
}

template<>
__device__ float intDiv(float a, float b) {
	float ret = a / b;
	return (isnan(ret) || isinf(ret)) ? ret : toInt32(ret);
}

template<typename T>
__device__ T modulus(T a, T b);

template<>
__device__ double modulus(double a, double b) {
	if (fabs(b) < DOUBLE_EPS)
		return CUDART_NAN;
	return a - intDiv(a, b) * b;
}

template<>
__device__ float modulus(float a, float b) {
	if (fabs(b) < FLOAT_EPS)
		return CUDART_NAN_F;
	return a - intDiv(a, b) * b;
}

template<typename T>
__device__ T bwAnd(T a, T b);

// ToDo: does not work with long long
template<>
__device__ double bwAnd(double a, double b) {
	return toInt64(a) & toInt64(b);
}

template<>
__device__ float bwAnd(float a, float b) {
	return toInt32(a) & toInt32(b);
}

template<typename T, typename AggOp, typename LoadOp>
__device__ T BLOCK_ROW_AGG(T *a, T *b, uint32_t len, AggOp agg_op, LoadOp load_op) {
	auto sdata = shared_memory_proxy<T>();
	uint tid = threadIdx.x;

	// Initalize shared mem and leave if tid > row length. 
	if(tid >= len) { return sdata[tid] = AggOp::init();; }

	__syncthreads();
	
			// if(blockIdx.x == 0 && threadIdx.x == 0)
		 //  printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
	uint i = tid;
	T v = AggOp::init();
			// if(blockIdx.x == 0 && threadIdx.x == 0)
		 //  printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
	while (i < len) {
		v = agg_op(v, load_op(a[i], b[i]));
		i += blockDim.x;
	}

		// if(blockIdx.x == 0 && threadIdx.x == 0)
		//   printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
	
	// each thread puts its local sum into shared memory
	sdata[tid] = v;
	// if(blockIdx.x==0)
		// printf("tid=%d v=%f, len=%d\n", tid, v, len);
	__syncthreads();

			// if(blockIdx.x == 0 && threadIdx.x == 0)
		 //  printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
	
	// do reduction in shared mem
	if (blockDim.x >= 1024) {
		if (tid < 512 && (tid+512) < len) {
				// if(blockIdx.x == 0 && threadIdx.x == 0)
		  // printf("tid=%d sdata[tid + 512]=%f\n", tid, sdata[tid+512]);
			sdata[tid] = v = agg_op(v, sdata[tid + 512]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256 && (tid+256) < len) {
				// if(blockIdx.x == 0 && threadIdx.x == 0)
		  // printf("tid=%d sdata[tid + 256]=%f\n", tid, sdata[tid+256]);
			sdata[tid] = v = agg_op(v, sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128 && (tid+128) < len) {
				// if(blockIdx.x == 0 && threadIdx.x == 0)
		  // printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
			sdata[tid] = v = agg_op(v, sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64 && (tid+64) < len) {
				// if(blockIdx.x == 0 && threadIdx.x == 0)
		  // printf("tid=%d sdata[tid + 64]=%f\n", tid, sdata[tid+64]);
			sdata[tid] = v = agg_op(v, sdata[tid + 64]);
		}
		__syncthreads();
	}
 
	if (tid < 32) {
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;
		if (blockDim.x >= 64 && (tid+32) < len) {
			smem[tid] = v = agg_op(v, smem[tid + 32]);
		}
		// if(blockIdx.x==0)
		  // printf("tid=%d smem[0]=%f\n", tid, smem[0]);
		if (blockDim.x >= 32 && (tid+16) < len) {
			smem[tid] = v = agg_op(v, smem[tid + 16]);
		}
		// if(blockIdx.x==0)
		  // printf("tid=%d smem[0]=%f\n", tid, smem[0]);
		if (blockDim.x >= 16 && (tid+8) < len) {
			smem[tid] = v = agg_op(v, smem[tid + 8]);
		}
		// if(blockIdx.x==0)
		  // printf("tid=%d smem[0]=%f\n", tid, smem[0]);
		if (blockDim.x >= 8 && (tid+4) < len) {
			smem[tid] = v = agg_op(v, smem[tid + 4]);
		}
		// if(blockIdx.x==0 && threadIdx.x ==0)
		  // printf("tid=%d smem[tid + 4]=%f\n", tid, smem[tid+4]);
		if (blockDim.x >= 4 && (tid+2) < len) {
			smem[tid] = v = agg_op(v, smem[tid + 2]);
		}
		// if(blockIdx.x==0 && threadIdx.x ==0)
		  // printf("tid=%d smem[0]=%f\n", tid, smem[0]);
		if (blockDim.x >= 2 && (tid+1) < len) {
		// if (blockDim.x >= 2) {
			smem[tid] = v = agg_op(v, smem[tid + 1]);
		}
		// if(blockIdx.x==0 && threadIdx.x ==0)
		  // printf("tid=%d smem[0]=%f\n", tid, smem[0]);
	}

	__syncthreads();
	return sdata[0];
}

template<typename T>
__device__ T dotProduct(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len) {
	SumOp<T> agg_op;
	ProductOp<T> load_op;
	T ret =  BLOCK_ROW_AGG(&a[ai], &b[bi], len, agg_op, load_op);
//	if(blockIdx.x < 4 && threadIdx.x == 0)
//		printf("bid=%d, ai=%d, dot=%f\n", blockIdx.x, ai, ret);
	return ret;
}

template<typename T>
__device__ T vectSum(T* a, int ai, int len) {
	SumOp<T> agg_op;
	IdentityOp<T> load_op;
	return BLOCK_ROW_AGG(&a[ai], &a[ai], len, agg_op, load_op);
}


template<typename T>
__device__ T vectSum(T* a, uint32_t ai, uint32_t len) {
	SumOp<T> agg_op;
	IdentityOp<T> load_op;
	return BLOCK_ROW_AGG(&a[ai], &a[ai], len, agg_op, load_op);
}

template<typename T>
__device__ T vectMin(T* a, int ai, int len) {
	MinOp<T> agg_op;
	IdentityOp<T> load_op;
	return BLOCK_ROW_AGG(&a[ai], &a[ai], len, agg_op, load_op);
}

template<typename T>
__device__ T vectMax(T* a, int ai, int len) {
	MaxOp<T> agg_op;
	IdentityOp<T> load_op;
	return BLOCK_ROW_AGG(&a[ai], &a[ai], len, agg_op, load_op);
}

template<typename T, typename Op>
__device__ void vectAdd_atomic(T* a, T b, T* c, uint32_t ai, uint32_t ci, uint32_t len, Op op) {
	uint tid = threadIdx.x;
	if(tid >= len)
		return;
	uint i = tid;

	while (i < len) {
//		if(blockIdx.x == 1 && threadIdx.x < 2)
//			printf("vectAdd_atomic: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, b=%f, c[%d]=%f, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai,
//		  			ci, len, b, ci * len + threadIdx.x, op(a[ai + i], b), ai, a[ai + i]);
		
		atomicAdd(&(c[ci * len + i]), op(a[ai + i], b));
		i += blockDim.x;
	}
}

template<typename T>
__device__ void vectMultAdd(T* a, T b, T* c, uint32_t ai, uint32_t ci, uint32_t len) {
	ProductOp<T> op;
	vectAdd_atomic<T, ProductOp<T>>(a, b, c, ai, ci, len, op);
}

template<typename T>
__device__ void vectDivAdd(T* a, T b, T* c, int ai, int ci, int len) {
	DivOp<T> op;
	vectAdd_atomic<T, DivOp<T>>(a, b, c, ai, ci, len, op);
}

template<typename T>
__device__ int vectCbindWrite(T* a, T b, T* c, uint32_t ai, uint32_t len) {
	if(threadIdx.x < len) {
//		 if(blockIdx.x==1 && threadIdx.x ==0)
//		 	printf("vecCbindWrite: bid=%d, tid=%d, ai=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, len, ai * len + threadIdx.x, a[ai * len + threadIdx.x]);
        c[blockIdx.x * (len+1) + threadIdx.x] = a[ai + threadIdx.x];
	}
	if(threadIdx.x == len) {
//		printf("---> block %d thread %d, b=%f,, len=%d, a[%d]=%f\n",blockIdx.x, threadIdx.x, b, len, ai, a[ai]);
        c[blockIdx.x * (len+1) + threadIdx.x] = b;
	}
	return len+1;
}

template<typename T>
__device__ int vectCbindWrite(T a, T b, T* c) {
	if (threadIdx.x == 0) {
		c[blockIdx.x * 2] = a;
		c[blockIdx.x * 2 + 1] = b;
	}
	return 2;
}

// vect-vect
template<typename T, typename OP>
__device__ int vectWrite_(T* a, T* b, T* c, int ai, int ci, int len) {
	uint i = threadIdx.x;
	if(blockIdx.x == 1 && threadIdx.x < 2)
		printf("vecWrite_vv: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, c[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ci * len + threadIdx.x, OP::exec(a[ai + i], b[ai + i]));
	
	while (i < len) {
		c[ci + i] = OP::exec(a[ai + i], b[ai + i]);
		i += blockDim.x;
	}
	return len;
}

// vect-scalar
template<typename T, typename OP>
__device__ int vectWrite_(T* a, T b, T* c, int ai, int ci, int len) {
    uint i = threadIdx.x;
	if(blockIdx.x < 2 && threadIdx.x < 1)
		printf("vecWrite_vs: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, c[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ci * len + threadIdx.x, OP::exec(a[ai + i], b));
    while (i < len) {
		c[ci + i] = OP::exec(a[ai + i], b);
        i += blockDim.x;
    }
    return len;
}

// scalar-vector
template<typename T, typename OP>
__device__ int vectWrite_(T a, T* b, T* c, int ai, int ci, int len) {
	uint i = threadIdx.x;
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vecWrite_sv: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, c[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ci * len + threadIdx.x, OP::exec(a, b[ai + i]););
	
	while (i < len) {
		c[ci + i] = OP::exec(a, b[ai + i]);
		i += blockDim.x;
	}
	return len;
}

template<typename T>
__device__ void vectWrite(T* a, T* c, int ai, int ci, int len) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vecWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ai + threadIdx.x, a[ai + threadIdx.x]);

	vectWrite_<T, IdentityOp<T>>(a, SumNeutralElement<T>::get(), c, ai, ci, len);
}

template<typename T>
__device__ void vectWrite(T* a, T* c, int ci, int len) {
    vectWrite(a, c, 0, ci, len);
}

template<typename T>
int vectLessequalWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, LessEqualOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectGreaterequalWrite(T* a, T b, T* c, int ai, int len) {
    return vectWrite_<T, GreaterEqualOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectDivWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, DivOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectDivWrite(T a, T* b, T* c, int ai, int len) {
	return vectWrite_<T, DivOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectMultWrite(T* a, T b, T* c, int ai, int len) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vectMultWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d\n", blockIdx.x, threadIdx.x, ai, 0, len);
	
	return vectWrite_<T, ProductOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectMultWrite(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t len) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vectMultWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d\n", blockIdx.x, threadIdx.x, ai, 0, len);

	return vectWrite_<T, ProductOp<T>>(a, b, c, ai, bi, len);
}

template<typename T>
int vectPlusWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, SumOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectMinusWrite(T* a, T* b, T* c, int ai, int ci, int len) {
	return vectWrite_<T, MinusOp<T>>(a, b, c, ai, ci, len);
}

template<typename T>
int vectMinusWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, MinusOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectMinusWrite(T a, T* b, T* c, int ai, int len) {
	return vectWrite_<T, MinusOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectSignWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, SignOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectAbsWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, AbsOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectExpWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, ExpOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectRoundWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, RoundOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectFloorWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, FloorOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectMinWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, MinOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectMaxWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, MaxOp<T>>(a, b, c, ai, 0, len);
}

template<typename T, typename OP>
__device__ int vectAdd_atomic_(T* a, T b, T* c, int ai, int ci, int len) {
    uint i = threadIdx.x;
    while (i < len) {
        atomicAdd(&(c[ci + i]), OP::exec(a[ai + i], b));
        i += blockDim.x;
    }
    return len;
}

template<typename T>
T* vectGreaterAdd(T* a, T b, T* c, int ai, int len) {
	vectAdd_atomic_<T, GreaterOp<T>>(a, b, c, ai, 0, len);
	return c;
}

template<typename T>
void vectGreaterAdd(T* a, T b, T* c, int ai, int ci, int len) {
    vectAdd_atomic_<T, GreaterOp<T>>(a, b, c, ai, ci, len);
}

template<typename T>
int vectMinAdd(T* a, T b, T* c, int ai, int ci, int len) {
	return vectAdd_atomic_<T, MinOp<T>>(a, b, c, ai, ci, len);
}

template<typename T>
int vectMaxAdd(T* a, T b, T* c, int ai, int ci, int len) {
	return vectAdd_atomic_<T, MaxOp<T>>(a, b, c, ai, ci, len);
}
#endif // SPOOF_UTILS_CUH
