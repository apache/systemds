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
#include "vector_write.cuh"
#include "vector_add.cuh"
#include "operators.cuh"

// #include "intellisense_cuda_intrinsics.h"

using uint32_t = unsigned int;

__device__ bool debug_row() { return blockIdx.x == 4; };
__device__ bool debug_thread() { return threadIdx.x == 0; }

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
//	if(blockIdx.x == 0 && threadIdx.x == 0)
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
__device__ Vector<T>& vectCbindWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {

	Vector<T>& c = fop->getTempStorage();

	if(threadIdx.x < len) {
//		 if(blockIdx.x==1 && threadIdx.x ==0)
//		 	printf("vecCbindWrite: bid=%d, tid=%d, ai=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, len, ai * len + threadIdx.x, a[ai * len + threadIdx.x]);
        c[blockIdx.x * (len+1) + threadIdx.x] = a[ai + threadIdx.x];
	}
	if(threadIdx.x == len) {
//		printf("---> block %d thread %d, b=%f,, len=%d, a[%d]=%f\n",blockIdx.x, threadIdx.x, b, len, ai, a[ai]);
        c[blockIdx.x * (len+1) + threadIdx.x] = b;
	}
	return c;
}

template<typename T>
__device__ Vector<T>& vectCbindWrite(T a, T b, SpoofOp<T>* fop) {
	Vector<T>& c = fop->getTempStorage();
	if (threadIdx.x == 0) {
		c[blockIdx.x * 2] = a;
		c[blockIdx.x * 2 + 1] = b;
	}
	return c;
}

template<typename T>
__device__ void vectWrite(T* a, T* c, uint32_t ai, uint32_t ci, uint32_t len) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vecWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ai + threadIdx.x, a[ai + threadIdx.x]);

	vectWriteBinary<T, IdentityOp<T>>(a, SumNeutralElement<T>::get(), c, ai, ci, len);
}

template<typename T>
__device__ void vectWrite(T* a, T* c, uint32_t ci, uint32_t len) {
    vectWrite(a, c, 0, ci, len);
}

template<typename T>
int vectDivWrite(T* a, T b, T* c, uint32_t ai, uint32_t len) {
	return vectWrite_<T, DivOp<T>>(a, b, c, ai, 0, len);
}

template<typename T>
int vectDivWrite(T a, T* b, T* c, uint32_t ai, uint32_t len) {
	return vectWrite_<T, DivOp<T>>(a, b, c, ai, 0, len);
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
int vectRoundWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, RoundOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectFloorWrite(T* a, T* c, int ai, int len) {
	return vectWrite_<T, FloorOp<T>>(a, 0.0, c, ai, 0, len);
}

template<typename T>
int vectMinWrite(T* a, T b, T* c, uint32_t ai, uint32_t len) {
	return vectWrite_<T, MinOp<T>>(a, b, c, ai, 0u, len);
}

template<typename T>
int vectMinWrite(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t len) {
	return vectWrite_<T, MinOp<T>>(a, b, c, ai, bi, len);
}

template<typename T>
int vectMaxWrite(T* a, T b, T* c, int ai, int len) {
	return vectWrite_<T, MaxOp<T>>(a, b, c, ai, 0, len);
}

template<typename T, typename OP>
__device__ int vectAdd_atomic_(T* a, T b, T* c, uint32_t ai, uint32_t ci, uint32_t len) {
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

template<typename T>
int vectPlusAdd(T* a, T b, T* c, int ai, int ci, int len) {
	return vectAdd_atomic_<T, SumOp<T>>(a, b, c, ai, ci, len);
}

template<typename T>
int vectMatrixMult(T* a, MatrixAccessor<T>& b, T* c, uint32_t ai, uint32_t bi, uint32_t len) {
//	uint32_t bix = bi + threadIdx.x * len;
	uint32_t m2clen = b.len() / len;
	
	for(uint32_t j = 0, bix = bi; j < m2clen; ++j, bix+=len) {
//	for(uint32_t j = 0; j < m2clen; ++j, bix+=len) {
		T result = dotProduct(a, b.vals(0), ai, bix, len);
		if(threadIdx.x == 0) {
			c[bi + j] = result;
//			if(debug_row())
//				printf("vectMatrixMult bid=%d bix=%d len=%d m2clen=%d c[%d]=%4.3f\n", blockIdx.x, bix, len, m2clen, bi+j, c[bi+j]);
		}
	}
	return len;
}

template<typename T>
uint32_t vectOuterMultAdd(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t ci, uint32_t len1, uint32_t len2) {
//	uint32_t cix = ci + threadIdx.x * len2;
//	if(threadIdx.x == 0 && blockIdx.x < 3)
//		printf("vectOuterMultAdd cix=%d\n", cix);
	// vectMultiplyAdd(a[ai+threadIdx.x], b, c, bi, cix, len2);
//	ProductOp<T> op;
//	return vectAdd_atomic<T, ProductOp<T>>(b, a[ai+threadIdx.x], c, bi, ci, len2, op);

	uint32_t i = threadIdx.x;
//	uint32_t bix = bi + blockIdx.x * len2;
uint32_t bix = 0;
	while (i < len1) {
		if(a[ai + i != 0]) {
			for(uint32_t j=0; j < len2; ++j) {
				atomicAdd(&(c[i * len2 + j]), a[ai + i] * b[bix + j]);
//				if (debug_row() && debug_thread())
//					printf("vectOuterMultAdd: bid=%d, tid=%d, ai=%d, bix=%d, ci=%d, len1=%d, len2=%d, a[%d]=%4.3f, b[%d]=%4.3f, c[%d]=%4.3f\n",
//							blockIdx.x, threadIdx.x, ai, bix, ci, len1, len2, ai+i, a[ai+i], bix+j, b[bix+j], i*len2+j, c[i*len2+j]);


//				vectAdd_atomic<T, ProductOp<T>>(b, a[ai+i], c, bi, ci, len2, op);
			}
		}
		i += blockDim.x;
	}


	return len1*len2;
}

/* --------------------------------------------------------------------------------------------------------------------
 * Binary to intermediate
 */

template<typename T>
Vector<T>& vectPlusWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, SumOp<T>>(a, b, ai, len, fop);
}

template<typename T>
Vector<T>& vectPlusWrite(T a, T* b, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, SumOp<T>>(a, b, bi, len, fop);
}

template<typename T>
Vector<T>& vectPlusWrite(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, SumOp<T>>(a, b, ai, bi, len, fop);
}

template<typename T>
Vector<T>& vectMinusWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, MinusOp<T>>(a, b, ai, len, fop);
}

template<typename T>
Vector<T>& vectMinusWrite(T a, T* b, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, MinusOp<T>>(a, b, bi, len, fop);
}

template<typename T>
Vector<T>& vectMinusWrite(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, MinusOp<T>>(a, b, ai, bi, len, fop);
}

template<typename T>
Vector<T>& vectMultWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vectMultWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d\n", blockIdx.x, threadIdx.x, ai, 0, len);
	
	return vectWriteBinary<T, ProductOp<T>>(a, b, ai, len, fop);
}

template<typename T>
Vector<T>& vectMultWrite(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, ProductOp<T>>(a, b, ai, bi, len, fop);
}

template<typename T>
Vector<T>& vectDivWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, DivOp<T>>(a, b, ai, len, fop);
}

template<typename T>
Vector<T>& vectDivWrite(T a, T* b, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, DivOp<T>>(a, b, bi, len, fop);
}

template<typename T>
Vector<T>& vectMinWrite(T* a, T b, int ai, int len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, MinOp<T>>(a, b, ai,len, fop);
}

template<typename T>
Vector<T>& vectMinWrite(T a, T* b, int bi, int len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, MinOp<T>>(a, b, bi, len, fop);
}

template<typename T>
Vector<T>& vectLessequalWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, LessEqualOp<T>>(a, b, ai, len, fop);
}

template<typename T>
Vector<T>& vectGreaterequalWrite(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteBinary<T, GreaterEqualOp<T>>(a, b, ai, len, fop);
}


/* --------------------------------------------------------------------------------------------------------------------
 * Binary to output
 */

template<typename T>
void vectMinusWrite(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t ci, uint32_t len) {
	return vectWriteBinary<T, MinusOp<T>>(a, b, c, ai, bi, ci, len);
}

template<typename T>
void vectMultWrite(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t ci, uint32_t len) {
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vectMultWrite: bid=%d, tid=%d, ai=%d, ci=%d, len=%d\n", blockIdx.x, threadIdx.x, ai, 0, len);
	return vectWriteBinary<T, ProductOp<T>>(a, b, c, ai, bi, ci, len);
}

/* --------------------------------------------------------------------------------------------------------------------
 * Unary to intermediate
 */

 template<typename T>
Vector<T>& vectExpWrite(T* a, int ai, int len, SpoofOp<T>* fop) {
	return vectWriteUnary<T, ExpOp<T>>(a, ai, len, fop);
}

 template<typename T>
Vector<T>& vectSignWrite(T* a, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteUnary<T, SignOp<T>>(a, ai, len, fop);
}

template<typename T>
Vector<T>& vectRoundWrite(T* a, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteUnary<T, RoundOp<T>>(a, ai, len, fop);
}

template<typename T>
Vector<T>& vectAbsWrite(T* a, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteUnary<T, AbsOp<T>>(a, ai, len, fop);
}

template<typename T>
Vector<T>& vectFloorWrite(T* a, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	return vectWriteUnary<T, FloorOp<T>>(a, ai, len, fop);
}

#endif // SPOOF_UTILS_CUH
