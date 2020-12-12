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
// #include "intellisense_cuda_intrinsics.h"

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
__device__ T BLOCK_ROW_AGG(T *a, T *b, int len, AggOp agg_op, LoadOp load_op) {
	auto sdata = shared_memory_proxy<T>();
	uint tid = threadIdx.x;

	// sdata[tid] = AggOp::init();
	// __syncthreads();

			// if(blockIdx.x == 0 && threadIdx.x == 0)
		 //  printf("tid=%d sdata[tid + 128]=%f\n", tid, sdata[tid+128]);
	
	// Initalize shared mem and leave if tid > row length. 
	if(tid >= len) { return sdata[tid] = AggOp::init();; }

	// sdata[tid+128] = AggOp::init();
	// sdata[tid+64] = AggOp::init();
	// sdata[tid+32] = AggOp::init();
	// sdata[tid+16] = AggOp::init();
	// sdata[tid+8] = AggOp::init();
	// sdata[tid+4] = AggOp::init();
	// sdata[tid+2] = AggOp::init();
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
__device__ T dotProduct(T* a, T* b, int ai, int bi, int len) {
	SumOp<T> agg_op;
	ProductOp<T> load_op;
	T ret =  BLOCK_ROW_AGG(&a[ai], &b[bi], len, agg_op, load_op);
	if( threadIdx.x == 0)
		printf("bid=%d, tid=%d, dot=%f\n", blockIdx.x, threadIdx.x, ret);
	return ret;
}

template<typename T>
__device__ T vectSum(T* a, int ai, int len) {
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

//template<typename T>
//__device__ void vectMultAdd_atomic(T* a, T b, T* c, int ai, int ci, int len) {
//    uint tid = threadIdx.x;
//    if(tid >= len)
//        return;
//    uint i = tid;
//
//    while (i < len) {
//        atomicAdd(&(c[ci + i]), a[ai + i] * b);
////        printf("block %d adding %f to c[%d]\n",blockIdx.x, b, tid);
//        i += blockDim.x;
//    }
//}

//template<typename T>
//__device__ void vectDivAdd_atomic(T* a, T b, T* c, int ai, int ci, int len) {
//    uint tid = threadIdx.x;
//    if(tid >= len)
//        return;
//    uint i = tid;
//
//    while (i < len) {
//        atomicAdd(&(c[ci + i]), a[ai + i] / b);
////        printf("block %d adding %f to c[%d]\n",blockIdx.x, b, tid);
//        i += blockDim.x;
//    }
//}

template<typename T, typename Op>
__device__ void vectAdd_atomic(T* a, T b, T* c, int ai, int ci, int len, Op op) {
	uint tid = threadIdx.x;
	if(tid >= len)
		return;
	uint i = tid;

	while (i < len) {
		atomicAdd(&(c[ci + i]), op(a[ai + i], b));
		i += blockDim.x;
	}
}

template<typename T, typename Op>
__device__ void vectWrite_atomic(T* a, T* c, int ai, int ci, int len, Op op) {
    uint tid = threadIdx.x;
    if(tid >= len)
        return;
    uint i = tid;

    while (i < len) {
        // atomicExch(&(c[ci + i]), op(a[ai + i], a[i]));
    	c[ci + i] = op(a[ai + i], a[i]);
        i += blockDim.x;
    }
}

// template<typename Op>
// __device__ void vectWrite_atomic(double* a, double* c, int ai, int ci, int len, Op op) {
// 	uint tid = threadIdx.x;
// 	if(tid >= len)
// 		return;
// 	uint i = tid;
//
// 	while (i < len) {
//
// 		double old = __longlong_as_double(atomicExch(reinterpret_cast<unsigned long long int*>(&(c[ci + i])), __double_as_longlong(op(a[ai + i], a[i]))));
// 		// printf("bid=%d, tid=%d, old=%f, c[%d]=%f, a[%d]=%f]\n", blockIdx.x, threadIdx.x, old, ci+i, c[ci+i], ai+i,a[ai + i]);
//
// 		i += blockDim.x;
// 	}
// }

template<typename T>
__device__ void vectMultAdd(T* a, T b, T* c, int ai, int ci, int len) {
	ProductOp<T> op;
	vectAdd_atomic<T, ProductOp<T>>(a, b, c, ai, ci, len, op);
}

template<typename T>
__device__ void vectDivAdd(T* a, T b, T* c, int ai, int ci, int len) {
	DivOp<T> op;
	vectAdd_atomic<T, DivOp<T>>(a, b, c, ai, ci, len, op);
}

template<typename T>
__device__ void vectWrite(T* a, T* c, int ai, int ci, int len) {
	IdentityOp<T> op;
	vectWrite_atomic<T, DivOp<T>>(a, c, ai, ci, len, op);
}

template<typename T>
__device__ T* vectCbindWrite(T* a, T b, int ai, int len) {
	if(threadIdx.x < len) {
//		 if(blockIdx.x==0 && threadIdx.x ==0)
		 	printf("vecCbindWrite: bid=%d, tid=%d, ai=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, len, ai * len + threadIdx.x, a[ai * len + threadIdx.x]);
//        printf("block %d thread %d, b=%f, ci=%d, len=%d, a[%d]=%f\n",blockIdx.x, threadIdx.x, b, ci, len, ai, a[ai]);
		TEMP_STORAGE[blockIdx.x * (len) + threadIdx.x] = a[ai * len + threadIdx.x];
//		TEMP_STORAGE[ai + blockIdx.x * (len+1) + threadIdx.x] = a[ai + threadIdx.x];
		// __longlong_as_double(atomicExch(reinterpret_cast<unsigned long long int*>(&(c[ci * (len+1) + threadIdx.x])), __double_as_longlong(a[ai + threadIdx.x])));
	}
	if(threadIdx.x == len) {
//		printf("---> block %d thread %d, b=%f, ci=%d, len=%d, a[%d]=%f\n",blockIdx.x, threadIdx.x, b, ci, len, ai, a[ai]);
//		TEMP_STORAGE[ai * (len+1) + threadIdx.x] = b;
		TEMP_STORAGE[blockIdx.x * (len+1) + threadIdx.x] = b;
	}
	return &TEMP_STORAGE[0];
}

template<typename T>
struct LessEqualOp {
	__device__  __forceinline__ static T execute(T a, T b) {
		return (a <= b) ? 1.0 : 0.0;
	}
};

template<typename T>
struct GreaterOp {
	__device__  __forceinline__ static T execute(T a, T b) {
		return (a > b) ? 1.0 : 0.0;
	}
};

template<typename T, typename OP>
__device__ void vectWrite_(T* a, T b, T* c, int ai, int ci, int len) {

	uint tid = threadIdx.x;
	uint bid = blockIdx.x;
//    if(tid >= len)
//        return;
    uint i = tid;

    len = len+1;
    while (i < len+1) {
        // atomicExch(&(c[ci + i]), op(a[ai + i], a[i]));
		printf("vecWrite: bid=%d, tid=%d, ai=%d, len=%d, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai, len, bid * len + i, a[bid * len + i]);

		c[ci + bid * len + i] = OP::execute(a[bid * len + i], b);
        i += blockDim.x;
    }
}
template<typename T>
T* vectLessequalWrite(T* a, T b, int ai, int len) {
	// LessEqualOp<T> op;
	vectWrite_<T, LessEqualOp<T>>(a, b, &TEMP_STORAGE[0], ai, 0, len);
	return &TEMP_STORAGE[0];
}

template<typename T>
__device__ void vectWrite(T* a, T* c, int ci, int len) {
	vectWrite_<T, IdentityOp<T>>(a, 0, c, 0, ci, len);

}

template<typename T>
T* vectDivWrite(T* a, T b, int ai, int len) {
	// LessEqualOp<T> op;
	vectWrite_<T, DivOp<T>>(a, b, &TEMP_STORAGE[0], ai, 0, len);
	return &TEMP_STORAGE[0];
}

template<typename T>
T* vectGreaterAdd(T* a, T b, int ai, int len) {
	// LessEqualOp<T> op;
	vectAdd_atomic()<T, GreaterOp<T>>(a, b, &TEMP_STORAGE[0], ai, 0, len);
	return &TEMP_STORAGE[0];
}
#endif // SPOOF_UTILS_CUH
