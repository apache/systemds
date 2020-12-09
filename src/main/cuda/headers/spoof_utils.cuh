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

template<typename T>
__device__ T BLOCK_DOT(T *a, T *b, int len, SumOp<T> reduction_op) {
    auto sdata = shared_memory_proxy<T>();

    uint tid = threadIdx.x;
    if(tid >= len)
        return 0;

    uint i = tid;

    T v = 0;
    while (i < len) {
        v = reduction_op(v, a[i] * b[i]);
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

    __syncthreads();
    return sdata[0];
}

template<typename T>
__device__ T dotProduct(T* a, T* b, int ai, int bi, int len) {
    SumOp<T> agg_op;
    return BLOCK_DOT(&a[ai], &b[bi], len, agg_op);
}

template<typename T>
__device__ void vectMultAdd_atomic(T* a, T b, T* c, int ai, int ci, int len) {
    uint tid = threadIdx.x;
    if(tid >= len)
        return;
    uint i = tid;

    while (i < len) {
        atomicAdd(&(c[ci + i]), a[ai + i] * b);
//        printf("block %d adding %f to c[%d]\n",blockIdx.x, b, tid);
        i += blockDim.x;
    }
}

#endif // SPOOF_UTILS_CUH
