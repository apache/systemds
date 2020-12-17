%TMP%

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

// RowType: %TYPE%

%TMP_MEM%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"
#include "Matrix.h"

template<typename T>
__device__ void printArray(T* a, int len) {
    if(blockIdx.x == 1 && threadIdx.x==0) {
        printf("block=%d, thread=%d, array[len=%d]:\n", blockIdx.x, threadIdx.x, len);
        for(auto i = 0; i < (blockIdx.x + 10); ++i)
            printf(" %f", a[i]);
        printf("\n");
    }
}

template<typename T>
struct SpoofRowwiseOp {
	T* a;
	Matrix<T>* b;
	T* c;
	T* scalars;
	int len, grix, c_len;

	SpoofRowwiseOp(T* a, Matrix<T>* b, T* scalars, T* c, int len, int grix) : a(a), b(b), c(c), scalars(scalars), len(len), grix(grix) {}

	__device__  __forceinline__ void operator()(int ai, int ci, int rix) const {
		
%BODY_dense%
//    printArray(TMP27, TMP27_len);
//        if(blockIdx.x == 1 && threadIdx.x==0) {
//			printf("TMP26=%f\n", TMP26);
//        }
	}
};

template<typename T>
__global__ void %TMP% (T* a, Matrix<T>* b, T* scalars, T* c, uint c_len, int len, int grix) {
	SpoofRowwiseOp<T> spoof_op(a, b, scalars, c, len, grix + blockIdx.x);
	spoof_op.c_len = c_len;
	int ai = blockIdx.x * len;
	int ci = blockIdx.x * c_len;
	spoof_op(ai, ci, blockIdx.x);
};
