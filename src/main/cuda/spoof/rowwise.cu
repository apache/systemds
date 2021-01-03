//%TMP%

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

//%TMP_MEM%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"
#include "Matrix.h"

template<typename T>
__device__ void printArray(T* a, uint32_t len) {
    if(blockIdx.x == 1 && threadIdx.x==0) {
        printf("block=%d, thread=%d, array[len=%d]:\n", blockIdx.x, threadIdx.x, len);
        for(auto i = 0; i < (blockIdx.x + 10); ++i)
            printf(" %f", a[i]);
        printf("\n");
    }
}

template<typename T, int NUM_B>
struct SpoofRowwiseOp {
	MatrixAccessor<T> a;
	MatrixAccessor<T> b[NUM_B];
	MatrixAccessor<T> c;
	T* scalars;
	uint32_t grix;

	SpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, uint32_t grix) : scalars(scalars), grix(grix) {
//		if(blockIdx.x==0)
//			printf("init B\n");
		a.init(A);
		c.init(C);
		if(B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(B[i]));
	}

	__device__  __forceinline__ void operator()(uint32_t ai, uint32_t ci, uint32_t rix) {
		
//%BODY_dense%
//    printArray(TMP27, TMP27_len);
//        if(blockIdx.x == 0 && threadIdx.x==0) {
//			printf("TMP27=%f\n", TMP27);
//        }
	}
};

template<typename T, int NUM_B>
__global__ void /*%TMP%*/SPOOF_OP_NAME (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, uint32_t grix) {
	const uint& rix = blockIdx.x;
//	if(threadIdx.x == 0 && blockIdx.x == 1) {
//		MatrixAccessor<T> ma;
//		ma.init(&b[0]);
//		printf("bid=%d len=%d c_len=%d b.pos=%d b.len=%d\n", blockIdx.x, len, c_len, ma.pos(blockIdx.x), ma.len());
//		for(auto i = 0; i < len; ++i) {
//			T val =  ma.val(blockIdx.x, i);
//			printf("%f ", val);
//		}
//		printf("\n");
//	}
//	return;
//	if(rix < 3)
//		printf("bla\n");
//	return;
	SpoofRowwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, grix + rix);
	// spoof_op.c_len = c_len;
	// spoof_op.c_len = c->len_r();

	if(threadIdx.x == 0) {
		printf("blubb\n");
		printf("c=%f\n", spoof_op.c.val(0, 0));
	}
//	if(threadIdx.x == 0 && blockIdx.x == 0) {
//		printf("bid=%d len=%d c_len=%d\n", blockIdx.x, len, c_len);

//if(b) {
//		MatrixAccessor<T> ma;
//		ma.init(&b[0]);
//		printf("bid=%d len=%d c_len=%d b.pos=%d b.len=%d\n", blockIdx.x, len, c_len, ma.pos(blockIdx.x), ma.len());
////		return;
//		for(auto i = ma.pos(blockIdx.x); i < ma.len(); ++i) {
//		//			T val =  ma.val(blockIdx.x, i);
//			T val =  ma.val(0, i);
//			printf("%f ", val);
//		}
//		printf("\n");
//		}
//	}

	uint32_t ai = blockIdx.x * a->cols;
	uint32_t ci = blockIdx.x * c->cols;
	spoof_op(ai, ci, rix);
};
