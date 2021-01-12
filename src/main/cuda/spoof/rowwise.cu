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
// ConstDim2: %CONST_DIM2%
// TB1: %TB1%
// VectMem: %VECT_MEM%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"
#include "Matrix.h"

template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
struct SpoofRowwiseOp //%HAS_TEMP_VECT%
{
	MatrixAccessor<T> a;
	MatrixAccessor<T> b[NUM_B];
	MatrixAccessor<T> c;
	T* scalars;
	uint32_t grix;

	//%TMP_MEM_DECLARATION%

	SpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) : scalars(scalars), grix(grix) {
		a.init(A);
		c.init(C);
		//%TMP_MEM%
		if(B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(B[i]));
	}

	__device__  __forceinline__ void exec(uint32_t ai, uint32_t ci, uint32_t rix) {
		
//%BODY_dense%
	}

//%GET_TEMP_STORAGE%
};

template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
	uint32_t ai = rix * a->cols;
	uint32_t ci = rix * c->cols;
	spoof_op.exec(ai, ci, rix);
};
