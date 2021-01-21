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
#include "TempStorage.cuh"

template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
struct SpoofRowwiseOp //%HAS_TEMP_VECT%
{
	MatrixAccessor<T> a;
	MatrixAccessor<T> b[NUM_B];
	MatrixAccessor<T> c;
	T* scalars;
	uint32_t grix;
	T* avals;
	uint32_t* aix;
	uint32_t alen;
		
	SpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :
		        scalars(scalars), grix(grix) /*%INIT_TEMP_VECT%*/ {
		a.init(A);
		c.init(C);
		
		if(B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(B[i]));
	}

	__device__  __forceinline__ void exec_dense(uint32_t ai, uint32_t ci, uint32_t rix) {
//%BODY_dense%
	}

	__device__  __forceinline__ void exec_sparse(uint32_t ai, uint32_t ci, uint32_t rix) {
//%BODY_sparse%
	}
};

template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_DENSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
	spoof_op.exec_dense(rix * a->cols, rix * c->cols, rix);
};

template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
	spoof_op.alen = spoof_op.a.row_len(rix);
	spoof_op.aix = spoof_op.a.col_idxs(0);
	spoof_op.avals = spoof_op.a.vals(0);
	spoof_op.exec_sparse(a->row_ptr[rix], rix * c->cols, rix);
}