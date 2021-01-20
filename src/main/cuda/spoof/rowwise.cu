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
#include "../headers/Matrix.h"

template<typename T>
struct TempStorage {
		__device__ virtual  Vector<T>& getTempStorage(uint32_t len) = 0;
};

template<typename T, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN >
struct TempStorageImpl : public TempStorage<T> {
	//%TMP_MEM_DECLARATION%
	
//	TempStorage() {
//		//%TMP_MEM%
//	}
	RingBuffer<T,NUM_TMP_VECT> temp_rb;
	
	TempStorageImpl(T* tmp_stor) {
		if(tmp_stor) {
			uint32_t tmp_row_offset = TMP_VECT_LEN * NUM_TMP_VECT * blockIdx.x;
			temp_rb.init(tmp_row_offset, TMP_VECT_LEN, tmp_stor);
		}
	}
	
	__device__ Vector<T>& getTempStorage(uint32_t len) {
		if(debug_row() && debug_thread())
			printf("getTempStorage(len=%d)\n", len);
		Vector<T>& vec = temp_rb.next();
		vec.length = len;
		return vec;
	}
	
};
//, public TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>
//TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>(tmp_stor),
template<typename T, int NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
struct SpoofRowwiseOp : public SpoofOp<T, NUM_B> //%HAS_TEMP_VECT%
{
	T* scalars;
	uint32_t grix;
	
	SpoofRowwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :
		SpoofOp<T, NUM_B>(A, B, C, scalars, tmp_stor, grix), //%INIT_TEMP_VECT%
		        scalars(scalars), grix(grix) {}

	__device__  __forceinline__ void exec_dense(uint32_t ai, uint32_t ci, uint32_t rix) {
		MatrixAccessor<T>& a = this->a;
		MatrixAccessor<T>* b = &(this->b[0]);
		MatrixAccessor<T>& c = this->c;
//%BODY_dense%
	}

	__device__  __forceinline__ void exec_sparse(uint32_t ai, uint32_t ci, uint32_t rix) {

//%BODY_sparse%
	}

//%GET_TEMP_STORAGE%

};

template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_DENSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
if(false) {
	if (debug_row() && debug_thread()) {
		printf("DENSE c rows=%d cols=%d nnz=%d\n",
			   spoof_op.c.rows(), spoof_op.c.cols(), spoof_op.c.nnz());
		for (auto i = 0; i < spoof_op.c.len(); i++) {
			printf("i=%d val=%4.3f\n", i, spoof_op.c.val(0,i));
		}
		
//		printf("DENSE b[1] rows=%d cols=%d nnz=%d\n",
//			   spoof_op.b[1].rows(), spoof_op.b[1].cols(), spoof_op.b[1].nnz());
//		for (auto i = 0; i < spoof_op.b[1].len(); i++) {
//			printf("i=%d val=%4.3f\n", i, spoof_op.b[1].val(0,i));
//		}
	}
}
//	return;
	spoof_op.exec_dense(rix * a->cols, rix * c->cols, rix);
};

template<typename T, uint32_t NUM_B, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	const uint& rix = blockIdx.x;
	SpoofRowwiseOp<T, NUM_B, NUM_TMP_VECT, TMP_VECT_LEN> spoof_op(a, b, c, scalars, tmp_stor, grix + rix);
//	spoof_op.alen = spoof_op.a.row_len(rix);
//	spoof_op.aix = spoof_op.a.col_idxs(0);
//	spoof_op.avals = spoof_op.a.vals(0);

//	if(debug_row() && debug_thread()) {
//		printf("a rows=%d cols=%d nnz=%d\n", a->rows, a->cols, a->nnz);
//		printf("row_len(%d)=%d\n", rix, spoof_op.alen);
//		for (auto i = 0; i < spoof_op.alen; i++) {
//			printf("i=%d col=%d val=%4.3f\n", i, spoof_op.aix[i], spoof_op.avals[i]);
//		}
//	}
//	if(blockIdx.x == 0)
	spoof_op.exec_sparse(a->row_ptr[rix], rix * c->cols, rix);
}