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

// CellType: %TYPE%
// AggOp: %AGG_OP_NAME%
// SparseSafe: %SPARSE_SAFE%
// SEQ: %SEQ%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"
#include "Matrix.h"

template<typename T, int NUM_B>
struct SpoofCellwiseOp {
		MatrixAccessor<T> A;
		MatrixAccessor<T> b[NUM_B];
		MatrixAccessor<T> c;
		T* scalars;
		uint32_t grix;
		T* avals;
		uint32_t* aix;
		uint32_t alen;
		uint32_t& n;

	SpoofCellwiseOp(Matrix<T>* _A, Matrix<T>* _B, Matrix<T>* _C, T* scalars, T* tmp_stor, uint32_t grix) :
			/*SpoofOp<T, NUM_B>(A, B, C, scalars, tmp_stor, grix),*/ n(_A->cols), scalars(scalars), grix(grix) {
		A.init(_A);
		c.init(_C);
		alen = A.row_len(grix);

		if(_B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(_B[i]));
	}

	__device__  __forceinline__ T operator()(T a, uint32_t idx) {
		uint32_t rix = idx / A.cols();
		uint32_t cix = idx % A.cols();
		grix+=rix;

%BODY_dense%
		return %OUT%;
	}
};

//__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {
template<typename T, int NUM_B>
__global__ void %TMP% (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	%AGG_OP%<T> agg_op;
	SpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, tmp_stor, grix);
	%TYPE%<T, %AGG_OP%<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), a->rows * a->cols, %INITIAL_VALUE%, agg_op, spoof_op);
};
