/*%TMP%*/SPOOF_OP_NAME
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
		T* avals;
		uint32_t* aix;
		uint32_t alen;
		uint32_t& n;
		uint32_t _grix;

	SpoofCellwiseOp(Matrix<T>* _A, Matrix<T>* _B, Matrix<T>* _C, T* scalars, uint32_t grix) :
			n(_A->cols), scalars(scalars), _grix(grix) {
		A.init(_A);
		c.init(_C);
		alen = A.row_len(grix);

		if(_B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(_B[i]));
	}

	__device__  __forceinline__ T operator()(T a, uint32_t idx, uint32_t rix, uint32_t cix) {
//%NEED_GRIX%
%BODY_dense%
		return %OUT%;
	}
};

template<typename T, int NUM_B>
__global__ void /*%TMP%*/SPOOF_OP_NAME_DENSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, uint32_t n, uint32_t grix) {
	%AGG_OP%<T> agg_op;
	SpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, grix);
	%TYPE%<T, %AGG_OP%<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), n, %INITIAL_VALUE%, agg_op, spoof_op);
};

template<typename T, int NUM_B>
__global__ void /*%TMP%*/SPOOF_OP_NAME_SPARSE (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, uint32_t n, uint32_t grix) {
	%AGG_OP%<T> agg_op;
	SpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, grix);
	%TYPE%_SPARSE<T, %AGG_OP%<T>, SpoofCellwiseOp<T, NUM_B>>(&(spoof_op.A), &(spoof_op.c), n, %INITIAL_VALUE%, agg_op, spoof_op);
};