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
struct SpoofCellwiseOp : public SpoofOp<T, NUM_B> {
//	T**b; T* scalars;
//	int m, n, grix_;
//	uint32_t m, n;

	T* scalars;
	uint32_t grix;
	
//	SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix) :
//		b(b), scalars(scalars), m(m), n(n), grix_(grix) {}
	SpoofCellwiseOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :
			SpoofOp<T, NUM_B>(A, B, C, scalars, tmp_stor, grix), scalars(scalars), grix(grix){}

	__device__  __forceinline__ T operator()(T a, uint32_t idx) {
		uint32_t rix = idx / this->a.cols();
		uint32_t cix = idx % this->a.cols();
		grix+=rix;

%BODY_dense%
		return %OUT%;
	}
};

//__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {
template<typename T, int NUM_B>
__global__ void %TMP% (Matrix<T>* a, Matrix<T>* b, Matrix<T>* c, T* scalars, T* tmp_stor, uint32_t grix) {
	%AGG_OP%<T> agg_op;
//	SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix);
	SpoofCellwiseOp<T, NUM_B> spoof_op(a, b, c, scalars, tmp_stor, grix);
	%TYPE%<T, %AGG_OP%<T>, SpoofCellwiseOp<T, NUM_B>>(%INITIAL_VALUE%, agg_op, spoof_op);
};
