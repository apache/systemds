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

#include "utils.cuh"
#include "agg_ops.cuh"
#include "reduction.cuh"
#include "Matrix.h"

/**
 * Do a summation over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stored in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template<typename T>
__device__ void reduce_sum(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t n) {
	SumOp<T> agg_op;	
	IdentityOp<T> spoof_op;
	FULL_AGG<T, SumOp<T>, IdentityOp<T>>(in, out, n, (T) 0.0, agg_op, spoof_op);
}

extern "C" __global__ void reduce_sum_f(Matrix<float>* in, Matrix<float>* out, uint32_t n) {
	MatrixAccessor<float> _in(in);
	MatrixAccessor<float> _out(out);
	reduce_sum(&_in, &_out, n);
}

extern "C" __global__ void reduce_sum_d(Matrix<double>* in, Matrix<double>* out, uint32_t n) {
	MatrixAccessor<double> _in(in);
	MatrixAccessor<double> _out(out);
	reduce_sum(&_in, &_out, n);
}

//extern "C" __global__ void reduce_sum_f(float *g_idata, float *g_odata, uint n) {
//	reduce_sum(g_idata, g_odata, n);
//}

/**
 * Do a summation over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_row_sum(T *g_idata, T *g_odata, uint rows, uint cols) {
//	SumOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	ROW_AGG<T, SumOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, 0.0, agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_row_sum_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_row_sum(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_row_sum_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_row_sum(g_idata, g_odata, rows, cols);
//}

/**
 * Do a summation over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_col_sum(T *g_idata, T *g_odata, uint rows, uint cols) {
//	SumOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	COL_AGG<T, SumOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, (T)0.0, agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_col_sum_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_col_sum(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_col_sum_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_col_sum(g_idata, g_odata, rows, cols);
//}


/**
 * Do a max over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template<typename T>
__device__ void reduce_max(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t n) {
	MaxOp<T> agg_op;
	IdentityOp<T> spoof_op;
	FULL_AGG<T, MaxOp<T>, IdentityOp<T>>(in, out, n, -MAX<T>(), agg_op, spoof_op);
}

extern "C" __global__ void reduce_max_f(Matrix<float>* in, Matrix<float>* out, uint32_t n) {
	MatrixAccessor<float> _in(in);
	MatrixAccessor<float> _out(out);
	reduce_max(&_in, &_out, n);
}

extern "C" __global__ void reduce_max_d(Matrix<double>* in, Matrix<double>* out, uint32_t n) {
	MatrixAccessor<double> _in(in);
	MatrixAccessor<double> _out(out);
	reduce_max(&_in, &_out, n);
}

/**
 * Do a max over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_row_max(T *g_idata, T *g_odata, uint rows, uint cols) {
//	MaxOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	ROW_AGG<T, MaxOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, -MAX<T>(), agg_op, spoof_op);
//}

//extern "C" __global__ void reduce_row_max_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_row_max(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_row_max_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_row_max(g_idata, g_odata, rows, cols);
//}

/**
 * Do a max over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_col_max(T *g_idata, T *g_odata, uint rows, uint cols) {
//	MaxOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	COL_AGG<T, MaxOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, -MAX<T>(), agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_col_max_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_col_max(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_col_max_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_col_max(g_idata, g_odata, rows, cols);
//}


/**
 * Do a min over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template<typename T>
__device__ void reduce_min(MatrixAccessor<T>* in, MatrixAccessor<T>* out, uint32_t n) {
	MinOp<T> agg_op;
	IdentityOp<T> spoof_op;
	FULL_AGG<T, MinOp<T>, IdentityOp<T>>(in, out, n, MAX<T>(), agg_op, spoof_op);
}

extern "C" __global__ void reduce_min_f(Matrix<float>* in, Matrix<float>* out, uint32_t n) {
	MatrixAccessor<float> _in(in);
	MatrixAccessor<float> _out(out);
	reduce_min(&_in, &_out, n);
}

extern "C" __global__ void reduce_min_d(Matrix<double>* in, Matrix<double>* out, uint32_t n) {
	MatrixAccessor<double> _in(in);
	MatrixAccessor<double> _out(out);
	reduce_min(&_in, &_out, n);
}

/**
 * Do a min over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_row_min(T *g_idata, T *g_odata, uint rows, uint cols) {
//	MinOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	ROW_AGG<T, MinOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, MAX<T>(), agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_row_min_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_row_min(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_row_min_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_row_min(g_idata, g_odata, rows, cols);
//}

/**
 * Do a min over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_col_min(T *g_idata, T *g_odata, uint rows, uint cols) {
//	MinOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	COL_AGG<T, MinOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, MAX<T>(), agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_col_min_d(double *g_idata, double *g_odata, uint rows, uint cols) {
//	reduce_col_min(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_col_min_f(float *g_idata, float *g_odata, uint rows, uint cols) {
//	reduce_col_min(g_idata, g_odata, rows, cols);
//}


/**
 * Do a summation over all squared elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stored in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
//template<typename T>
//__device__ void reduce_sum_sq(T *g_idata, T *g_odata, uint n) {
//	SumSqOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	FULL_AGG<T, SumSqOp<T>, IdentityOp<T>>(g_idata, g_odata, n, 1, (T) 0.0, agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_sum_sq_d(double *g_idata, double *g_odata, uint n) {
//	reduce_sum_sq(g_idata, g_odata, n);
//}
//
//extern "C" __global__ void reduce_sum_sq_f(float *g_idata, float *g_odata, uint n) {
//	reduce_sum_sq(g_idata, g_odata, n);
//}

/**
 * Do a summation over all squared elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stored in device memory (of size n)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
//template<typename T>
//__device__ void reduce_col_sum_sq(T* g_idata, T* g_odata, uint rows, uint cols) {
//	SumSqOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	COL_AGG<T, SumSqOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, (T)0.0, agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_col_sum_sq_d(double* g_idata, double* g_odata, uint rows, uint cols) {
//	reduce_col_sum_sq(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_col_sum_sq_f(float* g_idata, float* g_odata, uint rows, uint cols) {
//	reduce_col_sum_sq(g_idata, g_odata, rows, cols);
//}

//template<typename T>
//__device__ void reduce_row_sum_sq(T* g_idata, T* g_odata, uint rows, uint cols) {
//	SumSqOp<T> agg_op;
//	IdentityOp<T> spoof_op;
//	ROW_AGG<T, SumSqOp<T>, IdentityOp<T>>(g_idata, g_odata, rows, cols, (T)0.0, agg_op, spoof_op);
//}
//
//extern "C" __global__ void reduce_row_sum_sq_d(double* g_idata, double* g_odata, uint rows, uint cols) {
//	reduce_row_sum_sq(g_idata, g_odata, rows, cols);
//}
//
//extern "C" __global__ void reduce_row_sum_sq_f(float* g_idata, float* g_odata, uint rows, uint cols) {
//	reduce_row_sum_sq(g_idata, g_odata, rows, cols);
//}
