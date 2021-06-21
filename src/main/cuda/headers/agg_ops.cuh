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
#ifndef AGG_OPS_H
#define AGG_OPS_H

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

// #include "intellisense_cuda_intrinsics.h"

template<typename T>
struct SumNeutralElement {
	static __device__ __forceinline__ T get();
};

template<>
float SumNeutralElement<float>::get() { return 0.0f; }

template<>
double SumNeutralElement<double>::get() { return 0.0; }


template<typename T>
struct ProdNeutralElement {
	static __device__ __forceinline__ T get();
};

template<>
float ProdNeutralElement<float>::get() { return 1.0f; }

template<>
double ProdNeutralElement<double>::get() { return 1.0; }

template<typename T>
struct MinNeutralElement {
	static __device__ __forceinline__ T get();
};

template<>
float MinNeutralElement<float>::get() { return CUDART_INF_F; }

template<>
double MinNeutralElement<double>::get() { return CUDART_INF; }

template<typename T>
struct MaxNeutralElement {
	static __device__ __forceinline__ T get();
};

template<>
float MaxNeutralElement<float>::get() { return -CUDART_INF_F; }

template<>
double MaxNeutralElement<double>::get() { return -CUDART_INF_F; }

/**
 * Functor op for assignment op. This is a dummy/identity op.
 */
template<typename T>
struct IdentityOp {
	__device__  __forceinline__ T operator()(T a, uint32_t idx = 0, uint32_t rix = 0, uint32_t cix = 0) const {
		return a;
	}
	__device__  __forceinline__ static T exec(T a, T b) {
		return a;
	}
};

/**
 * Functor op for summation operation
 */
template<typename T>
struct SumOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return a + b;
	}

	__device__  __forceinline__ static T exec(T const & a, T const & b) {
		return a + b;
	}

    __device__  __forceinline__ static T init() {
	    return SumNeutralElement<T>::get();
	}
};

/**
 * Functor op for difference operation
 */
template<typename T>
struct MinusOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return a - b;
	}

	__device__  __forceinline__ static T exec(T const& a, T const& b) {
		return a - b;
	}

	__device__  __forceinline__ static T init() {
		return SumNeutralElement<T>::get();
	}
};

/**
 * Functor op for min operation
 */
template<typename T>
struct MinOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return a < b ? a : b;
	}

	__device__  __forceinline__ static T exec(T const & a, T const & b) {
		return a < b ? a : b;
	}

    __device__  __forceinline__ static T init() {
	    return MinNeutralElement<T>::get();
	}
};

template<>
struct MinOp<double> {
	__device__  __forceinline__  double operator()(double a, double b) const {
		// if(blockIdx.x==0 && threadIdx.x == 0)
		// if(threadIdx.x == 0)
			// printf("bid=%d, tid=%d, a=%f, b=%f =min=> %f\n", blockIdx.x, threadIdx.x, a, b, fmin(a, b));
		return fmin(a, b);
	}

	__device__  __forceinline__ static double exec(double const & a, double const & b) {
		return fmin(a, b);
	}

	__device__  __forceinline__ static double init() {
	    return MinNeutralElement<double>::get();
	}
};

template<>
struct MinOp<float> {
	__device__  __forceinline__ float operator()(float a, float b) const {
		return fminf(a, b);
	}

	__device__  __forceinline__ static float init() {
	    return MinNeutralElement<float>::get();
	}
};

/**
 * Functor op for max operation
 */
template<typename T>
struct MaxOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return fmax(a, b);
	}

	__device__  __forceinline__ static T exec(T const & a, T const & b) {
		return fmax(a, b);
	}

	__device__  __forceinline__ static T init() {
		return MaxNeutralElement<T>::get();
	}
};

template<>
struct MaxOp<float> {
	__device__ __forceinline__ float operator()(float a, float b) const {
		return fmaxf(a, b);
	}
	
	__device__  __forceinline__ static float exec(float const & a, float const & b) {
		return fmaxf(a, b);
	}
	
	__device__  __forceinline__ static float init() {
		return MaxNeutralElement<float>::get();
	}
};

/**
 * Functor op for product operation
 */
template<typename T>
struct ProductOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
//		if(blockIdx.x==0 && threadIdx.x ==0)
//		printf("prod a=%4.3f b=%4.3f\n", a, b);
		return a * b;
	}

	__device__  __forceinline__ static T exec(T const& a, T const& b) {
		return a * b;
	}

	__device__  __forceinline__ static T init() {
	    return 1.0;
	}
};

/**
 * Functor op for division operation
 */
template<typename T>
struct DivOp {
    __device__  __forceinline__ T operator()(T a, T b) const {
        ProductOp<T> prod_op;
        return prod_op(a, 1 / b);
    }

    __device__  __forceinline__ static T exec(T a, T b) {
        ProductOp<T> prod_op;
        return prod_op(a, 1 / b);
    }

	__device__  __forceinline__ static T init() {
	    return ProdNeutralElement<T>::get();
	}
};

/**
 * Functor op for mean operation
 */
template<typename T>
struct MeanOp {
	const long _size; ///< Number of elements by which to divide to calculate mean
	__device__ __forceinline__ MeanOp(long size) :
			_size(size) {
	}
	__device__  __forceinline__ T operator()(T total) const {
		return total / _size;
	}
};

#endif // __AGG_OPS_H
