
#ifndef __AGG_OPS_H
#define __AGG_OPS_H

#pragma once

#include <cuda_runtime.h>

/**
 * Functor op for assignment op. This is a dummy/identity op.
 */
template<typename T>
struct IdentityOp {
	__device__  __forceinline__ T operator()(T a) const {
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
};

/**
 * Functor op for min operation
 */
template<typename T>
struct MinOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return fmin(a, b);
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
};

template<>
struct MaxOp<float> {
	__device__ __forceinline__ float operator()(float a, float b) const {
		return fmaxf(a, b);
	}
};

/**
 * Functor op for product operation
 */
template<typename T>
struct ProductOp {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return a * b;
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
float MinNeutralElement<float>::get() { return INFINITY; }

template<>
double MinNeutralElement<double>::get() { return INFINITY; }

template<typename T>
struct MaxNeutralElement {
	static __device__ __forceinline__ T get();
};

template<>
float MaxNeutralElement<float>::get() { return -INFINITY; }

template<>
double MaxNeutralElement<double>::get() { return -INFINITY; }

#endif // __AGG_OPS_H
