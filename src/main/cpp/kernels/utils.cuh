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

#ifndef __UTILS_H
#define __UTILS_H

#pragma once

#include <cuda_runtime.h>

/**
 * Solution suggested by [1] to have different types of shared memory
 * Without this, compiling a templated kernel that used a template type
 * to declare shared memory usage, caused the compiler to emit a warning.
 *
 * [1] https://stackoverflow.com/a/49224531/12055283
 */
template<typename T>
__device__ T* shared_memory_proxy()
{
	// do we need an __align__() here? I don't think so...
	extern __shared__ unsigned char memory[];
	return reinterpret_cast<T*>(memory);
}

/**
*  Class for correctly addressing vectorized data arrays in a templated kernel
*  float version
*/
struct float2Accessor
{
	__device__  static float2 get(float* array, unsigned int idx)
	{
		return *reinterpret_cast<float2*>(&array[idx]);
	}

	__device__  static float2 make(float x, float y)
	{
		return make_float2(x, y);
	}

	__device__  static float2 init()
	{
		return make_float2(0.0f, 0.0f);
	}

	__device__ static void put(float* array, unsigned int idx, float val_x, float val_y)
	{
		*(reinterpret_cast<float2*>(array + idx)) = make_float2(val_x, val_y);
	}
};

/**
*  Class for correctly addressing vectorized data arrays in a templated kernel
*  double version
*/
struct double2Accessor
{
	__device__  static double2 get(double* array, unsigned int idx)
	{
		return *(reinterpret_cast<double2*>(&array[idx]));
	}

	__device__  static double2 init()
	{
		return make_double2(0.0, 0.0);
	}

	__device__  static double2 make(double x, double y)
	{
		return make_double2(x, y);
	}

	__device__ static void put(double* array, unsigned int idx, double val_x, double val_y)
	{
		*(reinterpret_cast<double2*>(array + idx )) = make_double2(val_x, val_y);
	}
};

extern "C" __global__ void double2float_f(double *A, float *ret, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		// TODO: Use __double2float_rd or __double2float_rn  or __double2float_ru or
		// __double2float_rz after
		ret[tid] = (float) A[tid];
	}
}

extern "C" __global__ void float2double_f(float *A, double *ret, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		ret[tid] = (double) A[tid];
	}
}

#endif // __UTILS_H
