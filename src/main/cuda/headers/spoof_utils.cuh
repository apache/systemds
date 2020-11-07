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
#ifndef SPOOF_UTILS_CUH
#define SPOOF_UTILS_CUH

#include <math_constants.h>

__constant__ double DOUBLE_EPS = 1.11022E-16; // 2 ^ -53
__constant__ double FLOAT_EPS = 1.49012E-08; // 2 ^ -26
__constant__ double EPSILON = 1E-11; // margin for comparisons ToDo: make consistent use of it

__device__ long long toInt64(double a) {
    return (signbit(a) == 0 ? 1.0 : -1.0) * abs(floor(a + DOUBLE_EPS));
}

__device__ int toInt32(float a) {
    return (signbit(a) == 0 ? 1.0 : -1.0) * abs(floor(a + FLOAT_EPS));
}

template<typename T>
__device__ T getValue(T* data, int rowIndex) {
    return data[rowIndex];
}

template<typename T>
__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {
    return data[rowIndex * n + colIndex];
}

template<typename T>
__device__ T intDiv(T a, T b);

template<>
__device__ double intDiv(double a, double b) {
    double ret = a / b;
    return (isnan(ret) || isinf(ret)) ? ret : toInt64(ret);
}

template<>
__device__ float intDiv(float a, float b) {
    float ret = a / b;
    return (isnan(ret) || isinf(ret)) ? ret : toInt32(ret);
}

template<typename T>
__device__ T modulus(T a, T b);

template<>
__device__ double modulus(double a, double b) {
    if (fabs(b) < DOUBLE_EPS)
        return CUDART_NAN;
    return a - intDiv(a, b) * b;
}

template<>
__device__ float modulus(float a, float b) {
    if (fabs(b) < FLOAT_EPS)
        return CUDART_NAN_F;
    return a - intDiv(a, b) * b;
}

template<typename T>
__device__ T bwAnd(T a, T b);

// ToDo: does not work with long long
template<>
__device__ double bwAnd(double a, double b) {
	return toInt64(a) & toInt64(b);
}

template<>
__device__ float bwAnd(float a, float b) {
	return toInt32(a) & toInt32(b);
}

#endif // SPOOF_UTILS_CUH
