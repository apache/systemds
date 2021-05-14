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
#ifndef SYSTEMDS_OPERATORS_CUH
#define SYSTEMDS_OPERATORS_CUH

template<typename T>
struct SignOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return signbit(a) == 0 ? 1.0 : -1.0;;
	}
};

template<typename T>
struct AbsOp {
	__device__  __forceinline__ static T exec(T a, T b);
};

template<>
struct AbsOp<double> {
	__device__  __forceinline__ static double exec(double a, double b) {
		return fabs(a);
	}
};

template<typename T>
struct RoundOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return round(a);
	}
};

template<typename T>
struct FloorOp {
	__device__  __forceinline__ static T exec(T a, T b);
};

template<>
struct FloorOp<double> {
	__device__  __forceinline__ static double exec(double a, double b) {
		return floor(a);
	}
};

template<>
struct FloorOp<float> {
	__device__  __forceinline__ static float exec(float a, float b) {
		return floorf(a);
	}
};

template<typename T>
struct CeilOp {
	__device__  __forceinline__ static T exec(T a, T b);
};

template<>
struct CeilOp<double> {
	__device__  __forceinline__ static double exec(double a, double b) {
		return ceil(a);
	}
};

template<>
struct CeilOp<float> {
	__device__  __forceinline__ static float exec(float a, float b) {
		return ceilf(a);
	}
};

template<typename T>
struct ExpOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return expf(a);
	}
};

template<typename T>
struct EqualOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return (a == b) ? 1.0 : 0.0;
	}
};

template<typename T>
struct NotEqualOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return (a != b) ? 1.0 : 0.0;
	}
};

template<typename T>
struct NotZero {
	__device__  __forceinline__ T operator()(T a, T b) const {
		return (a != 0) ? 1.0 : 0.0;
	}
	
	__device__  __forceinline__ static T exec(T a, T b) {
		return (a != 0) ? 1.0 : 0.0;
	}
};

template<typename T>
struct XorOp {
	__device__  __forceinline__ static T exec(T a, T b) {
		return (a != 0.0) != (b != 0.0) ? 1.0 : 0.0;
	}
};

template<typename T>
struct GreaterEqualOp {
    __device__  __forceinline__ static T exec(T a, T b) {
        return (a >= b) ? 1.0 : 0.0;
    }
};

template<typename T>
struct LessEqualOp {
    __device__  __forceinline__ static T exec(T a, T b) {
        return (a <= b) ? 1.0 : 0.0;
    }
};

template<typename T>
struct GreaterOp {
    __device__  __forceinline__ static T exec(T a, T b) {
        return (a > b) ? 1.0 : 0.0;
    }
};

template<typename T>
struct Pow2Op {
	__device__  __forceinline__ static T exec(T a, T b) {
		return a * a;
	}
};

#endif //SYSTEMDS_OPERATORS_CUH
