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

// RowType: %TYPE%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"

template<typename T>
struct SpoofRowwiseOp {
	T* a;
	T**b;
	T* c;
	T* scalars;
	int len, grix_;

	SpoofRowwiseOp(T* a, T** b, T* scalars, T* c, int len, int grix) : a(a), b(b), c(c), scalars(scalars), len(len), grix_(grix) {}

	__device__  __forceinline__ void operator()(int ai, int ci, int rix) const {
		
%BODY_dense%
	}
};

template<typename T>
__global__ void %TMP% (T* a, T** b, T* scalars, T* c, uint c_n, int len, int grix) {
	SpoofRowwiseOp<T> spoof_op(a, b, scalars, c, len, grix);
	int ai = blockIdx.x * len;
	int ci = blockIdx.x * c_n;
	spoof_op(ai, ci, blockIdx.x);
};
