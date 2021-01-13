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
#ifndef SYSTEMDS_VECTOR_WRITE_CUH
#define SYSTEMDS_VECTOR_WRITE_CUH


// unary transform vector by OP and write to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteUnary(T* a, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
//	if(blockIdx.x == 1 && threadIdx.x < 2)
//		printf("vecWrite_sv: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, c[%d]=%f\n", blockIdx.x, threadIdx.x, ai, ci, len, ci * len + threadIdx.x, OP::exec(a, b[ai + i]););
	
	while (i < len) {
		c[i] = OP::exec(a[ai + i], 0); //ToDo: remove b from all unary ops
		i += blockDim.x;
	}
	return c;
}

// unary transform vector by OP and write to output vector c
template<typename T, typename OP>
__device__ void vectWriteUnary(T* a, T* c, uint32_t ai, uint32_t ci, uint32_t len) {
	uint32_t i = threadIdx.x;
	while (i < len) {
		c[ci + i] = OP::exec(a[ai + i], 0);
		i += blockDim.x;
	}
}

// binary scalar-vector to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteBinary(T a, T* b, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < len) {
		c[i] = OP::exec(a, b[bi + i]);
		i += blockDim.x;
	}
	return c;
}

// binary vect-scalar to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteBinary(T* a, T b, uint32_t ai, uint32_t len, SpoofOp<T>* fop) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < len) {
		c[i] = OP::exec(a[ai + i], b);
		i += blockDim.x;
	}
	return c;
}

// bianry vector-vector to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteBinary(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len, SpoofOp<T>* fop) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	
	while (i < len) {
		c[i] = OP::exec(a[ai + i], b[bi + i]);
		i += blockDim.x;
	}
	return c;
}

// binary vector-scalar to output vector c
template<typename T, typename OP>
__device__ void vectWriteBinary(T* a, T b, T* c, uint32_t ai, uint32_t ci, uint32_t len) {
	uint32_t i = threadIdx.x;
	while (i < len) {
		c[ci + i] = OP::exec(a[ai + i], b);
		i += blockDim.x;
	}
}

// binary vector-vector to output vector c
template<typename T, typename OP>
__device__ void vectWriteBinary(T* a, T* b, T* c, uint32_t ai, uint32_t bi, uint32_t ci, uint32_t len) {
	uint32_t i = threadIdx.x;
	const uint32_t& bid = blockIdx.x;
	
	while (i < len) {
		c[ci + i] = OP::exec(a[ai + i], b[bi + i]);
		i += blockDim.x;
	}
}

#endif //SYSTEMDS_VECTOR_WRITE_CUH
