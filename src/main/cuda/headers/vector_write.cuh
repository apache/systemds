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

__device__ bool debug_row() { return blockIdx.x == 1; };
__device__ bool debug_thread() { return threadIdx.x == 0; }

// unary transform vector by OP and write to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteUnary(T* a, uint32_t ai, uint32_t len, TempStorage<T>* fop, const char* name = nullptr) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < len) {
		c[i] = OP::exec(a[ai + i], 0); //ToDo: remove b from all unary ops
//		if (blockIdx.x == 0) {
//			printf("DvecWriteUnary->tmp: bid=%d, tid=%d, len=%d, ai=%d, a[%d]=%4.3f, c[%d]=%4.3f\n",
//					blockIdx.x, threadIdx.x, len, ai, ai + i, a[ai + i], i, c[i]);
//		}
		i += blockDim.x;
	}
	return c;
}

// unary transform vector by OP and write to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteUnary(T* a, uint32_t* aix, uint32_t ai, uint32_t alen, uint32_t len, TempStorage<T>* fop, const char* name = nullptr) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < alen) {
		c[aix[i]] = OP::exec(a[ai + i], 0); //ToDo: remove b from all unary ops
//		if (debug_row() && debug_thread()) {
//		if (debug_row() && ((threadIdx.x < 10) || (threadIdx.x > 40 && threadIdx.x < 50))) {
//			const char* name_ = "";
//			if(name != nullptr)
//				name_ = name;
//			printf("SvecWriteUnary(%s)->tmp: bid=%d, tid=%d, len=%d, a[%d+%d=%d]=%4.3f, c[%d]=%4.3f\n",
//				   name_, blockIdx.x, threadIdx.x, len, ai, i, ai + i, a[ai + i], aix[i], c[aix[i]]);
//		}
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
__device__ Vector<T>& vectWriteBinary(T a, T* b, uint32_t bi, uint32_t len, TempStorage<T>* fop) {
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
__device__ Vector<T>& vectWriteBinary(T* a, T b, uint32_t ai, uint32_t len, TempStorage<T>* fop, const char* name = nullptr) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < len) {
		c[i] = OP::exec(a[ai + i], b);
//		if (debug_row() && ((threadIdx.x < 10) || (threadIdx.x > 40 && threadIdx.x < 50))) {
//		if (debug_row() && threadIdx.x > 40 && threadIdx.x < 50) {
//		if (debug_row() && threadIdx.x < 10) {
//			const char* name_ = "";
//			if(name != nullptr)
//				name_ = name;
//			printf("DvecWriteBinary(%s)->tmp vs: bid=%d, tid=%d, len=%d, b=%4.3f, a[%d+%d=%d]=%4.3f, c[%d]=%4.3f\n",
//				   name_, blockIdx.x, threadIdx.x, len, b, ai, i, ai + i, a[ai + i], i, c[i]);
//		}
		i += blockDim.x;
	}
	return c;
}

// binary vect-scalar to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteBinary(T* a, T b, uint32_t* aix, uint32_t ai, uint32_t alen, uint32_t len, TempStorage<T>* fop, const char* name = nullptr) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	while (i < alen) {
		c[aix[i]] = OP::exec(a[ai + i], b);
//		if (debug_row() && debug_thread()) {
////		if (debug_row() && ((threadIdx.x < 10) || (threadIdx.x > 40 && threadIdx.x < 50))) {
//			const char* name_ = "";
//			if(name != nullptr)
//				name_ = name;
//			printf("SvecWriteBinary(%s)->tmp vs: bid=%d, tid=%d, len=%d, b=%4.3f, a[%d]=%4.3f, c[aix[%d]=%d]=%4.3f\n",
//				   name_, blockIdx.x, threadIdx.x, len, b, ai + i, a[ai + i], i, aix[i], c[aix[i]]);
//		}
		i += blockDim.x;
	}
	return c;
}

// bianry vector-vector to intermediate vector
template<typename T, typename OP>
__device__ Vector<T>& vectWriteBinary(T* a, T* b, uint32_t ai, uint32_t bi, uint32_t len, TempStorage<T>* fop, const char* name = nullptr) {
	uint32_t i = threadIdx.x;
	Vector<T>& c = fop->getTempStorage(len);
	
	while (i < len) {
		c[i] = OP::exec(a[ai + i], b[bi + i]);
//		if (debug_row() && debug_thread()) {
//			const char* name_ = "";
//			if(name != nullptr)
//				name_ = name;
//			printf("DvecWriteBinary(%s)->tmp vv: bid=%d, tid=%d, len=%d, a[%d]=%4.3f, b[%d]=%4.3f, c[%d]=%4.3f\n",
//				   name_, blockIdx.x, threadIdx.x, len, ai + i, a[ai + i], bi+i, b[bi+i], i, c[i]);
//		}
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
//		if (debug_row() && debug_thread()) {
//			printf("DvecWriteBinary->out vv: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, a[%d]=%4.3f, b=%4.3f, c[%d]=%4.3f\n",
//				   blockIdx.x, threadIdx.x, ai, ci, len, ai + i, a[ai + i], b, ci+i, c[ci+i]);
//		}
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
