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
#ifndef SYSTEMDS_VECTOR_ADD_CUH
#define SYSTEMDS_VECTOR_ADD_CUH

template<typename T, typename Op>
__device__ uint32_t vectAdd_atomic(T* a, T b, T* c, uint32_t ai, uint32_t ci, uint32_t len, Op op) {
	uint tid = threadIdx.x;
	if(tid >= len)
		return len;
	uint i = tid;
	
	while (i < len) {
//		if(blockIdx.x == 0 && threadIdx.x < 4)
//			printf("vectAdd_atomic: bid=%d, tid=%d, ai=%d, ci=%d, len=%d, b=%f, c[%d]=%f, a[%d]=%f\n", blockIdx.x, threadIdx.x, ai,
//				   ci, len, b, ci + i, op(a[ai + i], b), ai+i, a[ai + i]);
		
		atomicAdd(&(c[ci + i]), op(a[ai + i], b));
		i += blockDim.x;
	}
	return len;
}

#endif //SYSTEMDS_VECTOR_ADD_CUH
