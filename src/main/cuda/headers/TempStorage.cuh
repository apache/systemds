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
#ifndef SYSTEMDS_TEMPSTORAGE_CUH
#define SYSTEMDS_TEMPSTORAGE_CUH

template<typename T>
struct TempStorage {
	__device__ virtual  Vector<T>& getTempStorage(uint32_t len) = 0;
};

template<typename T, uint32_t NUM_TMP_VECT, uint32_t TMP_VECT_LEN >
struct TempStorageImpl : public TempStorage<T> {
	RingBuffer<T,NUM_TMP_VECT> temp_rb;
	
	TempStorageImpl(T* tmp_stor) {
		if(tmp_stor) {
			uint32_t tmp_row_offset = TMP_VECT_LEN * NUM_TMP_VECT * blockIdx.x;
			temp_rb.init(tmp_row_offset, TMP_VECT_LEN, tmp_stor);
		}
	}
	
	__device__ Vector<T>& getTempStorage(uint32_t len) {
//		if(debug_row() && debug_thread())
//			printf("getTempStorage(len=%d)\n", len);
		Vector<T>& vec = temp_rb.next();
		vec.length = len;
		return vec;
	}
};

#endif //SYSTEMDS_TEMPSTORAGE_CUH
