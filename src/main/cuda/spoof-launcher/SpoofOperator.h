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
#ifndef SYSTEMDS_SPOOFOPERATOR_H
#define SYSTEMDS_SPOOFOPERATOR_H

#include <cmath>
#include <cstdint>
#include <string>
#include <jitify.hpp>
#include "host_utils.h"
#include "Matrix.h"

// these two constants have equivalents in Java code:
const uint32_t JNI_MAT_ENTRY_SIZE = 40;
const uint32_t TRANSFERRED_DATA_HEADER_SIZE = 32;

struct SpoofOperator {
	enum class AggType : int { NO_AGG, FULL_AGG, ROW_AGG, COL_AGG };
	enum class AggOp : int { SUM, SUM_SQ, MIN, MAX };
	enum class RowType : int { FULL_AGG = 4 };

	std::string name;
	std::unique_ptr<jitify::Program> program;
	
	[[nodiscard]] virtual bool isSparseSafe() const = 0;

	cudaStream_t stream{};
	
	SpoofOperator() { CHECK_CUDART(cudaStreamCreate(&stream));}
	virtual ~SpoofOperator() {CHECK_CUDART(cudaStreamDestroy(stream));}
};

struct SpoofCellwiseOp : public SpoofOperator {
	bool sparse_safe;
	AggType agg_type;
	AggOp agg_op;
	CUfunction agg_kernel{};
	SpoofCellwiseOp(AggType at, AggOp ao, bool ss) : agg_type(at), agg_op(ao), sparse_safe(ss) {}
	
	[[nodiscard]] bool isSparseSafe() const override { return sparse_safe; }
};

struct SpoofRowwiseOp : public SpoofOperator {
	bool TB1 = false;
	uint32_t num_temp_vectors;
	int32_t const_dim2;
	RowType row_type;
	
	SpoofRowwiseOp(RowType rt, bool tb1, uint32_t ntv, int32_t cd2) : row_type(rt), TB1(tb1), num_temp_vectors(ntv),
			const_dim2(cd2) {}
			
	[[nodiscard]] bool isSparseSafe() const override { return false; }
};

struct DataBufferWrapper {
	std::byte* staging_buffer;
	std::byte* device_buffer;


	template<typename T>
	Matrix<T>* in(std::byte* buffer, uint32_t idx) {
		return reinterpret_cast<Matrix<T>*>(&buffer[TRANSFERRED_DATA_HEADER_SIZE + idx * JNI_MAT_ENTRY_SIZE]);
	}

	template<typename T>
	Matrix<T>* sides(std::byte* buffer) {
		return reinterpret_cast<Matrix<T>*>(&buffer[TRANSFERRED_DATA_HEADER_SIZE + num_inputs() * JNI_MAT_ENTRY_SIZE]);
	}

	template<typename T>
	Matrix<T>* out(std::byte* buffer) {
		return reinterpret_cast<Matrix<T>*>(&buffer[TRANSFERRED_DATA_HEADER_SIZE + (num_inputs() + num_sides())
			* JNI_MAT_ENTRY_SIZE]);
	}

	template<typename T>
	T* scalars(std::byte* buffer, uint32_t idx) {
		return reinterpret_cast<T*>(&(buffer[TRANSFERRED_DATA_HEADER_SIZE + (num_inputs() + num_sides() + 1)
			* JNI_MAT_ENTRY_SIZE + idx * sizeof(T)]));
	}

public:
	explicit DataBufferWrapper(std::byte* staging, std::byte* dev_buf) : staging_buffer(staging),
		device_buffer(dev_buf) { }

	void toDevice(cudaStream_t &stream) const {
		CHECK_CUDART(cudaMemcpyAsync(device_buffer, staging_buffer, *reinterpret_cast<uint32_t*>(&staging_buffer[0]),
			cudaMemcpyHostToDevice, stream));
	}

	template<typename T>
	Matrix<T>* d_in(uint32_t num) { return in<T>(device_buffer, num); }

	template<typename T>
	Matrix<T>* h_in(uint32_t num) { return in<T>(staging_buffer, num); }

	template<typename T>
	Matrix<T>* d_sides() { return sides<T>(device_buffer); }

	template<typename T>
	Matrix<T>* h_sides() { return sides<T>(staging_buffer); }

	template<typename T>
	Matrix<T>* d_out() { return out<T>(device_buffer); }

	template<typename T>
	Matrix<T>* h_out() { return out<T>(staging_buffer); }

	template<typename T>
	T* d_scalars(uint32_t idx = 0) { return scalars<T>(device_buffer, idx); }

	template<typename T>
	T* h_scalars(uint32_t idx = 0) { return scalars<T>(staging_buffer, idx); }

	uint32_t op_id() const {
		return *reinterpret_cast<uint32_t*>(&staging_buffer[sizeof(int)]);
	}

	uint64_t grix() const {
		return *reinterpret_cast<uint64_t*>(&staging_buffer[2 * sizeof(int)]);
	}

	[[nodiscard]] uint32_t num_inputs() const {
		return *reinterpret_cast<uint32_t*>(&staging_buffer[3 * sizeof(int)]);
	}

	uint32_t num_sides() const {
		return *reinterpret_cast<uint32_t*>(&staging_buffer[4 * sizeof(int)]);
	}

	uint32_t num_scalars() const {
		return *reinterpret_cast<uint32_t*>(&staging_buffer[6 * sizeof(int)]);
	}
};

#endif //SYSTEMDS_SPOOFOPERATOR_H
