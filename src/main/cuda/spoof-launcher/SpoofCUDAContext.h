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
#ifndef SPOOFCUDACONTEXT_H
#define SPOOFCUDACONTEXT_H

#if defined(_WIN32) || defined(_WIN64)
	#define NOMINMAX
#endif

//#ifndef NDEBUG
//	#define JITIFY_PRINT_ALL 1
//#endif

#include <map>
#include <string>
#include <utility>
#include <cublas_v2.h>
#include <jitify.hpp>
#include "SpoofOperator.h"
#include "Matrix.h"

using jitify::reflection::type_of;

class SpoofCUDAContext {
	jitify::JitCache kernel_cache;
	std::vector<std::unique_ptr<SpoofOperator>> compiled_ops;
	CUmodule reductions;
	std::map<std::pair<SpoofOperator::AggType, SpoofOperator::AggOp>, CUfunction> reduction_kernels_f;
	std::map<std::pair<SpoofOperator::AggType, SpoofOperator::AggOp>, CUfunction> reduction_kernels_d;

	const std::string resource_path;
	const std::vector<std::string> include_paths;
	
public:
	size_t default_mem_size = 1024; // 1kb for hosting data pointers, scalars and some meta info. This default should
									// not require resizing these buffers in most cases.
	size_t current_mem_size = 0; // the actual staging buffer size (should be default unless there was a resize)
	std::byte* staging_buffer{}; // pinned host mem for async transfers
	std::byte* device_buffer{};  // this buffer holds the pointers to the data buffers

	explicit SpoofCUDAContext(const char* resource_path_, std::vector<std::string>  include_paths_) : reductions(nullptr),
			resource_path(resource_path_), include_paths(std::move(include_paths_)) { }

	static size_t initialize_cuda(uint32_t device_id, const char* resource_path_);

	static void destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id);

	size_t compile(std::unique_ptr<SpoofOperator> op, const std::string &src);

	template <typename T, typename CALL>
	int launch() {

		DataBufferWrapper dbw(staging_buffer, device_buffer);
		SpoofOperator* op = compiled_ops[dbw.op_id()].get();
		dbw.toDevice(op->stream);

		CALL::exec(this, op, &dbw);

		return 0;
	}
	
	std::string getOperatorName(uint32_t opID) { return compiled_ops.at(opID)->name; }

	template<typename T>
	CUfunction getReductionKernel(const std::pair<SpoofOperator::AggType, SpoofOperator::AggOp>& key);

	void resize_staging_buffer(size_t size);
};

#endif // SPOOFCUDACONTEXT_H
