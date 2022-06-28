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

#include "SpoofCUDAContext.h"

//#include <filesystem>
#include <iostream>
#include <cstddef>
#include <cstdlib>

//#include <sstream>
//using clk = std::chrono::high_resolution_clock;
//using sec = std::chrono::duration<double, std::ratio<1>>;

size_t SpoofCUDAContext::initialize_cuda(uint32_t device_id, const char* resource_path) {

#ifndef NDEBUG
	std::cout << "initializing cuda device " << device_id << std::endl;
#endif
	std::string cuda_include_path;
	
	char* cdp = std::getenv("CUDA_PATH");
	if(cdp != nullptr)
		cuda_include_path = std::string("-I") + std::string(cdp) + "/include";
	else {
		std::cout << "Warning: CUDA_PATH environment variable not set. Using default include path "
					 "/usr/local/cuda/include" << std::endl;
		cuda_include_path = std::string("-I/usr/local/cuda/include");
	}
	
	std::stringstream s1, s2;
	s1 << "-I" << resource_path << "/cuda/headers";
	s2 << "-I" << resource_path << "/cuda/spoof";
	auto ctx = new SpoofCUDAContext(resource_path,{s1.str(), s2.str(), cuda_include_path});

	CHECK_CUDA(cuModuleLoad(&(ctx->reductions), std::string(ctx->resource_path + std::string("/cuda/kernels/reduction.ptx")).c_str()));
	
	CUfunction func;
	
	// SUM and SUM_SQ have the same behavior for intermediate buffers (squaring is done in the initial reduction step,
	// after that it is just summing up the temporary data)
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_f"));
	ctx->reduction_kernels_f.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::SUM), func));
	ctx->reduction_kernels_f.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::SUM_SQ), func));
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_d"));
	ctx->reduction_kernels_d.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::SUM), func));
	ctx->reduction_kernels_d.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::SUM_SQ), func));

	// MIN
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_f"));
	ctx->reduction_kernels_f.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::MIN), func));
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_d"));
	ctx->reduction_kernels_d.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::MIN), func));
	
	// MAX
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_f"));
	ctx->reduction_kernels_f.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::MAX), func));
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_d"));
	ctx->reduction_kernels_d.insert(std::make_pair(std::make_pair(SpoofOperator::AggType::FULL_AGG, SpoofOperator::AggOp::MAX), func));
	
	CHECK_CUDART(cudaMallocHost(reinterpret_cast<void**>(&(ctx->staging_buffer)), ctx->default_mem_size));
	CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&(ctx->device_buffer)), ctx->default_mem_size));
	ctx->current_mem_size = ctx->default_mem_size;
	return reinterpret_cast<size_t>(ctx);
}

void SpoofCUDAContext::destroy_cuda(SpoofCUDAContext *ctx, [[maybe_unused]] uint32_t device_id) {
	cudaFreeHost(ctx->staging_buffer);
	cudaFree(ctx->device_buffer);
	delete ctx;
}

size_t SpoofCUDAContext::compile(std::unique_ptr<SpoofOperator> op, const std::string &src) {
//	auto compile_start = clk::now();
	op->program = std::make_unique<jitify::Program>(kernel_cache.program(src, 0, include_paths));
//	auto compile_end = clk::now();
//	auto compile_duration = std::chrono::duration_cast<sec>(compile_end - compile_start).count();
	compiled_ops.push_back(std::move(op));
//	compile_total += compile_duration;
//	std::cout << name << " compiled in "
//			<< compile_duration << " seconds. Total compile time (abs/rel): "
//			<< compile_total << "/" << compiled_ops.size() << std::endl;
	return compiled_ops.size() - 1;
}

template<typename T>
CUfunction SpoofCUDAContext::getReductionKernel(const std::pair<SpoofOperator::AggType, SpoofOperator::AggOp> &key) {
	return nullptr; // generic case never used
}

template<>
CUfunction SpoofCUDAContext::getReductionKernel<float>(const std::pair<SpoofOperator::AggType,
		SpoofOperator::AggOp> &key) {
	return reduction_kernels_f[key];
}

template<>
CUfunction SpoofCUDAContext::getReductionKernel<double>(const std::pair<SpoofOperator::AggType,
		SpoofOperator::AggOp> &key) {
	return reduction_kernels_d[key];
}

void SpoofCUDAContext::resize_staging_buffer(size_t size) {
	cudaFreeHost(staging_buffer);
	cudaFree(device_buffer);
	CHECK_CUDART(cudaMallocHost(reinterpret_cast<void**>(&(staging_buffer)), size));
	CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&(device_buffer)), size));
	current_mem_size = size;
}
