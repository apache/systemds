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

#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <sstream>

using clk = std::chrono::high_resolution_clock;
using sec = std::chrono::duration<double, std::ratio<1>>;

size_t SpoofCUDAContext::initialize_cuda(uint32_t device_id, const char* resource_path) {

#ifdef _DEBUG
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
	auto *ctx = new SpoofCUDAContext(resource_path, {s1.str(), s2.str(), cuda_include_path});
	// cuda device is handled by jCuda atm
	//cudaSetDevice(device_id);
	//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	//cudaDeviceSynchronize();
	
	CHECK_CUDA(cuModuleLoad(&(ctx->reductions), std::string(ctx->resource_path + std::string("/cuda/kernels/reduction.ptx")).c_str()));
	
	CUfunction func;
	
	// ToDo: implement a more scalable solution for these imports
	
	// SUM
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_d"));
	ctx->reduction_kernels.insert(std::make_pair("reduce_sum_d", func));
	//  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_f"));
	//  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_f", func));
	//
	//  // SUM_SQ
	//  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_sq_d"));
	//  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_sq_d", func));
	//  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_sq_f"));
	//  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_sq_f", func));
	//
	//  // MIN
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_d"));
	ctx->reduction_kernels.insert(std::make_pair("reduce_min_d", func));
	//  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_f"));
	//  ctx->reduction_kernels.insert(std::make_pair("reduce_min_f", func));
	//
	//  // MAX
	CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_d"));
	ctx->reduction_kernels.insert(std::make_pair("reduce_max_d", func));
	//  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_f"));
	//  ctx->reduction_kernels.insert(std::make_pair("reduce_max_f", func));

	return reinterpret_cast<size_t>(ctx);
}

void SpoofCUDAContext::destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id) {
  delete ctx;
  ctx = nullptr;
  // cuda device is handled by jCuda atm
  //cudaDeviceReset();
}

int SpoofCUDAContext::compile(const std::string &src, const std::string &name, SpoofOperator::OpType op_type,
		SpoofOperator::AggType agg_type, SpoofOperator::AggOp agg_op, SpoofOperator::RowType row_type, bool sparse_safe,
		int32_t const_dim2, uint32_t num_vectors, bool TB1) {
#ifdef _DEBUG
//	std::cout << "---=== START source listing of spoof cuda kernel [ " << name << " ]: " << std::endl;
//    uint32_t line_num = 0;
//	std::istringstream src_stream(src);
//    for(std::string line; std::getline(src_stream, line); line_num++)
//		std::cout << line_num << ": " << line << std::endl;
//
//	std::cout << "---=== END source listing of spoof cuda kernel [ " << name << " ]." << std::endl;
	std::cout << "cwd: " << std::filesystem::current_path() << std::endl;
	std::cout << "include_paths: ";
	for_each (include_paths.begin(), include_paths.end(), [](const std::string& line){ std::cout << line << '\n';});
	std::cout << std::endl;
#endif
// uncomment all related lines for temporary timing output:
//	auto start = clk::now();

//	auto compile_start = clk::now();
	jitify::Program program = kernel_cache.program(src, 0, include_paths);
//	auto compile_end = clk::now();
//	auto compile_duration = std::chrono::duration_cast<sec>(compile_end - compile_start).count();
	ops.insert(std::make_pair(name, SpoofOperator({std::move(program),  op_type, agg_type, agg_op, row_type, name,
			const_dim2, num_vectors, TB1, sparse_safe})));

//	auto end = clk::now();

//	auto handling_duration = std::chrono::duration_cast<sec>(end - start).count() - compile_duration;

//	compile_total += compile_duration;
//	handling_total += handling_duration;

//	std::cout << name << " times [s] handling/compile/totals(h/c)/count: "
//			<< handling_duration << "/"
//			<< compile_duration << "/"
//			<< handling_total << "/"
//			<< compile_total << "/" << compile_count + 1 << std::endl;
	return compile_count++;
}
