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

//	double handling_total, compile_total;
	
	const std::string resource_path;
	const std::vector<std::string> include_paths;
	
public:

	explicit SpoofCUDAContext(const char* resource_path_, std::vector<std::string>  include_paths_) : reductions(nullptr),
			resource_path(resource_path_), include_paths(std::move(include_paths_))
			//,handling_total(0.0), compile_total(0.0)
			{}

	static size_t initialize_cuda(uint32_t device_id, const char* resource_path_);

	static void destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id);
	
	size_t compile(std::unique_ptr<SpoofOperator> op, const std::string &src);
	
	template <typename T, typename CALL>
	int launch(uint32_t opID, std::vector<Matrix<T>>& input, std::vector<Matrix<T>>& sides, Matrix<T>& output,
			T* scalars, uint32_t grix) {
		// dp holds in/side/out/scalar pointers for GPU
		DevMatPtrs<T> dp;

		SpoofOperator* op = compiled_ops[opID].get();
		
		CHECK_CUDART(cudaMalloc((void **)&dp.in, sizeof(Matrix<T>) * input.size()));
		CHECK_CUDART(cudaMemcpy(dp.in, reinterpret_cast<void*>(&input[0]), sizeof(Matrix<T>) * input.size(),
				cudaMemcpyHostToDevice));

		if (!sides.empty()) {
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&dp.sides), sizeof(Matrix<T>) * sides.size()));
			CHECK_CUDART(cudaMemcpy(dp.sides, &sides[0], sizeof(Matrix<T>)  * sides.size(), cudaMemcpyHostToDevice));
		}
		
		if (op->isSparseSafe() && input.front().row_ptr != nullptr) {
			CHECK_CUDART(cudaMemcpy(output.row_ptr, input.front().row_ptr, (input.front().rows+1)*sizeof(uint32_t),
					cudaMemcpyDeviceToDevice));
		}
#ifndef NDEBUG
		std::cout << "output rows: " << output.rows << " cols: " << output.cols << " nnz: " << output.nnz << " format: " <<
				(output.row_ptr == nullptr ? "dense" : "sparse") << std::endl;
#endif
		size_t out_num_elements = output.rows * output.cols;
		if(output.row_ptr)
			if(op->isSparseSafe() && output.nnz > 0)
				out_num_elements = output.nnz;
		CHECK_CUDART(cudaMalloc((void **) &dp.out, sizeof(Matrix<T>)));
		CHECK_CUDART(cudaMemset(output.data, 0, out_num_elements * sizeof(T)));
		CHECK_CUDART(cudaMemcpy(dp.out, reinterpret_cast<void *>(&output), sizeof(Matrix<T>),
				cudaMemcpyHostToDevice));
		
		dp.scalars = scalars;

		CALL::exec(this, op, input, sides, output, grix, dp);
		
		return 0;
	}
	
	std::string getOperatorName(uint32_t opID) { return compiled_ops.at(opID)->name; }

	template<typename T>
	CUfunction getReductionKernel(const std::pair<SpoofOperator::AggType, SpoofOperator::AggOp>& key);
};

#endif // SPOOFCUDACONTEXT_H
