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
#ifndef SYSTEMDS_SPOOFROWWISE_H
#define SYSTEMDS_SPOOFROWWISE_H

#include "SpoofCUDAContext.h"
#include <algorithm>

template <typename T>
struct SpoofRowwise {
	
	static void exec([[maybe_unused]] SpoofCUDAContext* ctx, SpoofOperator* _op, std::vector<Matrix<T>>& input,
			std::vector<Matrix<T>>& sides, Matrix<T>& output, uint32_t grix, DevMatPtrs<T>& dp)  {
		uint32_t NT=256;
		T value_type;
		bool sparse_input = input.front().row_ptr != nullptr;
		auto* op = dynamic_cast<SpoofRowwiseOp*>(_op);
		dim3 grid(input.front().rows, 1, 1);
		dim3 block(NT, 1, 1);
		unsigned int shared_mem_size = NT * sizeof(T);
		
		uint32_t tmp_len = 0;
		uint32_t temp_buf_size = 0;
		T* d_temp = nullptr;
		if(op->num_temp_vectors > 0) {
			tmp_len = std::max(input.front().cols, op->const_dim2 < 0 ? 0 : static_cast<uint32_t>(op->const_dim2));
			temp_buf_size = op->num_temp_vectors * tmp_len * input.front().rows * sizeof(T);
#ifndef NDEBUG
			std::cout << "num_temp_vect: " << op->num_temp_vectors << " temp_buf_size: " << temp_buf_size << " tmp_len: " << tmp_len << std::endl;
#endif
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_temp), temp_buf_size));
			CHECK_CUDART(cudaMemset(d_temp, 0, temp_buf_size));
		}
		
		std::string op_name(op->name + "_DENSE");
		if(sparse_input)
			op_name = std::string(op->name + "_SPARSE");

#ifndef NDEBUG
		// ToDo: connect output to SystemDS logging facilities
		std::cout << "launching spoof rowwise kernel " << op_name << " with " << NT * input.front().rows << " threads in "
				<< input.front().rows << " blocks and " << shared_mem_size << " bytes of shared memory for "
				<< input.front().rows << " cols processed by " << NT << " threads per row, adding "
				<< temp_buf_size / 1024 << " kb of temp buffer in global memory." <<  std::endl;
#endif
		CHECK_CUDA(op->program->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<size_t>(1), sides.size()), op->num_temp_vectors, tmp_len)
						   .configure(grid, block, shared_mem_size)
						   .launch(dp.in, dp.sides, dp.out, dp.scalars, d_temp, grix));
		
		if(op->num_temp_vectors > 0)
			CHECK_CUDART(cudaFree(d_temp));
		
//		if (op->TB1)
//			CHECK_CUDART(cudaFree(b1_transposed));
	}
};

#endif //SYSTEMDS_SPOOFROWWISE_H
