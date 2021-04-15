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
	
	static void exec([[maybe_unused]] SpoofCUDAContext* ctx, SpoofOperator* _op, DataBufferWrapper* dbw)  {
		uint32_t NT=256;
		T value_type;
		bool sparse_input = dbw->h_in<T>(0)->row_ptr != nullptr;
		auto* op = dynamic_cast<SpoofRowwiseOp*>(_op);
		dim3 grid(dbw->h_in<T>(0)->rows, 1, 1);
		dim3 block(NT, 1, 1);
		unsigned int shared_mem_size = NT * sizeof(T);

		size_t out_num_elements = dbw->h_out<T>()->rows * dbw->h_out<T>()->cols;
		if(dbw->h_out<T>()->row_ptr)
			if(op->isSparseSafe() && dbw->h_out<T>()->nnz > 0)
				out_num_elements = dbw->h_out<T>()->nnz;
		//ToDo: only memset output when there is an output operation that *adds* to the buffer
		CHECK_CUDART(cudaMemsetAsync(dbw->h_out<T>()->data, 0, out_num_elements * sizeof(T), op->stream));

		//ToDo: handle this in JVM
		uint32_t tmp_len = 0;
		uint32_t temp_buf_size = 0;
		T* d_temp = nullptr;
		if(op->num_temp_vectors > 0) {
			tmp_len = std::max(dbw->h_in<T>(0)->cols, op->const_dim2 < 0 ? 0u : static_cast<uint32_t>(op->const_dim2));
			temp_buf_size = op->num_temp_vectors * tmp_len * dbw->h_in<T>(0)->rows * sizeof(T);
#ifndef NDEBUG
			std::cout << "num_temp_vect: " << op->num_temp_vectors << " temp_buf_size: " << temp_buf_size << " tmp_len: " << tmp_len << std::endl;
#endif
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_temp), temp_buf_size));
			CHECK_CUDART(cudaMemsetAsync(d_temp, 0, temp_buf_size, op->stream));
		}
		
		std::string op_name(op->name + "_DENSE");
		if(sparse_input)
			op_name = std::string(op->name + "_SPARSE");

#ifndef NDEBUG
		// ToDo: connect output to SystemDS logging facilities
		std::cout << "launching spoof rowwise kernel " << op_name << " with " << NT * dbw->h_in<T>(0)->rows << " threads in "
				<< dbw->h_in<T>(0)->rows << " blocks and " << shared_mem_size << " bytes of shared memory for "
				<< dbw->h_in<T>(0)->rows << " cols processed by " << NT << " threads per row, adding "
				<< temp_buf_size / 1024 << " kb of temp buffer in global memory." <<  std::endl;
#endif
		CHECK_CUDA(op->program->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<uint32_t>(1), dbw->num_sides()), op->num_temp_vectors, tmp_len)
						   .configure(grid, block, shared_mem_size, op->stream)
						   .launch(dbw->d_in<T>(0), dbw->d_sides<T>(), dbw->d_out<T>(), dbw->d_scalars<T>(), d_temp, dbw->grix()));
		
		if(op->num_temp_vectors > 0)
			CHECK_CUDART(cudaFree(d_temp));
	}
};

#endif //SYSTEMDS_SPOOFROWWISE_H
