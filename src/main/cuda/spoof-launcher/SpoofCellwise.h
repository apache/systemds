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
#ifndef SYSTEMDS_SPOOFCELLWISE_H
#define SYSTEMDS_SPOOFCELLWISE_H

#include "SpoofCUDAContext.h"
#include <algorithm>

template<typename T>
struct SpoofCellwiseFullAgg {
	
	static void exec(SpoofCellwiseOp* op, uint32_t NT, uint32_t N, const std::string& op_name,
			std::vector<Matrix<T>>& sides, uint32_t grix,  DevMatPtrs<T>& dp) {
		T value_type;
		
		// num ctas
		uint32_t NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
		dim3 grid(NB, 1, 1);
		dim3 block(NT, 1, 1);
		uint32_t shared_mem_size = NT * sizeof(T);
#ifndef NDEBUG
		// ToDo: connect output to SystemDS logging facilities
				std::cout << "launching spoof cellwise kernel " << op_name << " with "
						  << NT * NB << " threads in " << NB << " blocks and "
						  << shared_mem_size
						  << " bytes of shared memory for full aggregation of "
						  << N << " elements"
						  << std::endl;
#endif
		CHECK_CUDA(op->program.get()->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<size_t>(1), sides.size()))
						   .configure(grid, block, shared_mem_size)
						   .launch(dp.in, dp.sides, dp.out, dp.scalars, N, grix));
		
		if(NB > 1) {
			N = NB;
			while (NB > 1) {
				void* args[3] = { &dp.out, &dp.out, &N};
				
				NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
#ifndef NDEBUG
				std::cout << " launching spoof cellwise kernel " << op_name << " with "
                    << NT * NB << " threads in " << NB << " blocks and "
                    << shared_mem_size
                    << " bytes of shared memory for full aggregation of "
                    << N << " elements"
                    << std::endl;
#endif
				CHECK_CUDA(cuLaunchKernel(op->agg_kernel,NB, 1, 1, NT, 1, 1, shared_mem_size, nullptr, args, nullptr));
				N = NB;
			}
		}
	}
};


template<typename T>
struct SpoofCellwiseRowAgg {
	static void exec(SpoofOperator *op, uint32_t NT, uint32_t N, const std::string &op_name,
			  std::vector<Matrix<T>> &input, std::vector<Matrix<T>> &sides, uint32_t grix, DevMatPtrs<T>& dp) {
		T value_type;
		
		// num ctas
		uint32_t NB = input.front().rows;
		dim3 grid(NB, 1, 1);
		dim3 block(NT, 1, 1);
		uint32_t shared_mem_size = NT * sizeof(T);
#ifndef NDEBUG
		std::cout << " launching spoof cellwise kernel " << op_name << " with "
					<< NT * NB << " threads in " << NB << " blocks and "
					<< shared_mem_size << " bytes of shared memory for row aggregation of "
					<< N << " elements" << std::endl;
#endif
		CHECK_CUDA(op->program->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<size_t>(1), sides.size()))
						   .configure(grid, block, shared_mem_size)
						   .launch(dp.in, dp.sides, dp.out, dp.scalars, N, grix));
		
	}
};


template<typename T>
struct SpoofCellwiseColAgg {
	static void exec(SpoofOperator* op, uint32_t NT, uint32_t N, const std::string& op_name,
					 std::vector<Matrix<T>>& sides, uint32_t grix, DevMatPtrs<T>& dp) {
		T value_type;
		
		// num ctas
		uint32_t NB = std::ceil((N + NT - 1) / NT);
		
		dim3 grid(NB,1, 1);
		dim3 block(NT,1, 1);
		uint32_t shared_mem_size = 0;
#ifndef NDEBUG
		std::cout << " launching spoof cellwise kernel " << op_name << " with "
						<< NT * NB << " threads in " << NB << " blocks for column aggregation of "
						<< N << " elements" << std::endl;
#endif
		CHECK_CUDA(op->program->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<size_t>(1), sides.size()))
						   .configure(grid, block, shared_mem_size)
						   .launch(dp.in, dp.sides, dp.out, dp.scalars, N, grix));
		
	}
};


template<typename T>
struct SpoofCellwiseNoAgg {
	static void exec(SpoofOperator *op, uint32_t NT, uint32_t N, const std::string &op_name,
			  std::vector<Matrix<T>> &input, std::vector<Matrix<T>> &sides, uint32_t grix, DevMatPtrs<T>& dp) {
		T value_type;
		bool sparse_input = input.front().row_ptr != nullptr;
		
		// num ctas
		// ToDo: adaptive VT
		const uint32_t VT = 1;
		uint32_t NB = std::ceil((N + NT * VT - 1) / (NT * VT));
		if(sparse_input)
			NB = input.front().rows;
		dim3 grid(NB, 1, 1);
		dim3 block(NT, 1, 1);
		uint32_t shared_mem_size = 0;

#ifndef NDEBUG
		if(sparse_input) {
				std::cout << "launching sparse spoof cellwise kernel " << op_name << " with " << NT * NB
						  << " threads in " << NB << " blocks without aggregation for " << N << " elements"
						  << std::endl;
		}
		else {
			std::cout << "launching spoof cellwise kernel " << op_name << " with " << NT * NB
					  << " threads in " << NB << " blocks without aggregation for " << N << " elements"
					  << std::endl;
		}
#endif
		
		CHECK_CUDA(op->program->kernel(op_name)
						   .instantiate(type_of(value_type), std::max(static_cast<size_t>(1), sides.size()))
						   .configure(grid, block, shared_mem_size)
						   .launch(dp.in, dp.sides, dp.out, dp.scalars, N, grix));
	}
};

template<typename T>
struct SpoofCellwise {
	static void exec(SpoofCUDAContext* ctx, SpoofOperator* _op, std::vector<Matrix<T>>& input,
			std::vector<Matrix<T>>& sides, Matrix<T>& output, uint32_t grix,
			DevMatPtrs<T>& dp)  {
		
		T value_type;
		auto* op = dynamic_cast<SpoofCellwiseOp*>(_op);
		bool sparse_input = input.front().row_ptr != nullptr;
		uint32_t NT = 256; // ToDo: num threads
		uint32_t N = input.front().rows * input.front().cols;
		std::string op_name(op->name + "_DENSE");
		if(sparse_input) {
			op_name = std::string(op->name + "_SPARSE");
			if(op->isSparseSafe() && input.front().nnz > 0)
				N = input.front().nnz;
		}
		
		switch(op->agg_type) {
			case SpoofOperator::AggType::FULL_AGG:
				op->agg_kernel = ctx->template getReductionKernel<T>(std::make_pair(op->agg_type, op->agg_op));
				SpoofCellwiseFullAgg<T>::exec(op, NT, N, op_name, sides, grix, dp);
				break;
			case SpoofOperator::AggType::ROW_AGG:
				SpoofCellwiseRowAgg<T>::exec(op, NT, N, op_name, input, sides, grix, dp);
				break;
			case SpoofOperator::AggType::COL_AGG:
				SpoofCellwiseColAgg<T>::exec(op, NT, N, op_name, sides, grix, dp);
				break;
			case SpoofOperator::AggType::NO_AGG:
				SpoofCellwiseNoAgg<T>::exec(op, NT, N, op_name, input, sides, grix, dp);
				break;
			default:
				throw std::runtime_error("unknown cellwise agg type" + std::to_string(static_cast<int>(op->agg_type)));
		}
	}
};


#endif //SYSTEMDS_SPOOFCELLWISE_H
