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

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <algorithm>

#include "Matrix.h"

#ifndef NDEBUG
#define _DEBUG
#endif
#ifdef _DEBUG
     #define JITIFY_PRINT_ALL 1
#endif

#include <jitify.hpp>
#include <utility>
#include <cublas_v2.h>

#include "host_utils.h"

using jitify::reflection::type_of;

struct SpoofOperator {
	enum class OpType : int { CW, RA, MA, OP, NONE };
	enum class AggType : int { NONE, NO_AGG, FULL_AGG, ROW_AGG, COL_AGG };
	enum class AggOp : int { NONE, SUM, SUM_SQ, MIN, MAX };
	enum class RowType : int { NONE, FULL_AGG = 4 };
	
	jitify::Program program;
	OpType op_type;
	AggType agg_type;
	AggOp agg_op;
	RowType row_type;
    const std::string name;
	int32_t const_dim2;
	uint32_t num_temp_vectors;
	bool TB1 = false;
	bool sparse_safe = true;
};

class SpoofCUDAContext {
	jitify::JitCache kernel_cache;
	std::map<const std::string, SpoofOperator> ops;
	CUmodule reductions;
	std::map<const std::string, CUfunction> reduction_kernels;
	double handling_total, compile_total;
	uint32_t compile_count;
	
	const std::string resource_path;
	const std::vector<std::string> include_paths;
	
public:
	// ToDo: make launch config more adaptive
	// num threads
	const int NT = 256;

  	// values / thread
	const int VT = 4;

	explicit SpoofCUDAContext(const char* resource_path_, std::vector<std::string>  include_paths_) : reductions(nullptr),
			resource_path(resource_path_), include_paths(std::move(include_paths_)), handling_total(0.0), compile_total(0.0),
			compile_count(0) {}

	static size_t initialize_cuda(uint32_t device_id, const char* resource_path_);

	static void destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id);

	int compile(const std::string &src, const std::string &name, SpoofOperator::OpType op_type,
			SpoofOperator::AggType agg_type = SpoofOperator::AggType::NONE,
			SpoofOperator::AggOp agg_op = SpoofOperator::AggOp::NONE,
			SpoofOperator::RowType row_type = SpoofOperator::RowType::NONE, bool sparse_safe = true,
			int32_t const_dim2 = -1, uint32_t num_vectors = 0, bool TB1 = false);

	
	template <typename T>
	T execute_kernel(const std::string &name, std::vector<Matrix<T>>& input,  std::vector<Matrix<T>>& sides, Matrix<T>* output,
					 T *scalars_ptr, uint32_t num_scalars, uint32_t grix) {
		
		T result = 0.0;
		size_t dev_buf_size;
		Matrix<T>* d_in = nullptr;
		Matrix<T>* d_out = nullptr;
		Matrix<T>* d_sides = nullptr;
		T* b1_transposed = nullptr;
		T *d_scalars = nullptr;

		auto o = ops.find(name);
		if (o != ops.end()) {
			SpoofOperator *op = &(o->second);

			// ToDo: multiple inputs for SpoofOuterProduct template
			CHECK_CUDART(cudaMalloc((void **)&d_in, sizeof(Matrix<T>)));
			CHECK_CUDART(cudaMemcpy(d_in, reinterpret_cast<void*>(&input[0]), sizeof(Matrix<T>), cudaMemcpyHostToDevice));

			if(output != nullptr) {
				if (op->sparse_safe && input.front().row_ptr != nullptr) {
#ifdef _DEBUG
					std::cout << "copying sparse safe row ptrs" << std::endl;
#endif
					CHECK_CUDART(cudaMemcpy(output->row_ptr, input.front().row_ptr, (input.front().rows+1)*sizeof(uint32_t), cudaMemcpyDeviceToDevice));
				}

				CHECK_CUDART(cudaMalloc((void **) &d_out, sizeof(Matrix<T>)));
				//CHECK_CUDART(cudaMemset(out->data, 0, out->rows*out->cols*sizeof(T)));
				CHECK_CUDART(cudaMemcpy(d_out, reinterpret_cast<void *>(output), sizeof(Matrix<T>),
						cudaMemcpyHostToDevice));

			}
			else {
				uint32_t num_blocks = 1;
				if (op->op_type == SpoofOperator::OpType::CW)
					num_blocks = std::ceil(((input.front().rows * input.front().cols) + NT * 2 - 1) / (NT * 2));
				
				CHECK_CUDART(cudaMalloc((void **) &d_out, sizeof(Matrix<T>)));
				T* d_out_data = nullptr;
				CHECK_CUDART(cudaMalloc((void **) &d_out_data, sizeof(T) * num_blocks));
				Matrix<T> agg_out{d_out_data, 0, 0, num_blocks, 1, num_blocks};
				CHECK_CUDART(cudaMemcpy(d_out, reinterpret_cast<void *>(&agg_out), sizeof(Matrix<T>),
										cudaMemcpyHostToDevice));
			}
			
			if (!sides.empty()) {
				if(op->TB1) {
#ifdef _DEBUG
					std::cout << "transposing TB1 for " << op->name << std::endl;
#endif
					T* b1 = sides[0].data;
					uint32_t m = sides[0].rows;
					uint32_t n = sides[0].cols;
					
					cudaMalloc(reinterpret_cast<void**>(&b1_transposed), sizeof(T) * m * n);
					double alpha = 1.0;
					double beta  = 0.0;
					cublasHandle_t handle;
					
					CHECK_CUBLAS(cublasCreate(&handle));
					CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, b1, n, &beta, b1, n, b1_transposed, m));
					sides[0].data = b1_transposed;
					sides[0].rows = n;
					sides[0].cols = m;
					CHECK_CUBLAS(cublasDestroy(handle));
				}
				dev_buf_size = sizeof(Matrix<T>) * sides.size();
				CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&d_sides), dev_buf_size));
				CHECK_CUDART(cudaMemcpy(d_sides, &sides[0], dev_buf_size, cudaMemcpyHostToDevice));
			}

			if (num_scalars > 0) {
				dev_buf_size = sizeof(T) * num_scalars;
				CHECK_CUDART(cudaMalloc((void **)&d_scalars, dev_buf_size));
				CHECK_CUDART(cudaMemcpy(d_scalars, scalars_ptr, dev_buf_size, cudaMemcpyHostToDevice));
			}
			
		    switch(op->op_type) {
		    case SpoofOperator::OpType::CW:
				launch_cw_kernel(op, d_in, d_out, d_sides, sides.size(), d_scalars, input.front().rows,
					 input.front().cols, grix, input[0]);
				break;
		    case SpoofOperator::OpType::RA:
				launch_ra_kernel(op, d_in, d_out, d_sides, sides.size(), d_scalars, input.front().rows,
								 input.front().cols, grix, input[0].row_ptr!=nullptr);
		        break;
		    default:
				throw std::runtime_error("error: unknown spoof operator");
		    }
			
			if (num_scalars > 0)
				CHECK_CUDART(cudaFree(d_scalars));
			
			if (sides.size() > 0)
				CHECK_CUDART(cudaFree(d_sides));

			if (op->TB1)
				CHECK_CUDART(cudaFree(b1_transposed));
			
			if (op->agg_type == SpoofOperator::AggType::FULL_AGG || op->row_type == SpoofOperator::RowType::FULL_AGG) {
				Matrix<T> res_mat;
				CHECK_CUDART(cudaMemcpy(&res_mat, d_out, sizeof(Matrix<T>), cudaMemcpyDeviceToHost));
				CHECK_CUDART(cudaMemcpy(&result, res_mat.data, sizeof(T), cudaMemcpyDeviceToHost));

				CHECK_CUDART(cudaFree(res_mat.data));
				CHECK_CUDART(cudaFree(d_out));
			}
		} 
		else {
			throw std::runtime_error("kernel " + name + " not found.");
		}
		return result;
	}

	template<typename T>
	std::string determine_agg_kernel(SpoofOperator* op) {
	  std::string reduction_kernel_name;
	  std::string reduction_type;
	  std::string suffix = (typeid(T) == typeid(double) ? "_d" : "_f");
	  switch (op->agg_type) {
	  case SpoofOperator::AggType::FULL_AGG:
		  reduction_type = "_";
		  break;
	  case SpoofOperator::AggType::ROW_AGG:
		  reduction_type = "_row_";
		  break;
	  case SpoofOperator::AggType::COL_AGG:
		  reduction_type = "_col_";
		  break;
	  default:
		  throw std::runtime_error("unknown reduction type");
		  return "";
	  }
	
	  switch (op->agg_op) {
	  case SpoofOperator::AggOp::MIN:
		  reduction_kernel_name = "reduce" + reduction_type + "min" + suffix;
		  break;
	  case SpoofOperator::AggOp::MAX:
		  reduction_kernel_name = "reduce" + reduction_type + "max" + suffix;
		  break;
	  case SpoofOperator::AggOp::SUM_SQ:
		  reduction_kernel_name = "reduce" + reduction_type + "sum_sq" + suffix;
		  break;
	  case SpoofOperator::AggOp::SUM:
		  reduction_kernel_name = "reduce" + reduction_type + "sum" + suffix;
		  break;
	  default:
		  throw std::runtime_error("unknown reduction op");
	  }
	
	  return reduction_kernel_name;
	}

	template<typename T>
	void launch_cw_kernel(SpoofOperator* op, Matrix<T>* d_in, Matrix<T>* d_out, Matrix<T>* d_sides, uint32_t num_sides,
					   T* d_scalars, uint32_t in_rows, uint32_t in_cols, uint32_t grix, const Matrix<T>& h_in) {
		T value_type;
		bool sparse = h_in.row_ptr != nullptr;
		uint32_t N = in_rows * in_cols;
		std::string op_name(op->name + "_DENSE");
		if(sparse) {
			op_name = std::string(op->name + "_SPARSE");
			N = h_in.nnz;
		}
		
		switch (op->agg_type) {
		case SpoofOperator::AggType::FULL_AGG: {
			// num ctas
			uint32_t NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			uint32_t shared_mem_size = NT * sizeof(T);

#ifdef _DEBUG
				// ToDo: connect output to SystemDS logging facilities
				std::cout << "launching spoof cellwise kernel " << op_name << " with "
						  << NT * NB << " threads in " << NB << " blocks and "
						  << shared_mem_size
						  << " bytes of shared memory for full aggregation of "
						  << N << " elements"
						  << std::endl;
#endif
			
			CHECK_CUDA(op->program.kernel(op_name)
							   .instantiate(type_of(value_type), std::max(1u, num_sides))
							   .configure(grid, block, shared_mem_size)
							   .launch(d_in, d_sides, d_out, d_scalars, N, grix));
			
			if(NB > 1) {
				std::string reduction_kernel_name = determine_agg_kernel<T>(op);
				CUfunction reduce_kernel = reduction_kernels.find(reduction_kernel_name)->second;
				N = NB;
				uint32_t iter = 1;
				while (NB > 1) {
					void* args[3] = { &d_out, &d_out, &N};

					NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
#ifdef _DEBUG
					std::cout << "agg iter " << iter++ << " launching spoof cellwise kernel " << op_name << " with "
                    << NT * NB << " threads in " << NB << " blocks and "
                    << shared_mem_size
                    << " bytes of shared memory for full aggregation of "
                    << N << " elements"
                    << std::endl;
#endif
					CHECK_CUDA(cuLaunchKernel(reduce_kernel,
							NB, 1, 1,
							NT, 1, 1,
							shared_mem_size, nullptr, args, nullptr));
							N = NB;
				}
			}
			break;
		}
		case SpoofOperator::AggType::COL_AGG: {
			// num ctas
			uint32_t NB = std::ceil((N + NT - 1) / NT);
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			uint32_t shared_mem_size = 0;
#ifdef _DEBUG
			std::cout << " launching spoof cellwise kernel " << op_name << " with "
					<< NT * NB << " threads in " << NB << " blocks for column aggregation of "
					<< N << " elements" << std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op_name)
							   .instantiate(type_of(value_type), std::max(1u, num_sides))
							   .configure(grid, block, shared_mem_size)
							   .launch(d_in, d_sides, d_out, d_scalars, N, grix));

			break;
		}
		case SpoofOperator::AggType::ROW_AGG: {
			// num ctas
			uint32_t NB = in_rows;
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			uint32_t shared_mem_size = NT * sizeof(T);
#ifdef _DEBUG
			std::cout << " launching spoof cellwise kernel " << op_name << " with "
					<< NT * NB << " threads in " << NB << " blocks and "
					<< shared_mem_size << " bytes of shared memory for row aggregation of "
					<< N << " elements" << std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op_name)
							   .instantiate(type_of(value_type), std::max(1u, num_sides))
							   .configure(grid, block, shared_mem_size)
							   .launch(d_in, d_sides, d_out, d_scalars, N, grix));

			break;
		}
		case SpoofOperator::AggType::NO_AGG: 
		default: {
			// num ctas
			// ToDo: VT not a template parameter anymore
			uint32_t NB = std::ceil((N + NT * VT - 1) / (NT * VT));
			if(sparse)
				NB = in_rows;
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			uint32_t shared_mem_size = 0;

#ifdef _DEBUG
			if(sparse) {
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

			CHECK_CUDA(op->program.kernel(op_name)
							   .instantiate(type_of(value_type), std::max(1u, num_sides))
							   .configure(grid, block, shared_mem_size)
							   .launch(d_in, d_sides, d_out, d_scalars, N, grix));
		}
		}
	}

	template<typename T>
	void launch_ra_kernel(SpoofOperator* op, Matrix<T>* d_in, Matrix<T>* d_out, Matrix<T>* d_sides, uint32_t num_sides,
			T* d_scalars, uint32_t in_rows, uint32_t in_cols, uint32_t grix, bool sparse) {
		T value_type;
		dim3 grid(in_rows, 1, 1);
		dim3 block(NT, 1, 1);
		unsigned int shared_mem_size = NT * sizeof(T);

		uint32_t tmp_len = 0;
		uint32_t temp_buf_size = 0;
		T* d_temp = nullptr;
		if(op->num_temp_vectors>0) {
			tmp_len = std::max(in_cols, op->const_dim2 < 0 ? 0 : static_cast<uint32_t>(op->const_dim2));
			temp_buf_size = op->num_temp_vectors * tmp_len * in_rows * sizeof(T);
			CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_temp), temp_buf_size));
			CHECK_CUDART(cudaMemset(d_temp, 0, temp_buf_size));
		}

		std::string name(op->name + "_DENSE");
		if(sparse)
			name = std::string(op->name + "_SPARSE");

#ifdef _DEBUG
		// ToDo: connect output to SystemDS logging facilities
		std::cout << "launching spoof rowwise kernel " << name << " with " << NT * in_rows << " threads in " << in_rows
			<< " blocks and " << shared_mem_size << " bytes of shared memory for " << in_cols << " cols processed by "
			<< NT << " threads per row, adding " << temp_buf_size / 1024 << " kb of temp buffer in global memory." <<  std::endl;
#endif
		CHECK_CUDA(op->program.kernel(name)
				.instantiate(type_of(value_type), std::max(1u, num_sides), op->num_temp_vectors, tmp_len)
				.configure(grid, block, shared_mem_size)
				.launch(d_in, d_sides, d_out, d_scalars, d_temp, grix));

		if(op->num_temp_vectors>0)
			CHECK_CUDART(cudaFree(d_temp));
	}
};

#endif // SPOOFCUDACONTEXT_H
