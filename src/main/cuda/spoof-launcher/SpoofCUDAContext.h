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

#define NOMINMAX

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <algorithm>

#include "Matrix.h"

#ifdef _DEBUG
#define __DEBUG
#endif
// #ifdef __DEBUG
    // #define JITIFY_PRINT_ALL 1
// #endif

#include <jitify.hpp>
#include <cublas_v2.h>

#include "host_utils.h"

using jitify::reflection::type_of;

struct SpoofOperator {
	enum class AggType : int { NO_AGG, NO_AGG_CONST, ROW_AGG, COL_AGG, FULL_AGG, COL_AGG_T, NONE };
	enum class AggOp : int {SUM, SUM_SQ, MIN, MAX, NONE };
	enum class OpType : int { CW, RA, MA, OP, NONE };
	
	jitify::Program program;
	AggType agg_type;
	AggOp agg_op;
	OpType op_type;
    const std::string name;
	bool TB1 = false;
};

class SpoofCUDAContext {

  jitify::JitCache kernel_cache;
  std::map<const std::string, SpoofOperator> ops;
  CUmodule reductions;
  std::map<const std::string, CUfunction> reduction_kernels;

public:
  // ToDo: make launch config more adaptive
  // num threads
  const int NT = 256;

  // values / thread
  const int VT = 4;

  const std::string resource_path;

  SpoofCUDAContext(const char* resource_path_) : reductions(nullptr), resource_path(resource_path_) {}

  static size_t initialize_cuda(uint32_t device_id, const char* resource_path_);

  static void destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id);

  bool compile_cuda(const std::string &src, const std::string &name);

	template <typename T>
	T execute_kernel(const std::string &name, T **in_ptrs, int num_inputs, T **side_ptrs, int num_sides, T *out_ptr, 
			T *scalars_ptr, int num_scalars, uint32_t m, uint32_t n, int out_len, int grix, 
			std::vector<Matrix<T>>& sides, Matrix<T>* out) {
		
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

			Matrix<T> in{in_ptrs[0], nullptr, nullptr, m, n, m*n};
			CHECK_CUDART(cudaMalloc((void **)&d_in, sizeof(Matrix<T>)));
			CHECK_CUDART(cudaMemcpy(d_in, reinterpret_cast<void*>(&in), sizeof(Matrix<T>), cudaMemcpyHostToDevice));

			if(out != nullptr) {
				CHECK_CUDART(cudaMalloc((void **) &d_out, sizeof(Matrix<T>)));
				CHECK_CUDART(cudaMemcpy(d_out, reinterpret_cast<void *>(out), sizeof(Matrix<T>),
						cudaMemcpyHostToDevice));
			}
			else {
				std::cout << "fixing scalar out" << std::endl;
				CHECK_CUDART(cudaMalloc((void **) &d_out, sizeof(Matrix<T>)));
				T* d_out_data = nullptr;
				CHECK_CUDART(cudaMalloc((void **) &d_out_data, sizeof(T)));
				Matrix<T> scalar_out{d_out_data, 0, 0, 1, 1, 1};
				CHECK_CUDART(cudaMemcpy(d_out, reinterpret_cast<void *>(&scalar_out), sizeof(Matrix<T>),
										cudaMemcpyHostToDevice));
			}
			
			if (num_sides > 0) {
				if(op->TB1) {
					T* b1 = sides[0].data;
					uint32_t m = sides[0].rows;
					uint32_t n = sides[0].cols;

					std::cout << "--- b1:" << std::endl;
					std::vector<T> tmp_b1(m*n);
					CHECK_CUDART(cudaMemcpy(tmp_b1.data(), sides[0].data, sizeof(T) * tmp_b1.size(), cudaMemcpyDeviceToHost));
					for (auto i = 0; i < m; ++i) {
						for(auto j = i * n; j < (i+1) * n; ++j)
							std::cout << tmp_b1[j] << " ";
						std::cout << std::endl;
					}
					
					cudaMalloc(reinterpret_cast<void**>(&b1_transposed), sizeof(T) * m * n);
					double alpha = 1.0;
					double beta  = 0.0;
					cublasHandle_t handle;
					
					CHECK_CUBLAS(cublasCreate(&handle));
					CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, b1, n, &beta, b1, n, b1_transposed, m));					
					// CHECK_CUDART(cudaDeviceSynchronize());
					//
					// CHECK_CUDART(cudaMemcpy(b1_transposed, b1, sizeof(T) * m * n, cudaMemcpyDeviceToDevice));

					std::cout << "--- b1_transposed:" << std::endl;
					std::vector<T> tmp_b1t(m*n);
					CHECK_CUDART(cudaMemcpy(tmp_b1t.data(), b1_transposed, sizeof(T) * tmp_b1t.size(), cudaMemcpyDeviceToHost));
					for (auto i = 0; i < n; ++i) {
						for(auto j = i * m; j < (i+1) * m; ++j)
							std::cout << tmp_b1t[j] << " ";
						std::cout << std::endl;
					}

					sides[0].data = b1_transposed;
					sides[0].rows = n;
					sides[0].cols = m;
					CHECK_CUBLAS(cublasDestroy(handle));
				}
				dev_buf_size = sizeof(Matrix<T>) * num_sides;
				CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&d_sides), dev_buf_size));
				CHECK_CUDART(cudaMemcpy(d_sides, &sides[0], dev_buf_size, cudaMemcpyHostToDevice));
			}

			if (num_scalars > 0) {
				dev_buf_size = sizeof(T) * num_scalars;
				CHECK_CUDART(cudaMalloc((void **)&d_scalars, dev_buf_size));
				CHECK_CUDART(cudaMemcpy(d_scalars, scalars_ptr, dev_buf_size, cudaMemcpyHostToDevice));
			}
			
		    switch(op->op_type) {
//		    case SpoofOperator::OpType::CW:
//		        result = launch_cw_kernel(op, in_ptrs, out_ptr, d_sides, d_scalars, m, n, grix);
//		        break;
		    case SpoofOperator::OpType::RA:
				result = launch_ra_kernel(op, d_in, d_out, d_sides, num_sides, d_scalars, m, n, grix);
		        break;
		    default:
				std::cerr << "error: unknown spoof operator" << std::endl;
		        return result;
		    }
			
			if (num_scalars > 0)
				CHECK_CUDART(cudaFree(d_scalars));
			
			if (num_sides > 0)
				CHECK_CUDART(cudaFree(d_sides));

			if(op->TB1)
				cudaFree(b1_transposed);
			
			if(op->agg_type == SpoofOperator::AggType::FULL_AGG) {
				std::cout << "retrieving scalar result" << std::endl;
				
				Matrix<T> res_mat;
				CHECK_CUDART(cudaMemcpy(&res_mat, d_out, sizeof(Matrix<T>), cudaMemcpyDeviceToHost));
				CHECK_CUDART(cudaMemcpy(&result, res_mat.data, sizeof(T), cudaMemcpyDeviceToHost));

//				result = res_mat.data[0];
				CHECK_CUDART(cudaFree(res_mat.data));
				CHECK_CUDART(cudaFree(d_out));
			}
		} 
		else {
			std::cerr << "kernel " << name << " not found." << std::endl;
			return result;
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
          std::cerr << "unknown reduction type" << std::endl;
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
          std::cerr << "unknown reduction op" << std::endl;
          return "";
      }

      return reduction_kernel_name;
  }

	template<typename T>
	T launch_cw_kernel(SpoofOperator* op, T **in_ptrs, T *out_ptr, T** d_sides, T* d_scalars, int m, int n, int grix) {

		T result = 0.0;
  	    uint32_t N = m * n;
  		size_t dev_buf_size = 0;
		T *d_temp_agg_buf;

		switch (op->agg_type) {
		case SpoofOperator::AggType::FULL_AGG: {
			// num ctas
			int NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			unsigned int shared_mem_size = NT * sizeof(T);

			size_t dev_buf_size = sizeof(T) * NB;
			CHECK_CUDART(cudaMalloc((void **)&d_temp_agg_buf, dev_buf_size));
#ifdef __DEBUG
			// ToDo: connect output to SystemDS logging facilities
			std::cout << "launching spoof cellwise kernel " << op->name << " with "
					<< NT * NB << " threads in " << NB << " blocks and "
					<< shared_mem_size
					<< " bytes of shared memory for full aggregation of "
					<< N << " elements"
					<< std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op->name)
					.instantiate(type_of(result))
					.configure(grid, block, shared_mem_size)
					.launch(in_ptrs[0], d_sides, d_temp_agg_buf, d_scalars, m, n, grix));

			if(NB > 1) {
				std::string reduction_kernel_name = determine_agg_kernel<T>(op);

				CUfunction reduce_kernel = reduction_kernels.find(reduction_kernel_name)->second;
				N = NB;
				int iter = 1;
				while (NB > 1) {
					void* args[3] = { &d_temp_agg_buf, &d_temp_agg_buf, &N};
					
					NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
#ifdef __DEBUG
					std::cout << "agg iter " << iter++ << " launching spoof cellwise kernel " << op->name << " with "
                    << NT * NB << " threads in " << NB << " blocks and "
                    << shared_mem_size
                    << " bytes of shared memory for full aggregation of "
                    << N << " elements"
                    << std::endl;
#endif
					CHECK_CUDA(cuLaunchKernel(reduce_kernel, 
							NB, 1, 1, 
							NT, 1, 1,
							shared_mem_size, 0, args, 0));
							N = NB;
				}
			}
                            
			CHECK_CUDART(cudaMemcpy(&result, d_temp_agg_buf, sizeof(T), cudaMemcpyDeviceToHost));
			CHECK_CUDART(cudaFree(d_temp_agg_buf));
			break;
		}
		case SpoofOperator::AggType::COL_AGG: {
			// num ctas
			int NB = std::ceil((N + NT - 1) / NT);
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			unsigned int shared_mem_size = 0;
#ifdef __DEBUG
			std::cout << " launching spoof cellwise kernel " << op->name << " with "
					<< NT * NB << " threads in " << NB << " blocks for column aggregation of "
					<< N << " elements" << std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op->name)
					.instantiate(type_of(result))
					.configure(grid, block)
					.launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix));
			
			break;
		}
		case SpoofOperator::AggType::ROW_AGG: {
			// num ctas
			int NB = m;
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
			unsigned int shared_mem_size = NT * sizeof(T);				
#ifdef __DEBUG
			std::cout << " launching spoof cellwise kernel " << op->name << " with "
					<< NT * NB << " threads in " << NB << " blocks and "
					<< shared_mem_size << " bytes of shared memory for row aggregation of "
					<< N << " elements" << std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op->name)
					.instantiate(type_of(result))
					.configure(grid, block, shared_mem_size)
					.launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix));
			
			break;
		}
		case SpoofOperator::AggType::NO_AGG: 
		default: {
			// num ctas
			// ToDo: VT not a template parameter anymore
			int NB = std::ceil((N + NT * VT - 1) / (NT * VT));
			dim3 grid(NB, 1, 1);
			dim3 block(NT, 1, 1);
#ifdef __DEBUG
			std::cout << "launching spoof cellwise kernel " << op->name << " with " << NT * NB
					<< " threads in " << NB << " blocks without aggregation for " 
					<< N << " elements"
					<< std::endl;
#endif
			CHECK_CUDA(op->program.kernel(op->name)
					.instantiate(type_of(result))
					.configure(grid, block)
					.launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix));
		}
		}
		return result;
	}

	template<typename T>
	T launch_ra_kernel(SpoofOperator* op, Matrix<T>* d_in, Matrix<T>* d_out, Matrix<T>* d_sides, uint32_t num_sides,
			T* d_scalars, uint32_t in_rows, uint32_t in_cols, uint32_t grix) {

		T result = 0.0;
		T *d_temp_agg_buf = nullptr;
//		size_t out_buf_size = 0;
//		if(d_out == nullptr) {
//			out_buf_size = sizeof(T) * 1;
//			CHECK_CUDART(cudaMalloc((void **) &d_out, out_buf_size));
//
//			// Matrix<T> out{in_ptrs[0], nullptr, nullptr, static_cast<uint32_t>(m), n, m*n};
//			// CHECK_CUDART(cudaMalloc((void **)&d_in, sizeof(Matrix<T>)));
//			// CHECK_CUDART(cudaMemcpy(d_in, reinterpret_cast<void*>(&in), sizeof(Matrix<T>), cudaMemcpyHostToDevice));
//
//		}
		
		dim3 grid(in_rows, 1, 1);
		dim3 block(NT, 1, 1);
		unsigned int shared_mem_size = NT * sizeof(T);

//#ifdef __DEBUG
			// ToDo: connect output to SystemDS logging facilities
			std::cout << "launching spoof rowwise kernel " << op->name << " with " << NT * in_rows << " threads in " << in_rows
				<< " blocks and " << shared_mem_size << " bytes of shared memory for " << in_cols << " cols processed by " << NT << " threads per row " << std::endl;
//#endif
		
		CHECK_CUDA(op->program.kernel(op->name)
				.instantiate(type_of(result), std::max(1u, num_sides))
				.configure(grid, block, shared_mem_size)
				.launch(d_in, d_sides, d_out, d_scalars, grix));
		
		return result;
	}
};

#endif // SPOOFCUDACONTEXT_H
