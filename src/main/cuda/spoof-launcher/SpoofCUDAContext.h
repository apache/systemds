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

#include <cmath>
#include <cstdint>
#include <map>
#include <string>

#ifdef __DEBUG
    #define JITIFY_PRINT_ALL 1
#endif

#include <jitify.hpp>

#include "host_utils.h"

using jitify::reflection::type_of;

struct SpoofOperator {
  enum class AggType : int { NO_AGG, ROW_AGG, COL_AGG, FULL_AGG, NONE };
  enum class AggOp : int {SUM, SUM_SQ, MIN, MAX, NONE };

  jitify::Program program;
  AggType agg_type;
  AggOp agg_op;

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
  T execute_kernel(const std::string &name, T **in_ptrs, int num_inputs,
                   T **side_ptrs, int num_sides, T *out_ptr, T *scalars_ptr,
                   int num_scalars, int m, int n, int grix) {

    T result = 0.0;
    size_t dev_buf_size;
    T **d_sides = nullptr;
    T *d_scalars = nullptr;
    T *d_temp_agg_buf;
    uint32_t N = m * n;

    auto o = ops.find(name);
    if (o != ops.end()) {
      SpoofOperator *op = &(o->second);

      if (num_sides > 0) {
        dev_buf_size = sizeof(T *) * num_sides;
        CHECK_CUDART(cudaMalloc((void **)&d_sides, dev_buf_size));
        CHECK_CUDART(cudaMemcpy(d_sides, side_ptrs, dev_buf_size, cudaMemcpyHostToDevice));
      }

      if (num_scalars > 0) {
        dev_buf_size = sizeof(T) * num_scalars;
        CHECK_CUDART(cudaMalloc((void **)&d_scalars, dev_buf_size));
        CHECK_CUDART(cudaMemcpy(d_scalars, scalars_ptr, dev_buf_size, cudaMemcpyHostToDevice));
      }

      switch (op->agg_type) {
          case SpoofOperator::AggType::FULL_AGG: {
            // num ctas
            int NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
            dim3 grid(NB, 1, 1);
            dim3 block(NT, 1, 1);
            unsigned int shared_mem_size = NT * sizeof(T);

            dev_buf_size = sizeof(T) * NB;
            CHECK_CUDART(cudaMalloc((void **)&d_temp_agg_buf, dev_buf_size));
#ifdef __DEBUG
            // ToDo: connect output to SystemDS logging facilities
            std::cout << "launching spoof cellwise kernel " << name << " with "
                      << NT * NB << " threads in " << NB << " blocks and "
                      << shared_mem_size
                      << " bytes of shared memory for full aggregation of "
                      << N << " elements"
                      << std::endl;
#endif
            CHECK_CUDA(op->program.kernel(name)
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
                    std::cout << "agg iter " << iter++ << " launching spoof cellwise kernel " << name << " with "
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
              std::cout << " launching spoof cellwise kernel " << name << " with "
                  << NT * NB << " threads in " << NB << " blocks for column aggregation of "
                  << N << " elements" << std::endl;
#endif
              CHECK_CUDA(op->program.kernel(name)
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
              std::cout << " launching spoof cellwise kernel " << name << " with "
                  << NT * NB << " threads in " << NB << " blocks and "
                  << shared_mem_size << " bytes of shared memory for row aggregation of "
                  << N << " elements" << std::endl;
#endif
              CHECK_CUDA(op->program.kernel(name)
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
            std::cout << "launching spoof cellwise kernel " << name << " with " << NT * NB
                      << " threads in " << NB << " blocks without aggregation for " 
                      << N << " elements"
                      << std::endl;
#endif
            CHECK_CUDA(op->program.kernel(name)
                .instantiate(type_of(result))
                .configure(grid, block)
                .launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix));
          }
      }
      
      if (num_scalars > 0)
        CHECK_CUDART(cudaFree(d_scalars));

      if (num_sides > 0)
        CHECK_CUDART(cudaFree(d_sides));
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
};

#endif // SPOOFCUDACONTEXT_H
