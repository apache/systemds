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

size_t SpoofCUDAContext::initialize_cuda(uint32_t device_id, const char* resource_path) {

#ifdef __DEBUG
	std::cout << "initializing cuda device " << device_id << std::endl;
#endif

  SpoofCUDAContext *ctx = new SpoofCUDAContext(resource_path);
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
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_f"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_f", func));

  // SUM_SQ
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_sq_d"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_sq_d", func));
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_sum_sq_f"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_sum_sq_f", func));

  // MIN
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_d"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_min_d", func));
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_min_f"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_min_f", func));

  // MAX
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_d"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_max_d", func));
  CHECK_CUDA(cuModuleGetFunction(&func, ctx->reductions, "reduce_max_f"));
  ctx->reduction_kernels.insert(std::make_pair("reduce_max_f", func));

  return reinterpret_cast<size_t>(ctx);
}

void SpoofCUDAContext::destroy_cuda(SpoofCUDAContext *ctx, uint32_t device_id) {
  delete ctx;
  ctx = nullptr;
  // cuda device is handled by jCuda atm
  //cudaDeviceReset();
}

bool SpoofCUDAContext::compile_cuda(const std::string &src,
                                    const std::string &name) {
    std::string cuda_include_path("");
    char* cdp = std::getenv("CUDA_PATH");
    if(cdp != nullptr)
        cuda_include_path = std::string("-I") + std::string(cdp) + "/include";
    else {
    	std::cout << "Warning: CUDA_PATH environment variable not set. Using default include path"
    			"/usr/local/cuda/include" << std::endl;
    	cuda_include_path = std::string("-I/usr/local/cuda/include");
    }

#ifdef __DEBUG
  std::cout << "compiling cuda kernel " << name << std::endl;
  std::cout << src << std::endl;
  std::cout << "cwd: " << std::filesystem::current_path() << std::endl;
  std::cout << "cuda_path: " << cuda_include_path << std::endl;
#endif

  SpoofOperator::AggType type = SpoofOperator::AggType::NONE;
  SpoofOperator::AggOp op = SpoofOperator::AggOp::NONE;

  auto pos = 0;
  if((pos = src.find("CellType")) != std::string::npos) {
      if(src.substr(pos, pos+30).find("FULL_AGG") != std::string::npos)
          type = SpoofOperator::AggType::FULL_AGG;
      else if(src.substr(pos, pos+30).find("ROW_AGG") != std::string::npos)
          type = SpoofOperator::AggType::ROW_AGG;
      else if(src.substr(pos, pos+30).find("COL_AGG") != std::string::npos)
          type = SpoofOperator::AggType::COL_AGG;
      else if(src.substr(pos, pos+30).find("NO_AGG") != std::string::npos)
          type = SpoofOperator::AggType::NO_AGG;
      else {
          std::cerr << "error: unknown aggregation type" << std::endl;
          return false;
      }

      if(type != SpoofOperator::AggType::NO_AGG) {
          if((pos = src.find("AggOp")) != std::string::npos) {
              if(src.substr(pos, pos+30).find("AggOp.SUM") != std::string::npos)
                  op = SpoofOperator::AggOp::SUM;
              else if(src.substr(pos, pos+30).find("AggOp.SUM_SQ") != std::string::npos)
                  op = SpoofOperator::AggOp::SUM_SQ;
              else if(src.substr(pos, pos+30).find("AggOp.MIN") != std::string::npos)
                  op = SpoofOperator::AggOp::MIN;
              else if(src.substr(pos, pos+30).find("AggOp.MAX") != std::string::npos)
                  op = SpoofOperator::AggOp::MAX;
              else {
                std::cerr << "error: unknown aggregation operator" << std::endl;
                return false;
              }
          }
      }
  }

  std::stringstream s1, s2, s3;
  s1 << "-I" << resource_path << "/cuda/headers";
  s2 << "-I" << resource_path << "/cuda/spoof";

  jitify::Program program = kernel_cache.program(src, 0, {s1.str(), s2.str(), cuda_include_path});
  ops.insert(std::make_pair(name, SpoofOperator({std::move(program), type, op})));
  return true;
}
