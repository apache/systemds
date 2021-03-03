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
#ifndef SYSTEMDS_SPOOFOPERATOR_H
#define SYSTEMDS_SPOOFOPERATOR_H

#include <cmath>
#include <cstdint>
#include <string>
#include <jitify.hpp>
#include "host_utils.h"
#include "Matrix.h"

struct SpoofOperator {
//	enum class OpType : int { CW, RA, MA, OP, NONE };
	enum class AggType : int { NO_AGG, FULL_AGG, ROW_AGG, COL_AGG };
	enum class AggOp : int { SUM, SUM_SQ, MIN, MAX };
	enum class RowType : int { FULL_AGG = 4 };
	
//	OpType op_type;
	std::string name;
//	jitify::Program program;
	std::unique_ptr<jitify::Program> program;
	
	[[nodiscard]] virtual bool isSparseSafe() const = 0;
};

struct SpoofCellwiseOp : public SpoofOperator {
	bool sparse_safe;
	AggType agg_type;
	AggOp agg_op;
	CUfunction agg_kernel{};
	SpoofCellwiseOp(AggType at, AggOp ao, bool ss) : agg_type(at), agg_op(ao), sparse_safe(ss) {}
	
	[[nodiscard]] bool isSparseSafe() const override { return sparse_safe; }
};

struct SpoofRowwiseOp : public SpoofOperator {
	bool TB1 = false;
	uint32_t num_temp_vectors;
	int32_t const_dim2;
	RowType row_type;
	
	SpoofRowwiseOp(RowType rt, bool tb1, uint32_t ntv, int32_t cd2)  : row_type(rt), TB1(tb1), num_temp_vectors(ntv),
			const_dim2(cd2) {}
			
	[[nodiscard]] bool isSparseSafe() const override { return false; }
};

template<typename T>
struct DevMatPtrs {
	Matrix<T>* ptrs[3] = {0,0,0};
	
	Matrix<T>*& in = ptrs[0];
	Matrix<T>*& sides = ptrs[1];
	Matrix<T>*& out = ptrs[2];
	T* scalars{};

	~DevMatPtrs() {
#ifndef NDEBUG
		std::cout << "~DevMatPtrs() before cudaFree:\n";
		int i = 0;
		for (auto& p : ptrs) {
			std::cout << " p[" << i << "]=" << p;
			i++;
		}
		std::cout << std::endl;
#endif
		for (auto& p : ptrs) {
			if (p) {
				CHECK_CUDART(cudaFree(p));
				p = nullptr;
			}
		}
#ifndef NDEBUG
		std::cout << "~DevMatPtrs() after cudaFree:\n";
		i = 0;
		for (auto& p : ptrs) {
			std::cout << " p[" << i << "]=" << p;
			i++;
		}
		std::cout << std::endl;
#endif
	}
};

#endif //SYSTEMDS_SPOOFOPERATOR_H
