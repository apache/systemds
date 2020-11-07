%TMP%

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

// CellType: %TYPE%
// AggOp: %AGG_OP_NAME%
// SparseSafe: %SPARSE_SAFE%
// SEQ: %SEQ%

#include "agg_ops.cuh"
#include "reduction.cuh"
#include "spoof_utils.cuh"
#include "utils.cuh"

template<typename T>
struct SpoofCellwiseOp {
   T**b; T* scalars; 
   int m, n, grix_;

   SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix) : 
       b(b), scalars(scalars), m(m), n(n), grix_(grix) {}

   __device__  __forceinline__ T operator()(T a, int idx) const {
        int rix = idx / n;
        int cix = idx % n;
        int grix = grix_ + rix;
%BODY_dense%
        return %OUT%;
   }
};

template<typename T>
__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {
   %AGG_OP%<T> agg_op;
   SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix);
   %TYPE%<T, %AGG_OP%<T>, SpoofCellwiseOp<T>>(a, c, m, n, %INITIAL_VALUE%, agg_op, spoof_op);
};
