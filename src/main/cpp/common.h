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
#ifndef COMMON_H
#define COMMON_H

// Since we call cblas_dgemm in openmp for loop,
// we call "extension" APIs for setting the number of threads.
#ifdef USE_INTEL_MKL
#include <mkl.h>
	#if INTEL_MKL_VERSION < 20170000
		// Will throw an error at development time in non-standard settings
		PLEASE DONOT COMPILE SHARED LIBRARIES WITH OLDER MKL VERSIONS
	#endif
#include <mkl_service.h>
extern "C" void mkl_set_num_threads(int numThreads);
#else
#include <cblas.h>
extern "C" void openblas_set_num_threads(int numThreads);
#endif

template<class FP>
size_t computeNNZ(FP* arr, int limit) {
    size_t nnz = 0;
#ifndef USE_INTEL_MKL
#pragma omp parallel for reduction(+: nnz)
#endif
    for(int i=0; i<limit; i++)
        nnz += (arr[i]!=0) ? 1 : 0;
    return nnz;
}

static int SYSDS_CURRENT_NUM_THREADS = -1;
static void setNumThreadsForBLAS(int numThreads) {
	if (SYSDS_CURRENT_NUM_THREADS != numThreads) {
#ifdef USE_OPEN_BLAS
		openblas_set_num_threads(numThreads);
#else
		mkl_set_num_threads(numThreads);
#endif
		SYSDS_CURRENT_NUM_THREADS = numThreads;
	}
}

#endif // COMMON_H
