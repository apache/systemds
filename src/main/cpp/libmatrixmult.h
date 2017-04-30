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
 
#ifndef _libmatrixmult_h
#define _libmatrixmult_h

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// *****************************************************************
// We support Intel MKL (recommended) or OpenBLAS.
// These flags are used for conditional compilation with mkl and openblas
// #define USE_INTEL_MKL
// #define USE_GNU_THREADING
// #define USE_OPEN_BLAS
// *****************************************************************

//#ifdef __cplusplus
//extern "C" {
//#endif
//#ifdef __cplusplus
//}
//#endif

// Since we call cblas_dgemm in openmp for loop,
// we call "extension" APIs for setting number of threads of the given API.
// For example: for OpenBLAS we use openblas_set_num_threads and  
// for MKL we use mkl_set_num_threads. This avoids performance degradation due to overprovisioning.
#ifdef USE_OPEN_BLAS
#include <cblas.h>
// extern "C" void openblas_set_num_threads(int numThreads);
#elif defined USE_INTEL_MKL
#include <mkl.h>
#include <mkl_service.h>
#endif

void setNumThreadsForBLAS(int numThreads);

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads);
             
void tsmm(double* m1Ptr, double* retPtr, int m1rlen, int m1clen, bool isLeftTranspose,  int numThreads);
             
#endif
