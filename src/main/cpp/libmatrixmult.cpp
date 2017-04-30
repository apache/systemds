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

#include "config.h"
#include "libmatrixmult.h"
#include <cstdlib>
#include "omp.h"
#include <cmath>

int SYSML_CURRENT_NUM_THREADS = -1;
void setNumThreadsForBLAS(int numThreads) {
	if(SYSML_CURRENT_NUM_THREADS != numThreads) {
#ifdef USE_OPEN_BLAS
		openblas_set_num_threads(numThreads);
#else
		mkl_set_num_threads(numThreads);
#endif
	    SYSML_CURRENT_NUM_THREADS = numThreads;
	}
}
 
// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads) {
  int m = m1rlen;
  int n = m2clen;
  int k = m1clen;
  
  setNumThreadsForBLAS(numThreads);
  
  // if(m2clen == 1)
  // 	cblas_dgemv(CblasRowMajor, CblasNoTrans, m1rlen, m1clen, 1, m1Ptr, m1clen, m2Ptr, 1, 0, retPtr, 1);
  // else 
  	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, m1Ptr, k, m2Ptr, n, 0, retPtr, n);
}

void tsmm(double* m1Ptr, double* retPtr, int m1rlen, int m1clen, bool isLeftTranspose, int numThreads) {
  int m = isLeftTranspose ? m1clen : m1rlen;
  int n = isLeftTranspose ? m1clen : m1rlen;
  int k = isLeftTranspose ? m1rlen : m1clen;
  
  setNumThreadsForBLAS(numThreads);
  
  cblas_dgemm(CblasRowMajor, isLeftTranspose ? CblasTrans : CblasNoTrans, isLeftTranspose ? CblasNoTrans : CblasTrans, m, n, k, 1, m1Ptr, k, m1Ptr, n, 0, retPtr, n);
}
 
