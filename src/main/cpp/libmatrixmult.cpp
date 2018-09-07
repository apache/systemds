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

#ifdef USE_OPEN_BLAS
	#include <cblas.h>
#else
  #include <mkl_service.h>  
#endif

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
 
void dmatmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m, int k, int n, int numThreads) {
  //BLAS routine dispatch according to input dimension sizes (we don't use cblas_dgemv 
  //with CblasColMajor for matrix-vector because it was generally slower than dgemm)
  setNumThreadsForBLAS(numThreads);
  if( m == 1 && n == 1 ) //VV
    retPtr[0] = cblas_ddot(k, m1Ptr, 1, m2Ptr, 1);
  else if( n == 1 ) //MV
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, 1, m1Ptr, k, m2Ptr, 1, 0, retPtr, 1);  
  else //MM
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, m1Ptr, k, m2Ptr, n, 0, retPtr, n);
}

void smatmult(float* m1Ptr, float* m2Ptr, float* retPtr, int m, int k, int n, int numThreads) {  
  //BLAS routine dispatch according to input dimension sizes (we don't use cblas_sgemv 
  //with CblasColMajor for matrix-vector because it was generally slower than sgemm)
  setNumThreadsForBLAS(numThreads);
  if( m == 1 && n == 1 ) //VV
    retPtr[0] = cblas_sdot(k, m1Ptr, 1, m2Ptr, 1);
  else if( n == 1 ) //MV
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, 1, m1Ptr, k, m2Ptr, 1, 0, retPtr, 1);  
  else //MM
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, m1Ptr, k, m2Ptr, n, 0, retPtr, n);
}

void tsmm(double* m1Ptr, double* retPtr, int m1rlen, int m1clen, bool leftTrans, int numThreads) {
  setNumThreadsForBLAS(numThreads);
  if( (leftTrans && m1clen == 1) || (!leftTrans && m1rlen == 1) ) {
    retPtr[0] = cblas_ddot(leftTrans ? m1rlen : m1clen, m1Ptr, 1, m1Ptr, 1);
  }
  else { //general case
    int n = leftTrans ? m1clen : m1rlen;
    int k = leftTrans ? m1rlen : m1clen;
    cblas_dsyrk(CblasRowMajor, CblasUpper, leftTrans ? CblasTrans : CblasNoTrans, n, k, 1, m1Ptr, n, 0, retPtr, n);
  }
}
