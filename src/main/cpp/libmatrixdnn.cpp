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
#include "libmatrixdnn.h"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "omp.h"

#ifdef USE_OPEN_BLAS
#include <cblas.h>
extern void openblas_set_num_threads(int numThreads);
#elif defined USE_INTEL_MKL
#include <mkl.h>
#include <mkl_service.h>
#endif

void rotate180(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
    int PQ = P*Q;
    int KQ = K*Q;
	for (int k = 0; k < K; k++) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				outputArray[p*KQ + q*K + k] = inputArray[k*PQ + p*Q + q];
			}
		}
	}
}

void col2im(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
	for (int p = 0; p < P; p++) {
		// h = p*stride_h + r - pad_h
		//   = r + hOffset
		// Based on restrictions: h >= 0 and r >= 0 and h < H and r < R, we get
		// max(0, - hOffset) <= r < min(R, H - hOffset)
		int hOffset = p*stride_h - pad_h;
		int rStart = MAX(0, - hOffset);
		int rEnd = MIN(R, H - hOffset);
		for (int q = 0; q < Q; q++) {
			// Using the same logic as above on following:
			// w = q*stride_w + s - pad_w
			int wOffset = q*stride_w - pad_w;
			int sStart = MAX(0, - wOffset);
			int sEnd = MIN(S, W - wOffset);
			int tempOffset = (p*Q + q)*C*R*S;
			for (int c = 0; c < C; c++) {
				int outOffset = c*H*W;
				int inputOffset = tempOffset + c*R*S;
				for (int r = rStart; r < rEnd; r++) {
					for (int s = sStart; s < sEnd; s++) {
						int inputIndex = inputOffset + r*S + s;
						int outIndex = outOffset + (hOffset + r)*W + wOffset + s;
						outputArray[outIndex] += inputArray[inputIndex];
					}
				}
			}
		}
	}
}

void im2col(double* inputArray, double* outputArray, int N, int C, int H, int W,
            int K, int R, int S, int stride_h, int stride_w, int pad_h,
            int pad_w, int P, int Q) {
  int CRS = C * R * S;
  std::size_t size = Q * sizeof(double);
  if (stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
    for (int c = 0; c < CRS; ++c) {
      int wOffset = c % S;
      int hOffset = (c / S) % R;
      int cInput = c / R / S;
      for (int h = 0; h < P; ++h) {
        int hPadded = h + hOffset;
        int outOffset = (c * P + h) * Q;
        int inputOffset = (cInput * H + hPadded) * W;
        std::memcpy(outputArray + outOffset, inputArray + inputOffset + wOffset,
                    size);
        int w = Q - 1;
        int wPadded = w + wOffset;
        if (hPadded < H && wPadded < W)
          outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
        else
          outputArray[outOffset + w] = 0;
      }
    }
  } else {
    for (int c = 0; c < CRS; ++c) {
      int wOffset = c % S;
      int hOffset = (c / S) % R;
      int cInput = c / R / S;
      for (int h = 0; h < P; ++h) {
        int outOffset = (c * P + h) * Q;
        int hPadded = h * stride_h - pad_h + hOffset;
        int inputOffset = (cInput * H + hPadded) * W;
        if (hPadded < 0 || hPadded >= H) {
          std::memset(outputArray + outOffset, 0, size);
        } else {
          for (int w = 0; w < Q; ++w) {
            int wPadded = w * stride_w - pad_w + wOffset;
            if (wPadded >= 0 && wPadded < W)
              outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
            else
              outputArray[outOffset + w] = 0;
          }
        }
      }
    }
  }
} 


void conv2dBackwardFilterDense(double* inputPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
  // First step: Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CHW = C * H * W;
  int CRS = C*R*S;
  int PQ = P*Q;
  int KPQ = K*PQ;
  int numRotatedElem = KPQ;
  int numIm2ColElem = CRS * PQ;
  int numTempElem = CRS * K;
  
  int m1 = CRS;
  int n1 = K;
  int k1 = PQ;
  
  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  double* temp = new double[numTempElem*numOpenMPThreads];
  std::memset(temp, 0, numTempElem*numOpenMPThreads*sizeof(double));
  double* rotatedDoutPtrArrays = new double[numRotatedElem*numOpenMPThreads];
  double* loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];

#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
  	double* loweredMat = loweredMatArrays + numIm2ColElem*omp_get_thread_num();

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);
           
    // Step 2: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*omp_get_thread_num();
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);
    
    // Multiply to get CRS X K
    double* temp1 = temp + numTempElem*omp_get_thread_num();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1, n1, k1, 1.0, loweredMat, k1,
              rotatedDoutPtr, n1, 1.0, temp1, n1);
              
  } // end omp parallel for
  
  // Inplace transpose addition
  int numRow = CRS;
  for(int t = 0; t < numOpenMPThreads; t++) {
  	int iter = 0;
  	double* temp1 = temp + numTempElem*t;
	for(int i = 0; i < CRS; i++) {
		for(int j = 0; j < K; j++, iter++) {
			int index = j*numRow+i;
			retPtr[index] += temp1[iter];
		}
	}
  } 
  
  delete [] temp;
  delete [] loweredMatArrays;
  delete [] rotatedDoutPtrArrays;
}

void conv2dBackwardDataDense(double* filterPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
   // First step: Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CRS = C * R * S;
  int CHW = C * H * W;
  int PQ = P * Q;
  int KPQ = K * PQ;
  int numRotatedElem = PQ * K;
  int numCol2ImElem = PQ * CRS;

  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  double* rotatedDoutPtrArrays = new double[numRotatedElem*numOpenMPThreads];
  double* col2imInputArrays = new double[numCol2ImElem*numOpenMPThreads];

#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    // Step 1: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*omp_get_thread_num();
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);

    // Step 2: t(rotatedDout (PQ X K) %*% filter (K X CRS))
    double* col2imInput = col2imInputArrays + numCol2ImElem*omp_get_thread_num();
    matmult(rotatedDoutPtr, filterPtr, col2imInput,
            PQ, K, CRS, 1);

    // Step 3: Perform col2im
    col2im(col2imInput, retPtr + n * CHW, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);

  } // end omp parallel for
  
  delete [] rotatedDoutPtrArrays;
  delete [] col2imInputArrays;
    
}

void conv2dBiasAddDense(double* inputPtr, double* biasPtr, double* filterPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, bool addBias, int numThreads) {
  // First step:  Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CHW = C * H * W;
  int KPQ = K * P * Q;
  int PQ = P * Q;
  int numIm2ColElem = C * R * S * P * Q;
  
  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  double* loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];
  
#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    double* loweredMat = loweredMatArrays + numIm2ColElem*omp_get_thread_num();

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, K,
            C * R * S, P * Q, 1);
    
    // Step 3: Add bias
    if(addBias) {
	    double* outputArr = retPtr + n*KPQ;
	    int index = 0;
		for(int k = 0; k < K; k++) {
			for(int pq = 0; pq < PQ; pq++, index++) {
				outputArr[index] += biasPtr[k];
			}
		}
    }
  } // end omp parallel for
  
  delete [] loweredMatArrays;
}
