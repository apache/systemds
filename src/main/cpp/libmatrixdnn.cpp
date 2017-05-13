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
#ifdef USE_INTEL_MKL
  #include "mkl_dnn.h"
#else
  #include "omp.h"
#endif

int computeNNZ(double* arr, int limit) {
  int nnz = 0;
#ifndef USE_INTEL_MKL
  #pragma omp parallel for reduction(+: nnz)
#endif
  for(int i=0; i<limit; i++)
    nnz += (arr[i]!=0) ? 1 : 0;
  return nnz;
}

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

#ifdef USE_INTEL_MKL
// Returns true if error
bool MKL_DNN_ERROR(dnnError_t code) {
  if(code == E_SUCCESS) return false;
  else if(code == E_INCORRECT_INPUT_PARAMETER) std::cerr << "ERROR: Incorrect input parameter\n";
  else if(code == E_MEMORY_ERROR) std::cerr << "ERROR: Memory error\n";
  else if(code == E_UNSUPPORTED_DIMENSION) std::cerr << "ERROR: Unsupported dimensions\n";
  else if(code == E_UNIMPLEMENTED) std::cerr << "ERROR: Unimplemented operation\n";
  return true;
} 
#endif

int conv2dBackwardFilterDense(double* inputPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
  int CRS = C*R*S;
#ifdef USE_INTEL_MKL
  setNumThreadsForBLAS(numThreads);
  // Step 1: Create a description of a DNN operation
  dnnPrimitive_t pConvolution;
  size_t dimension = 4;
  size_t srcSize[4] = {W, H, C, N};
  size_t dstSize[4] = {Q, P, K, N};
  size_t filterSize[4] = {S, R, C, K};
  size_t convolutionStrides[2] = {stride_w, stride_h};
  int pads[2] = {-pad_w, -pad_h};
  void* resources[dnnResourceNumber] = {0};
  resources[dnnResourceDiffDst] = doutPtr;
  resources[dnnResourceSrc] = inputPtr;
  resources[dnnResourceDiffFilter] = retPtr;
  dnnConvolutionCreateBackwardFilter_F64(&pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, 
      srcSize, dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros);
  
  // Step 2: Perform the DNN operation
  if(MKL_DNN_ERROR(dnnExecute_F64(pConvolution, resources))) {
    return -1; // nnz == -1 indicates error.
  }
  
  // Step 3: Destroy the description of the operation
  dnnDelete_F64(pConvolution);
#else
  // First step: Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CHW = C * H * W;
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
    int threadID = omp_get_thread_num();
  	double* loweredMat = loweredMatArrays + numIm2ColElem*threadID;

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);
           
    // Step 2: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*threadID;
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);
    
    // Multiply to get tmp1 = CRS X K
    double* temp1 = temp + numTempElem*threadID;
    // Step 3: temp1 = alpha * (loweredMat (CRS X PQ) %*% rotated_dout (PQ X K)) + beta*temp1
    int m1rlen = C * R * S; int m1clen = P * Q; int m2clen = K;
    double* m1Ptr = loweredMat; double* m2Ptr = rotatedDoutPtr; double alpha = 1; double beta = 1;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1rlen, m2clen, m1clen, alpha, m1Ptr, m1clen, m2Ptr, m2clen, beta, temp1, m2clen);
  } // end omp parallel for
  
  delete [] loweredMatArrays;
  delete [] rotatedDoutPtrArrays;
  
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
#endif
  return computeNNZ(retPtr, K*CRS);
}

int conv2dBackwardDataDense(double* filterPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
  int CHW = C * H * W;
#ifdef USE_INTEL_MKL
  setNumThreadsForBLAS(numThreads);
  // Step 1: Create a description of a DNN operation
  dnnPrimitive_t pConvolution;
  size_t dimension = 4;
  size_t srcSize[4] = {W, H, C, N};
  size_t dstSize[4] = {Q, P, K, N};
  size_t filterSize[4] = {S, R, C, K};
  size_t convolutionStrides[2] = {stride_w, stride_h};
  int pads[2] = {-pad_w, -pad_h};
  void* resources[dnnResourceNumber] = {0};
  resources[dnnResourceDiffDst] = doutPtr;
  resources[dnnResourceFilter] = filterPtr;
  resources[dnnResourceDiffSrc] = retPtr;
  dnnConvolutionCreateBackwardData_F64(&pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, 
      srcSize, dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros);
  
  // Step 2: Perform the DNN operation
  if(MKL_DNN_ERROR(dnnExecute_F64(pConvolution, resources))) {
    return -1; // nnz == -1 indicates error.
  }
  
  // Step 3: Destroy the description of the operation
  dnnDelete_F64(pConvolution);
#else 
   // First step: Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CRS = C * R * S;
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
    int threadID = omp_get_thread_num();
    // Step 1: Rotate dout
    double* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem*threadID;
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);

    // Step 2: t(rotatedDout (PQ X K) %*% filter (K X CRS))
    double* col2imInput = col2imInputArrays + numCol2ImElem*threadID;
    matmult(rotatedDoutPtr, filterPtr, col2imInput,
            PQ, K, CRS, 1);

    // Step 3: Perform col2im
    double* outputArr = retPtr + n * CHW;
    col2im(col2imInput, outputArr, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);
  } // end omp parallel for
  
  delete [] rotatedDoutPtrArrays;
  delete [] col2imInputArrays;
#endif
  return computeNNZ(retPtr, N*CHW);
}

void conv2dSparse(int apos, int alen, int* aix, double* avals, double* filterPtr, double* retPtr, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
	setNumThreadsForBLAS(1);
	double* loweredMat = new double[C * R * S * P * Q];
	
	// Step 1: Perform im2col
	double* temp = new double[C * H * W];
	std::size_t size = C * H * W * sizeof(double);
	std::memset(temp, 0, size);
	for(int j=apos; j<apos+alen; j++)
		temp[ aix[j] ] = avals[j];
	im2col(temp, loweredMat, 1, C, H, W, K,
       R, S, stride_h, stride_w, pad_h, pad_w,
       P, Q);	
	delete [] temp;
	
	// Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr, K, C * R * S, P * Q, 1);
    
	delete [] loweredMat;
}

void conv2dBackwardFilterSparseDense(int apos, int alen, int* aix, double* avals, double* rotatedDoutPtr, double* retPtr, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads) {
	setNumThreadsForBLAS(1);
	int CHW = C * H * W;
	int CRS = C*R*S;
	int PQ = P*Q;
	int KPQ = K*PQ;
	int m1 = CRS;
	int n1 = K;
	int k1 = PQ;
	
	double* loweredMat = new double[CRS * PQ];
	
	// Step 1: Perform im2col
	double* temp = new double[C * H * W];
	std::size_t size = C * H * W * sizeof(double);
	std::memset(temp, 0, size);
	for(int j=apos; j<apos+alen; j++)
		temp[ aix[j] ] = avals[j];
	im2col(temp, loweredMat, 1, C, H, W, K,
       R, S, stride_h, stride_w, pad_h, pad_w,
       P, Q);
    delete [] temp;
	
	// Multiply to get CRS X K
	double* temp1 = new double[CRS * K];
	// Step 3: loweredMat (CRS X PQ) %*% rotatedDoutPtr (PQ X K) 
    matmult(loweredMat, rotatedDoutPtr, temp1, C * R * S, P * Q, K, 1);
    delete [] loweredMat;
     
    // Inplace addition
    for(int iter = 0; iter<K*CRS; iter++) {
    	retPtr[iter] += temp1[iter];
    }
    
	delete [] temp1;
}


int conv2dBiasAddDense(double* inputPtr, double* biasPtr, double* filterPtr, double* retPtr, int N, int C, int H, int W, int K, int R, int S,
    int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, bool addBias, int numThreads) {
  int KPQ = K * P * Q;
  
#ifdef USE_INTEL_MKL
  setNumThreadsForBLAS(numThreads);
  // Step 1: Create a description of a DNN operation
  dnnPrimitive_t pConvolution;
  size_t dimension = 4;
  size_t srcSize[4] = {W, H, C, N};
  size_t dstSize[4] = {Q, P, K, N};
  size_t filterSize[4] = {S, R, C, K};
  size_t convolutionStrides[2] = {stride_w, stride_h};
  int pads[2] = {-pad_w, -pad_h};
  void* resources[dnnResourceNumber] = {0};
  resources[dnnResourceSrc] = inputPtr;
  resources[dnnResourceFilter] = filterPtr;
  resources[dnnResourceDst] = retPtr;
  if(addBias) {
    dnnConvolutionCreateForwardBias_F64(&pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, 
      srcSize, dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros);
    resources[dnnResourceBias] = biasPtr;
  }
  else { 
    dnnConvolutionCreateForward_F64(&pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, 
      srcSize, dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros);
  }
  
  // Step 2: Perform the DNN operation
  if(MKL_DNN_ERROR(dnnExecute_F64(pConvolution, resources))) {
    return -1; // nnz == -1 indicates error.
  }
  
  // Step 3: Destroy the description of the operation
  dnnDelete_F64(pConvolution);
#else 
  // ------------------------------------------------------------------------------------
  // First step:  Avoids oversubscription and other openmp/internal blas threading issues
  setNumThreadsForBLAS(1);
  
  int CHW = C * H * W;
  int PQ = P * Q;
  int numIm2ColElem = C * R * S * P * Q;
  
  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  double* loweredMatArrays = new double[numIm2ColElem*numOpenMPThreads];
  
#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    int threadID = omp_get_thread_num();
    double* loweredMat = loweredMatArrays + numIm2ColElem*threadID;

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K,
           R, S, stride_h, stride_w, pad_h, pad_w,
           P, Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, K,
            C * R * S, P * Q, 1);
    
    // Step 3: Add bias
    double* outputArr = retPtr + n*KPQ;
    if(addBias) {
	    int index = 0;
		for(int k = 0; k < K; k++) {
			for(int pq = 0; pq < PQ; pq++, index++) {
				outputArr[index] += biasPtr[k];
			}
		}
    }
  } // end omp parallel for
  delete [] loweredMatArrays;
  // ------------------------------------------------------------------------------------
#endif
  
  return computeNNZ(retPtr, N*KPQ);
}
