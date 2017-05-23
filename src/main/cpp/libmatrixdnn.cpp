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

#define STR1(x) #x
#define STR(x) STR1(x)

template <typename T>
int computeNNZ(T* arr, int limit) {
  int nnz = 0;
#ifndef USE_INTEL_MKL
#pragma omp parallel for reduction(+ : nnz)
#endif
  for (int i = 0; i < limit; i++) nnz += (arr[i] != 0) ? 1 : 0;
  return nnz;
}

template <typename T>
void rotate180(T* inputArray, T* outputArray, int N, int C, int H, int W, int K,
               int R, int S, int stride_h, int stride_w, int pad_h, int pad_w,
               int P, int Q) {
  int PQ = P * Q;
  int KQ = K * Q;
  for (int k = 0; k < K; k++) {
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < Q; q++) {
        outputArray[p * KQ + q * K + k] = inputArray[k * PQ + p * Q + q];
      }
    }
  }
}

template <typename T>
void col2im(T* inputArray, T* outputArray, int N, int C, int H, int W, int K,
            int R, int S, int stride_h, int stride_w, int pad_h, int pad_w,
            int P, int Q) {
  for (int p = 0; p < P; p++) {
    // h = p*stride_h + r - pad_h
    //   = r + hOffset
    // Based on restrictions: h >= 0 and r >= 0 and h < H and r < R, we get
    // max(0, - hOffset) <= r < min(R, H - hOffset)
    int hOffset = p * stride_h - pad_h;
    int rStart = MAX(0, -hOffset);
    int rEnd = MIN(R, H - hOffset);
    for (int q = 0; q < Q; q++) {
      // Using the same logic as above on following:
      // w = q*stride_w + s - pad_w
      int wOffset = q * stride_w - pad_w;
      int sStart = MAX(0, -wOffset);
      int sEnd = MIN(S, W - wOffset);
      int tempOffset = (p * Q + q) * C * R * S;
      for (int c = 0; c < C; c++) {
        int outOffset = c * H * W;
        int inputOffset = tempOffset + c * R * S;
        for (int r = rStart; r < rEnd; r++) {
          for (int s = sStart; s < sEnd; s++) {
            int inputIndex = inputOffset + r * S + s;
            int outIndex = outOffset + (hOffset + r) * W + wOffset + s;
            outputArray[outIndex] += inputArray[inputIndex];
          }
        }
      }
    }
  }
}

template <typename T>
void im2col(T* inputArray, T* outputArray, int N, int C, int H, int W, int K,
            int R, int S, int stride_h, int stride_w, int pad_h, int pad_w,
            int P, int Q) {
  int CRS = C * R * S;
  std::size_t size = Q * sizeof(T);
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
#define CHECK_MKL_ERROR(dnnCall) { \
  dnnError_t code = dnnCall; \
  if (code != E_SUCCESS) { \
  	if (code == E_INCORRECT_INPUT_PARAMETER) \
    	std::cerr << "ERROR: Incorrect input parameter:" << STR(dnnCall) << "\n"; \
  	else if (code == E_MEMORY_ERROR) \
    	std::cerr << "ERROR: Memory error:" << STR(dnnCall) << "\n"; \
  	else if (code == E_UNSUPPORTED_DIMENSION) \
    	std::cerr << "ERROR: Unsupported dimensions:" << STR(dnnCall) << "\n"; \
  	else if (code == E_UNIMPLEMENTED) \
    	std::cerr << "ERROR: Unimplemented operation:" << STR(dnnCall) << "\n"; \
    return -1; \
  } \
}

#define MKL_DNN_EXECUTE() \
  if (isSinglePrecision()) { \
    CHECK_MKL_ERROR(dnnExecute_F32(pConvolution, resources)) \
    dnnDelete_F32(pConvolution); \
  } else { \
    CHECK_MKL_ERROR(dnnExecute_F64(pConvolution, resources)) \
    dnnDelete_F64(pConvolution); \
  } \

#endif

void matMultAndAdd(double* m1Ptr, double* m2Ptr, double* ret, int m1rlen,
                   int m1clen, int m2clen) {
  double alpha = 1;
  double beta = 1;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1rlen, m2clen, m1clen,
              alpha, m1Ptr, m1clen, m2Ptr, m2clen, beta, ret, m2clen);
}

void matMultAndAdd(float* m1Ptr, float* m2Ptr, float* ret, int m1rlen,
                   int m1clen, int m2clen) {
  double alpha = 1;
  double beta = 1;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1rlen, m2clen, m1clen,
              alpha, m1Ptr, m1clen, m2Ptr, m2clen, beta, ret, m2clen);
}

template <typename T>
int conv2dBackwardFilterDenseHelper(T* inputPtr, T* doutPtr, T* retPtr, int N,
                                    int C, int H, int W, int K, int R, int S,
                                    int stride_h, int stride_w, int pad_h,
                                    int pad_w, int P, int Q, int numThreads) {
  int CRS = C * R * S;
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
  if (isSinglePrecision())
    CHECK_MKL_ERROR(dnnConvolutionCreateBackwardFilter_F32(
        &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, srcSize,
        dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros))
  else
    CHECK_MKL_ERROR(dnnConvolutionCreateBackwardFilter_F64(
        &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, srcSize,
        dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros))

  MKL_DNN_EXECUTE()
    
#else
  // First step: Avoids oversubscription and other openmp/internal blas
  // threading issues
  setNumThreadsForBLAS(1);

  int CHW = C * H * W;
  int PQ = P * Q;
  int KPQ = K * PQ;
  int numRotatedElem = KPQ;
  int numIm2ColElem = CRS * PQ;
  int numTempElem = CRS * K;

  int m1 = CRS;
  int n1 = K;
  int k1 = PQ;

  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  T* temp = new T[numTempElem * numOpenMPThreads];
  std::memset(temp, 0, numTempElem * numOpenMPThreads * sizeof(T));
  T* rotatedDoutPtrArrays = new T[numRotatedElem * numOpenMPThreads];
  T* loweredMatArrays = new T[numIm2ColElem * numOpenMPThreads];

#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    int threadID = omp_get_thread_num();
    T* loweredMat = loweredMatArrays + numIm2ColElem * threadID;

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K, R, S, stride_h,
           stride_w, pad_h, pad_w, P, Q);

    // Step 2: Rotate dout
    T* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem * threadID;
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K, R, S, stride_h,
              stride_w, pad_h, pad_w, P, Q);

    // Multiply to get tmp1 = CRS X K
    T* temp1 = temp + numTempElem * threadID;
    // Step 3: temp1 = alpha * (loweredMat (CRS X PQ) %*% rotated_dout (PQ X K))
    // + beta*temp1
    int m1rlen = C * R * S;
    int m1clen = P * Q;
    int m2clen = K;
    matMultAndAdd(loweredMat, rotatedDoutPtr, temp1, m1rlen, m1clen, m2clen);
  }  // end omp parallel for

  delete[] loweredMatArrays;
  delete[] rotatedDoutPtrArrays;

  // Inplace transpose addition
  int numRow = CRS;
  for (int t = 0; t < numOpenMPThreads; t++) {
    int iter = 0;
    T* temp1 = temp + numTempElem * t;
    for (int i = 0; i < CRS; i++) {
      for (int j = 0; j < K; j++, iter++) {
        int index = j * numRow + i;
        retPtr[index] += temp1[iter];
      }
    }
  }

  delete[] temp;
#endif
  return computeNNZ(retPtr, K * CRS);
}

int conv2dBackwardFilterDense(double* inputPtr, double* doutPtr, double* retPtr,
                              int N, int C, int H, int W, int K, int R, int S,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int P, int Q, int numThreads) {
  return conv2dBackwardFilterDenseHelper(inputPtr, doutPtr, retPtr, N, C, H, W,
                                         K, R, S, stride_h, stride_w, pad_h,
                                         pad_w, P, Q, numThreads);
}

int conv2dBackwardFilterDense(float* inputPtr, float* doutPtr, float* retPtr,
                              int N, int C, int H, int W, int K, int R, int S,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int P, int Q, int numThreads) {
  return conv2dBackwardFilterDenseHelper(inputPtr, doutPtr, retPtr, N, C, H, W,
                                         K, R, S, stride_h, stride_w, pad_h,
                                         pad_w, P, Q, numThreads);
}

template <typename T>
int conv2dBackwardDataDenseHelper(T* filterPtr, T* doutPtr, T* retPtr, int N,
                                  int C, int H, int W, int K, int R, int S,
                                  int stride_h, int stride_w, int pad_h,
                                  int pad_w, int P, int Q, int numThreads) {
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
  if (isSinglePrecision())
    CHECK_MKL_ERROR(dnnConvolutionCreateBackwardData_F32(
        &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, srcSize,
        dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros))
  else
    CHECK_MKL_ERROR(dnnConvolutionCreateBackwardData_F64(
        &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension, srcSize,
        dstSize, filterSize, convolutionStrides, pads, dnnBorderZeros))

  MKL_DNN_EXECUTE()

#else
  // First step: Avoids oversubscription and other openmp/internal blas
  // threading issues
  setNumThreadsForBLAS(1);

  int CRS = C * R * S;
  int PQ = P * Q;
  int KPQ = K * PQ;
  int numRotatedElem = PQ * K;
  int numCol2ImElem = PQ * CRS;

  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  T* rotatedDoutPtrArrays = new T[numRotatedElem * numOpenMPThreads];
  T* col2imInputArrays = new T[numCol2ImElem * numOpenMPThreads];

#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    int threadID = omp_get_thread_num();
    // Step 1: Rotate dout
    T* rotatedDoutPtr = rotatedDoutPtrArrays + numRotatedElem * threadID;
    rotate180(doutPtr + n * KPQ, rotatedDoutPtr, 1, C, H, W, K, R, S, stride_h,
              stride_w, pad_h, pad_w, P, Q);

    // Step 2: t(rotatedDout (PQ X K) %*% filter (K X CRS))
    T* col2imInput = col2imInputArrays + numCol2ImElem * threadID;
    matmult(rotatedDoutPtr, filterPtr, col2imInput, PQ, K, CRS, 1);

    // Step 3: Perform col2im
    T* outputArr = retPtr + n * CHW;
    col2im(col2imInput, outputArr, 1, C, H, W, K, R, S, stride_h, stride_w,
           pad_h, pad_w, P, Q);
  }  // end omp parallel for

  delete[] rotatedDoutPtrArrays;
  delete[] col2imInputArrays;
#endif
  return computeNNZ(retPtr, N * CHW);
}

int conv2dBackwardDataDense(double* filterPtr, double* doutPtr, double* retPtr,
                            int N, int C, int H, int W, int K, int R, int S,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int P, int Q, int numThreads) {
  return conv2dBackwardDataDenseHelper(filterPtr, doutPtr, retPtr, N, C, H, W,
                                       K, R, S, stride_h, stride_w, pad_h,
                                       pad_w, P, Q, numThreads);
}

int conv2dBackwardDataDense(float* filterPtr, float* doutPtr, float* retPtr,
                            int N, int C, int H, int W, int K, int R, int S,
                            int stride_h, int stride_w, int pad_h, int pad_w,
                            int P, int Q, int numThreads) {
  return conv2dBackwardDataDenseHelper(filterPtr, doutPtr, retPtr, N, C, H, W,
                                       K, R, S, stride_h, stride_w, pad_h,
                                       pad_w, P, Q, numThreads);
}

template <typename T>
void conv2dSparseHelper(int apos, int alen, int* aix, double* avals,
                        T* filterPtr, T* retPtr, int N, int C, int H, int W,
                        int K, int R, int S, int stride_h, int stride_w,
                        int pad_h, int pad_w, int P, int Q, int numThreads) {
  setNumThreadsForBLAS(1);
  T* loweredMat = new T[C * R * S * P * Q];

  // Step 1: Perform im2col
  T* temp = new T[C * H * W];
  std::size_t size = C * H * W * sizeof(T);
  std::memset(temp, 0, size);
  for (int j = apos; j < apos + alen; j++)
    temp[aix[j]] = static_cast<T>(avals[j]);
  im2col(temp, loweredMat, 1, C, H, W, K, R, S, stride_h, stride_w, pad_h,
         pad_w, P, Q);
  delete[] temp;

  // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
  matmult(filterPtr, loweredMat, retPtr, K, C * R * S, P * Q, 1);

  delete[] loweredMat;
}

void conv2dSparse(int apos, int alen, int* aix, double* avals, float* filter,
                  float* ret, int N, int C, int H, int W, int K, int R, int S,
                  int stride_h, int stride_w, int pad_h, int pad_w, int P,
                  int Q, int numThreads) {
  conv2dSparseHelper(apos, alen, aix, avals, filter, ret, N, C, H, W, K, R, S,
                     stride_h, stride_w, pad_h, pad_w, P, Q, numThreads);
}

void conv2dSparse(int apos, int alen, int* aix, double* avals, double* filter,
                  double* ret, int N, int C, int H, int W, int K, int R, int S,
                  int stride_h, int stride_w, int pad_h, int pad_w, int P,
                  int Q, int numThreads) {
  conv2dSparseHelper(apos, alen, aix, avals, filter, ret, N, C, H, W, K, R, S,
                     stride_h, stride_w, pad_h, pad_w, P, Q, numThreads);
}

template <typename T>
void conv2dBackwardFilterSparseDenseHelper(
    int apos, int alen, int* aix, double* avals, T* rotatedDoutPtr, T* retPtr,
    int N, int C, int H, int W, int K, int R, int S, int stride_h, int stride_w,
    int pad_h, int pad_w, int P, int Q, int numThreads) {
  setNumThreadsForBLAS(1);
  int CHW = C * H * W;
  int CRS = C * R * S;
  int PQ = P * Q;
  int KPQ = K * PQ;
  int m1 = CRS;
  int n1 = K;
  int k1 = PQ;

  T* loweredMat = new T[CRS * PQ];

  // Step 1: Perform im2col
  T* temp = new T[C * H * W];
  std::size_t size = C * H * W * sizeof(T);
  std::memset(temp, 0, size);
  for (int j = apos; j < apos + alen; j++)
    temp[aix[j]] = static_cast<T>(avals[j]);
  im2col(temp, loweredMat, 1, C, H, W, K, R, S, stride_h, stride_w, pad_h,
         pad_w, P, Q);
  delete[] temp;

  // Multiply to get CRS X K
  T* temp1 = new T[CRS * K];
  // Step 3: loweredMat (CRS X PQ) %*% rotatedDoutPtr (PQ X K)
  matmult(loweredMat, rotatedDoutPtr, temp1, C * R * S, P * Q, K, 1);
  delete[] loweredMat;

  // Inplace addition
  for (int iter = 0; iter < K * CRS; iter++) {
    retPtr[iter] += temp1[iter];
  }

  delete[] temp1;
}

void conv2dBackwardFilterSparseDense(int apos, int alen, int* aix,
                                     double* avals, double* rotatedDoutPtr,
                                     double* retPtr, int N, int C, int H, int W,
                                     int K, int R, int S, int stride_h,
                                     int stride_w, int pad_h, int pad_w, int P,
                                     int Q, int numThreads) {
  conv2dBackwardFilterSparseDenseHelper(
      apos, alen, aix, avals, rotatedDoutPtr, retPtr, N, C, H, W, K, R, S,
      stride_h, stride_w, pad_h, pad_w, P, Q, numThreads);
}

void conv2dBackwardFilterSparseDense(int apos, int alen, int* aix,
                                     double* avals, float* rotatedDoutPtr,
                                     float* retPtr, int N, int C, int H, int W,
                                     int K, int R, int S, int stride_h,
                                     int stride_w, int pad_h, int pad_w, int P,
                                     int Q, int numThreads) {
  conv2dBackwardFilterSparseDenseHelper(
      apos, alen, aix, avals, rotatedDoutPtr, retPtr, N, C, H, W, K, R, S,
      stride_h, stride_w, pad_h, pad_w, P, Q, numThreads);
}

template <typename T>
int conv2dBiasAddDenseHelper(T* inputPtr, T* biasPtr, T* filterPtr, T* retPtr,
                             int N, int C, int H, int W, int K, int R, int S,
                             int stride_h, int stride_w, int pad_h, int pad_w,
                             int P, int Q, bool addBias, int numThreads) {
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
  if (addBias) {
    if (isSinglePrecision())
      CHECK_MKL_ERROR(dnnConvolutionCreateForwardBias_F32(
          &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension,
          srcSize, dstSize, filterSize, convolutionStrides, pads,
          dnnBorderZeros))
    else
      CHECK_MKL_ERROR(dnnConvolutionCreateForwardBias_F64(
          &pConvolution, NULL, dnnAlgorithmConvolutionDirect, dimension,
          srcSize, dstSize, filterSize, convolutionStrides, pads,
          dnnBorderZeros))
    resources[dnnResourceBias] = biasPtr;
  } else {
    if (isSinglePrecision())
      CHECK_MKL_ERROR(dnnConvolutionCreateForward_F32(&pConvolution, NULL,
                                      dnnAlgorithmConvolutionDirect, dimension,
                                      srcSize, dstSize, filterSize,
                                      convolutionStrides, pads, dnnBorderZeros))
    else
      CHECK_MKL_ERROR(dnnConvolutionCreateForward_F64(&pConvolution, NULL,
                                      dnnAlgorithmConvolutionDirect, dimension,
                                      srcSize, dstSize, filterSize,
                                      convolutionStrides, pads, dnnBorderZeros))
  }

  MKL_DNN_EXECUTE()

#else
  // ------------------------------------------------------------------------------------
  // First step:  Avoids oversubscription and other openmp/internal blas
  // threading issues
  setNumThreadsForBLAS(1);

  int CHW = C * H * W;
  int PQ = P * Q;
  int numIm2ColElem = C * R * S * P * Q;

  // Allocate temporary data structures used in parallel for
  int numOpenMPThreads = MIN(numThreads, N);
  T* loweredMatArrays = new T[numIm2ColElem * numOpenMPThreads];

#pragma omp parallel for num_threads(numOpenMPThreads)
  for (int n = 0; n < N; n++) {
    int threadID = omp_get_thread_num();
    T* loweredMat = loweredMatArrays + numIm2ColElem * threadID;

    // Step 1: Perform im2col
    im2col(inputPtr + n * CHW, loweredMat, 1, C, H, W, K, R, S, stride_h,
           stride_w, pad_h, pad_w, P, Q);

    // Step 2: filter (K X CRS) %*% loweredMat (CRS X PQ)
    matmult(filterPtr, loweredMat, retPtr + n * KPQ, K, C * R * S, P * Q, 1);

    // Step 3: Add bias
    T* outputArr = retPtr + n * KPQ;
    if (addBias) {
      int index = 0;
      for (int k = 0; k < K; k++) {
        for (int pq = 0; pq < PQ; pq++, index++) {
          outputArr[index] += biasPtr[k];
        }
      }
    }
  }  // end omp parallel for
  delete[] loweredMatArrays;
// ------------------------------------------------------------------------------------
#endif

  return computeNNZ(retPtr, N * KPQ);
}

int conv2dBiasAddDense(double* inputPtr, double* biasPtr, double* filterPtr,
                       double* retPtr, int N, int C, int H, int W, int K, int R,
                       int S, int stride_h, int stride_w, int pad_h, int pad_w,
                       int P, int Q, bool addBias, int numThreads) {
  return conv2dBiasAddDenseHelper(inputPtr, biasPtr, filterPtr, retPtr, N, C, H,
                                  W, K, R, S, stride_h, stride_w, pad_h, pad_w,
                                  P, Q, addBias, numThreads);
}

int conv2dBiasAddDense(float* inputPtr, float* biasPtr, float* filterPtr,
                       float* retPtr, int N, int C, int H, int W, int K, int R,
                       int S, int stride_h, int stride_w, int pad_h, int pad_w,
                       int P, int Q, bool addBias, int numThreads) {
  return conv2dBiasAddDenseHelper(inputPtr, biasPtr, filterPtr, retPtr, N, C, H,
                                  W, K, R, S, stride_h, stride_w, pad_h, pad_w,
                                  P, Q, addBias, numThreads);
}
