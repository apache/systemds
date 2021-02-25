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
#ifndef LIBMATRIXDNN_H
#define LIBMATRIXDNN_H

#include <cstddef> 

size_t conv2dBackwardFilterDense(double* inputPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K,
							  int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q,
							  int numThreads);

size_t conv2dBackwardDataDense(double* filterPtr, double* doutPtr, double* retPtr, int N, int C, int H, int W, int K,
							int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q,
							int numThreads);

size_t dconv2dBiasAddDense(double* inputPtr, double* biasPtr, double* filterPtr, double* retPtr, int N, int C, int H,
						int W, int K, int R, int S,int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q,
						bool addBias, int numThreads);

size_t sconv2dBiasAddDense(float* inputPtr, float* biasPtr, float* filterPtr, float* retPtr, int N, int C, int H, int W,
						int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q,
						bool addBias, int numThreads);

void conv2dSparse(int apos, int alen, int* aix, double* avals, double* filter, double* ret, int N, int C, int H, int W, 
				  int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);

void conv2dBackwardFilterSparseDense(int apos, int alen, int* aix, double* avals, double* rotatedDoutPtr, double* ret,
									 int N, int C, int H, int W, int K, int R, int S, int stride_h, int stride_w,
									 int pad_h, int pad_w, int P, int Q, int numThreads);
			
#endif // LIBMATRIXDNN_H
