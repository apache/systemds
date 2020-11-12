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

#ifndef LIBMATRIXMULT_H
#define LIBMATRIXMULT_H

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape (m1rlen, m1clen) and (m1clen, m2clen)
void dmatmult(double *m1Ptr, double *m2Ptr, double *retPtr, int m, int k, int n, int numThreads);

// Same matrix multiply for single precision
void smatmult(float *m1Ptr, float *m2Ptr, float *retPtr, int m, int k, int n, int numThreads);

// Matrix transpose
void tsmm(double *m1Ptr, double *retPtr, int m1rlen, int m1clen, bool isLeftTrans, int numThreads);

#endif // LIBMATRIXMULT_H
