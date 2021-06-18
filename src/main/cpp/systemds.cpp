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

#include "common.h"
#include "libmatrixdnn.h"
#include "libmatrixmult.h"
#include "systemds.h"

// Results from Matrix-vector/vector-matrix 1M x 1K, dense show that GetDoubleArrayElements creates a copy on OpenJDK.

// Logic:
// 1. We chose GetDoubleArrayElements over GetPrimitiveArrayCritical in a multi-threaded scenario. This avoids any potential OOM related to GC halts.
// 2. For input array, we don't copy back the array using JNI_ABORT.

// JNI Methods to get/release double*
#define GET_DOUBLE_ARRAY(env, input, numThreads) \
	((double*)env->GetPrimitiveArrayCritical(input, NULL))
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ? ((double*)env->GetPrimitiveArrayCritical(input, NULL)) :  env->GetDoubleArrayElements(input,NULL) )
 
// ------------------------------------------------------------------- 
// From: https://developer.android.com/training/articles/perf-jni.html
// 0
// Actual: the array object is un-pinned.
// Copy: data is copied back. The buffer with the copy is freed.
// JNI_ABORT
// Actual: the array object is un-pinned. Earlier writes are not aborted.
// Copy: the buffer with the copy is freed; any changes to it are lost.
#define RELEASE_INPUT_ARRAY(env, input, inputPtr, numThreads) \
	env->ReleasePrimitiveArrayCritical(input, inputPtr, JNI_ABORT)
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ? env->ReleasePrimitiveArrayCritical(input, inputPtr, JNI_ABORT) : env->ReleaseDoubleArrayElements(input, inputPtr, JNI_ABORT) )

#define RELEASE_ARRAY(env, input, inputPtr, numThreads) \
	env->ReleasePrimitiveArrayCritical(input, inputPtr, 0)
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ? env->ReleasePrimitiveArrayCritical(input, inputPtr, 0) :  env->ReleaseDoubleArrayElements(input, inputPtr, 0) )
  
// -------------------------------------------------------------------
JNIEXPORT void JNICALL Java_org_apache_sysds_utils_NativeHelper_setMaxNumThreads
  (JNIEnv *, jclass, jint jmaxThreads) {
	setNumThreadsForBLAS(jmaxThreads);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_dmmdd(
    JNIEnv* env, jclass cls, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret,
    jint m1rlen, jint m1clen, jint m2clen, jint numThreads)
{
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1, numThreads);
  double* m2Ptr = GET_DOUBLE_ARRAY(env, m2, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(m1Ptr == NULL || m2Ptr == NULL || retPtr == NULL)
    return -1;

  dmatmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen, (int)numThreads);
  size_t nnz = computeNNZ<double>(retPtr, m1rlen * m2clen);

  RELEASE_INPUT_ARRAY(env, m1, m1Ptr, numThreads);
  RELEASE_INPUT_ARRAY(env, m2, m2Ptr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads); 

  return static_cast<jlong>(nnz);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_smmdd(
    JNIEnv* env, jclass cls, jobject m1, jobject m2, jobject ret,
    jint m1rlen, jint m1clen, jint m2clen, jint numThreads)
{
  float* m1Ptr = (float*) env->GetDirectBufferAddress(m1);
  float* m2Ptr = (float*) env->GetDirectBufferAddress(m2);
  float* retPtr = (float*) env->GetDirectBufferAddress(ret);
  if(m1Ptr == NULL || m2Ptr == NULL || retPtr == NULL)
    return -1;

  smatmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen, (int)numThreads);

  return static_cast<jlong>(computeNNZ<float>(retPtr, m1rlen * m2clen));
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_tsmm
  (JNIEnv * env, jclass cls, jdoubleArray m1, jdoubleArray ret, jint m1rlen, jint m1clen, jboolean leftTrans, jint numThreads) {
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(m1Ptr == NULL || retPtr == NULL)
    return -1;

  tsmm(m1Ptr, retPtr, (int)m1rlen, (int)m1clen, (bool)leftTrans, (int)numThreads);

  int n = leftTrans ? m1clen : m1rlen;
  size_t nnz = computeNNZ<double>(retPtr, n * n);

  RELEASE_INPUT_ARRAY(env, m1, m1Ptr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads);

  return static_cast<jlong>(nnz);
}

JNIEXPORT jboolean JNICALL Java_org_apache_sysds_utils_NativeHelper_conv2dSparse
		(JNIEnv * env, jclass, jint apos, jint alen, jintArray aix, jdoubleArray avals, jdoubleArray filter,
		jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
		jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  int* aixPtr = ((int*)env->GetPrimitiveArrayCritical(aix, NULL));
  double* avalsPtr = GET_DOUBLE_ARRAY(env, avals, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  
  conv2dSparse((int)apos, (int)alen, aixPtr, avalsPtr, filterPtr, retPtr, (int)N, (int)C, (int)H, (int)W, 
		(int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P, (int)Q, (int)numThreads);
  
  RELEASE_INPUT_ARRAY(env, avals, avalsPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, filter, filterPtr, numThreads);
  env->ReleasePrimitiveArrayCritical(aix, aixPtr, JNI_ABORT);
  RELEASE_ARRAY(env, ret, retPtr, numThreads); 
  return (jboolean) true;
}

JNIEXPORT jboolean JNICALL Java_org_apache_sysds_utils_NativeHelper_conv2dBackwardFilterSparseDense
		(JNIEnv * env, jclass, jint apos, jint alen, jintArray aix, jdoubleArray avals, jdoubleArray dout,  
		jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
		jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  int* aixPtr = ((int*)env->GetPrimitiveArrayCritical(aix, NULL));
  double* avalsPtr = GET_DOUBLE_ARRAY(env, avals, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  
  conv2dBackwardFilterSparseDense((int)apos, (int)alen, aixPtr, avalsPtr, doutPtr, retPtr, (int)N, (int)C, (int)H, (int)W, 
		(int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P, (int)Q, (int)numThreads);
  
  RELEASE_INPUT_ARRAY(env, avals, avalsPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, dout, doutPtr, numThreads);
  env->ReleasePrimitiveArrayCritical(aix, aixPtr, JNI_ABORT);
  RELEASE_ARRAY(env, ret, retPtr, numThreads); 
  return (jboolean) true;
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_conv2dDense(
		JNIEnv* env, jclass, jdoubleArray input, jdoubleArray filter,
		jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
		jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads)
{
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(inputPtr == NULL || filterPtr == NULL || retPtr == NULL)
    return -1;
  
  size_t nnz = dconv2dBiasAddDense(inputPtr, 0, filterPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K,
		(int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P,
		(int) Q, false, (int) numThreads);
  
  RELEASE_INPUT_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads);
  return static_cast<jlong>(nnz);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_dconv2dBiasAddDense(
		JNIEnv* env, jclass, jdoubleArray input, jdoubleArray bias, jdoubleArray filter,
		jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
		jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads)
{
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* biasPtr = GET_DOUBLE_ARRAY(env, bias, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(inputPtr == NULL || biasPtr == NULL || filterPtr == NULL || retPtr == NULL)
	return -1;
  
  size_t nnz = dconv2dBiasAddDense(inputPtr, biasPtr, filterPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K,
		(int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P,
		(int) Q, true, (int) numThreads);
  
  RELEASE_INPUT_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, bias, biasPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads);
  return static_cast<jlong>(nnz);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_sconv2dBiasAddDense(
		JNIEnv* env, jclass, jobject input, jobject bias, jobject filter,
		jobject ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
		jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) 
{
  float* inputPtr = (float*) env->GetDirectBufferAddress(input);
  float* biasPtr =  (float*) env->GetDirectBufferAddress(bias);
  float* filterPtr = (float*) env->GetDirectBufferAddress(filter);
  float* retPtr = (float*) env->GetDirectBufferAddress(ret);
  if(inputPtr == NULL || biasPtr == NULL || filterPtr == NULL || retPtr == NULL)
    return -1;
  
  size_t nnz = sconv2dBiasAddDense(inputPtr, biasPtr, filterPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K,
    (int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P,
		(int) Q, true, (int) numThreads);

  return static_cast<jlong>(nnz);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_conv2dBackwardDataDense(
    JNIEnv* env, jclass, jdoubleArray filter, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {

  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(doutPtr == NULL || filterPtr == NULL || retPtr == NULL)
    return -1;

  size_t nnz = conv2dBackwardDataDense(filterPtr, doutPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K,
    (int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w,
    (int) P, (int) Q, (int) numThreads);

  RELEASE_INPUT_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, dout, doutPtr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads);
  return static_cast<jlong>(nnz);
}

JNIEXPORT jlong JNICALL Java_org_apache_sysds_utils_NativeHelper_conv2dBackwardFilterDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if(doutPtr == NULL || inputPtr == NULL || retPtr == NULL)
    return -1;

  size_t nnz = conv2dBackwardFilterDense(inputPtr, doutPtr, retPtr, (int)N, (int) C, (int) H, (int) W, (int) K, (int) R,
    (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P,
    (int) Q, (int) numThreads);

  RELEASE_INPUT_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_ARRAY(env, dout, doutPtr, numThreads);
  RELEASE_ARRAY(env, ret, retPtr, numThreads);
  return static_cast<jlong>(nnz);
}
