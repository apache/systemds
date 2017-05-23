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
#include "systemml.h"
#include "libmatrixmult.h"
#include "libmatrixdnn.h"

// Linux:
// g++ -o lib/libsystemml_mkl-Linux-x86_64.so *.cpp  -I$JAVA_HOME/include -I$MKLROOT/include -I$JAVA_HOME/include/linux -lmkl_rt -lpthread  -lm -ldl -DUSE_INTEL_MKL -DUSE_MKL_DNN -L$MKLROOT/lib/intel64 -m64 -O3 -shared -fPIC
// g++ -o lib/libsystemml_openblas-Linux-x86_64.so *.cpp  -I$JAVA_HOME/include  -I$JAVA_HOME/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib/ -fopenmp -O3 -shared -fPIC

// Mac OSX:	
// g++ -o libsystemml_mkl-linux-x86_64.dylib *.cpp  -I$JAVA_HOME/include -I$MKLROOT/include -I$JAVA_HOME/include/linux -lmkl_rt -lpthread  -lm -ldl -DUSE_INTEL_MKL -DUSE_GNU_THREADING -L$MKLROOT/lib/intel64 -m64 -fopenmp -O3 -dynamiclib -fPIC -undefined dynamic_lookup
// g++ -o libsystemml_openblas-linux-x86_64.dylib *.cpp  -I$JAVA_HOME/include  -I$JAVA_HOME/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -L/opt/OpenBLAS/lib/ -fopenmp -O3 -dynamiclib -fPIC -undefined dynamic_lookup

// Windows MKL: 
// "C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
// "%MKLROOT%"\bin\mklvars.bat intel64
// set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
// cl *.cpp -I. -I"%MKLROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -DUSE_INTEL_MKL -Fesystemml_mkl-windows-x86_64.dll -MD -LD  "%MKLROOT%"\lib\intel64_win\mkl_intel_thread_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_core_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_intel_lp64_dll.lib
// Windows OpenBLAS:
// "C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
// set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
// cl *.cpp -I. -I"%OPENBLASROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -DUSE_OPEN_BLAS -Fesystemml_openblas-windows-x86_64.dll -MD -LD "%OPENBLASROOT%"\lib\libopenblas.dll.a

// Results from Matrix-vector/vector-matrix 1M x 1K, dense show that
// GetDoubleArrayElements creates a copy on OpenJDK.

// Logic:
// 1. We chose GetDoubleArrayElements over GetPrimitiveArrayCritical in a
// multi-threaded scenario. This avoids any potential OOM related to GC halts.
// 2. For input array, we don't copy back the array using JNI_ABORT.

// JNI Methods to get/release double*
#define GET_DOUBLE_ARRAY(env, input, numThreads) \
  ((double*)env->GetPrimitiveArrayCritical(input, NULL))
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ?
// ((double*)env->GetPrimitiveArrayCritical(input, NULL)) :
// env->GetDoubleArrayElements(input,NULL) )

#define GET_FLOAT_ARRAY(env, input, numThreads) \
  ((float*)env->GetPrimitiveArrayCritical(input, NULL))

// -------------------------------------------------------------------
// From: https://developer.android.com/training/articles/perf-jni.html
// 0
// Actual: the array object is un-pinned.
// Copy: data is copied back. The buffer with the copy is freed.
// JNI_ABORT
// Actual: the array object is un-pinned. Earlier writes are not aborted.
// Copy: the buffer with the copy is freed; any changes to it are lost.
#define RELEASE_INPUT_DOUBLE_ARRAY(env, input, inputPtr, numThreads) \
  env->ReleasePrimitiveArrayCritical(input, inputPtr, JNI_ABORT)
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ?
// env->ReleasePrimitiveArrayCritical(input, inputPtr, JNI_ABORT) :
// env->ReleaseDoubleArrayElements(input, inputPtr, JNI_ABORT) )

// For consistency
#define RELEASE_INPUT_FLOAT_ARRAY(env, input, inputPtr, numThreads) \
  env->ReleasePrimitiveArrayCritical(input, inputPtr, JNI_ABORT)

#define RELEASE_DOUBLE_ARRAY(env, input, inputPtr, numThreads) \
  env->ReleasePrimitiveArrayCritical(input, inputPtr, 0)
// ( maxThreads != -1 && ((int)numThreads) == maxThreads ?
// env->ReleasePrimitiveArrayCritical(input, inputPtr, 0) :
// env->ReleaseDoubleArrayElements(input, inputPtr, 0) )

// -------------------------------------------------------------------

int maxThreads = -1;
JNIEXPORT void JNICALL
Java_org_apache_sysml_utils_NativeHelper_setMaxNumThreads(JNIEnv*, jclass,
                                                          jint jmaxThreads) {
  maxThreads = (int)jmaxThreads;
}

JNIEXPORT void JNICALL
Java_org_apache_sysml_utils_NativeHelper_setFloatDatatype(
    JNIEnv* env, jclass cls, jboolean useFloatDataType1) {
  setSinglePrecision((bool)useFloatDataType1);
}

void copyFP32ToFP64(float* src, double* dest, int size) {
  for (int i = 0; i < size; i++) {
    dest[i] = static_cast<double>(src[i]);
  }
}

void copyFP64ToFP32(double* src, float* dest, int size) {
  for (int i = 0; i < size; i++) {
    dest[i] = static_cast<float>(src[i]);
  }
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysml_utils_NativeHelper_matrixMultDenseDense(
    JNIEnv* env, jclass cls, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret,
    jint m1rlen, jint m1clen, jint m2clen, jint numThreads) {
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1, numThreads);
  double* m2Ptr = GET_DOUBLE_ARRAY(env, m2, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (m1Ptr == NULL || m2Ptr == NULL || retPtr == NULL) return (jboolean) false;

  if (isSinglePrecision()) {
    int m1PtrLen = (int)m1rlen * (int)m1clen;
    int m2PtrLen = (int)m1clen * (int)m2clen;
    int retPtrLen = (int)m1rlen * (int)m2clen;
    float* m1PtrFP32 = new float[m1PtrLen];
    float* m2PtrFP32 = new float[m2PtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(m1Ptr, m1PtrFP32, m1PtrLen);
    copyFP64ToFP32(m2Ptr, m2PtrFP32, m2PtrLen);
    matmult(m1PtrFP32, m2PtrFP32, retPtrFP32, (int)m1rlen, (int)m1clen,
            (int)m2clen, (int)numThreads);
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] m1PtrFP32;
    delete[] m2PtrFP32;
    delete[] retPtrFP32;
  } else {
    matmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen,
            (int)numThreads);
  }

  RELEASE_INPUT_DOUBLE_ARRAY(env, m1, m1Ptr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, m2, m2Ptr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jboolean) true;
}

JNIEXPORT jboolean JNICALL Java_org_apache_sysml_utils_NativeHelper_tsmm(
    JNIEnv* env, jclass cls, jdoubleArray m1, jdoubleArray ret, jint m1rlen,
    jint m1clen, jboolean isLeftTranspose, jint numThreads) {
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (m1Ptr == NULL || retPtr == NULL) return (jboolean) false;

  if (isSinglePrecision()) {
    int m1PtrLen = (int)m1rlen * (int)m1clen;
    int retPtrLen = ((bool)isLeftTranspose) ? ((int)m1clen * (int)m1clen)
                                            : ((int)m1rlen * (int)m1rlen);
    float* m1PtrFP32 = new float[m1PtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(m1Ptr, m1PtrFP32, m1PtrLen);
    tsmm(m1PtrFP32, retPtrFP32, (int)m1rlen, (int)m1clen, (bool)isLeftTranspose,
         (int)numThreads);
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] m1PtrFP32;
    delete[] retPtrFP32;
  } else {
    tsmm(m1Ptr, retPtr, (int)m1rlen, (int)m1clen, (bool)isLeftTranspose,
         (int)numThreads);
  }

  RELEASE_INPUT_DOUBLE_ARRAY(env, m1, m1Ptr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jboolean) true;
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dSparse(
    JNIEnv* env, jclass, jint apos, jint alen, jintArray aix,
    jdoubleArray avals, jdoubleArray filter, jdoubleArray ret, jint N, jint C,
    jint H, jint W, jint K, jint R, jint S, jint stride_h, jint stride_w,
    jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  int* aixPtr = ((int*)env->GetPrimitiveArrayCritical(aix, NULL));
  double* avalsPtr = GET_DOUBLE_ARRAY(env, avals, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);

  conv2dSparse((int)apos, (int)alen, aixPtr, avalsPtr, filterPtr, retPtr,
               (int)N, (int)C, (int)H, (int)W, (int)K, (int)R, (int)S,
               (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P,
               (int)Q, (int)numThreads);

  RELEASE_INPUT_DOUBLE_ARRAY(env, avals, avalsPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, filter, filterPtr, numThreads);
  env->ReleasePrimitiveArrayCritical(aix, aixPtr, JNI_ABORT);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jboolean) true;
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dSparseFP32(
    JNIEnv* env, jclass, jint apos, jint alen, jintArray aix,
    jdoubleArray avals, jfloatArray filter, jdoubleArray ret, jint N, jint C,
    jint H, jint W, jint K, jint R, jint S, jint stride_h, jint stride_w,
    jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  int* aixPtr = ((int*)env->GetPrimitiveArrayCritical(aix, NULL));
  double* avalsPtr = GET_DOUBLE_ARRAY(env, avals, numThreads);
  float* filterPtr = GET_FLOAT_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);

  int retPtrLen = (int)N * (int)K * (int)P * (int)Q;
  float* retPtrFP32 = new float[retPtrLen];
  conv2dSparse((int)apos, (int)alen, aixPtr, avalsPtr, filterPtr, retPtrFP32,
               (int)N, (int)C, (int)H, (int)W, (int)K, (int)R, (int)S,
               (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P,
               (int)Q, (int)numThreads);
  copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
  delete[] retPtrFP32;

  RELEASE_INPUT_DOUBLE_ARRAY(env, avals, avalsPtr, numThreads);
  RELEASE_INPUT_FLOAT_ARRAY(env, filter, filterPtr, numThreads);
  env->ReleasePrimitiveArrayCritical(aix, aixPtr, JNI_ABORT);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jboolean) true;
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dBackwardFilterSparseDense(
    JNIEnv* env, jclass, jint apos, jint alen, jintArray aix,
    jdoubleArray avals, jdoubleArray dout, jdoubleArray ret, jint N, jint C,
    jint H, jint W, jint K, jint R, jint S, jint stride_h, jint stride_w,
    jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  int* aixPtr = ((int*)env->GetPrimitiveArrayCritical(aix, NULL));
  double* avalsPtr = GET_DOUBLE_ARRAY(env, avals, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);

  if (isSinglePrecision()) {
    int doutPtrLen = (int)N * (int)K * (int)P * (int)Q;
    int retPtrLen = (int)K * (int)C * (int)R * (int)S;
    float* doutPtrFP32 = new float[doutPtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(doutPtr, doutPtrFP32, doutPtrLen);
    conv2dBackwardFilterSparseDense(
        (int)apos, (int)alen, aixPtr, avalsPtr, doutPtrFP32, retPtrFP32, (int)N,
        (int)C, (int)H, (int)W, (int)K, (int)R, (int)S, (int)stride_h,
        (int)stride_w, (int)pad_h, (int)pad_w, (int)P, (int)Q, (int)numThreads);
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] doutPtrFP32;
    delete[] retPtrFP32;
  } else {
    conv2dBackwardFilterSparseDense(
        (int)apos, (int)alen, aixPtr, avalsPtr, doutPtr, retPtr, (int)N, (int)C,
        (int)H, (int)W, (int)K, (int)R, (int)S, (int)stride_h, (int)stride_w,
        (int)pad_h, (int)pad_w, (int)P, (int)Q, (int)numThreads);
  }
  RELEASE_INPUT_DOUBLE_ARRAY(env, avals, avalsPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, dout, doutPtr, numThreads);
  env->ReleasePrimitiveArrayCritical(aix, aixPtr, JNI_ABORT);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jboolean) true;
}

JNIEXPORT jint JNICALL Java_org_apache_sysml_utils_NativeHelper_conv2dDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q,
    jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (inputPtr == NULL || filterPtr == NULL || retPtr == NULL) return (jint)-1;
  
  int nnz = -1;
  if (isSinglePrecision()) {
    int inputPtrLen = (int)N * (int)C * (int)H * (int)W;
    int filterPtrLen = (int)K * (int)C * (int)R * (int)S;
    int retPtrLen = (int)N * (int)K * (int)P * (int)Q;
    float* inputPtrFP32 = new float[inputPtrLen];
    float* filterPtrFP32 = new float[filterPtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(inputPtr, inputPtrFP32, inputPtrLen);
    copyFP64ToFP32(filterPtr, filterPtrFP32, filterPtrLen);
    // to avoid template argument deduction/substitution failed error while compilation.
    float* ignoreBiasPtr = filterPtrFP32;
    nnz = conv2dBiasAddDense(
      inputPtrFP32, ignoreBiasPtr, filterPtrFP32, retPtrFP32, (int)N, (int)C, (int)H,
      (int)W, (int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h,
      (int)pad_w, (int)P, (int)Q, false, (int)numThreads);
    delete[] inputPtrFP32;
    delete[] filterPtrFP32;
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] retPtrFP32;
  } else {
  	// to avoid template argument deduction/substitution failed error while compilation.
    double* ignoreBiasPtr = filterPtr;
  	nnz = conv2dBiasAddDense(
      inputPtr, ignoreBiasPtr, filterPtr, retPtr, (int)N, (int)C, (int)H,
      (int)W, (int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h,
      (int)pad_w, (int)P, (int)Q, false, (int)numThreads);
  }

  RELEASE_INPUT_DOUBLE_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jint)nnz;
}

JNIEXPORT jint JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dBiasAddDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray bias,
    jdoubleArray filter, jdoubleArray ret, jint N, jint C, jint H, jint W,
    jint K, jint R, jint S, jint stride_h, jint stride_w, jint pad_h,
    jint pad_w, jint P, jint Q, jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* biasPtr = GET_DOUBLE_ARRAY(env, bias, numThreads);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (inputPtr == NULL || biasPtr == NULL || filterPtr == NULL ||
      retPtr == NULL)
    return (jint)-1;

  int nnz = -1;
  if (isSinglePrecision()) {
    int inputPtrLen = (int)N * (int)C * (int)H * (int)W;
    int filterPtrLen = (int)K * (int)C * (int)R * (int)S;
    int biasPtrLen = (int)K;
    int retPtrLen = (int)N * (int)K * (int)P * (int)Q;
    float* inputPtrFP32 = new float[inputPtrLen];
    float* filterPtrFP32 = new float[filterPtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    float* biasPtrFP32 = new float[biasPtrLen];
    copyFP64ToFP32(inputPtr, inputPtrFP32, inputPtrLen);
    copyFP64ToFP32(biasPtr, biasPtrFP32, biasPtrLen);
    copyFP64ToFP32(filterPtr, filterPtrFP32, filterPtrLen);
    nnz = conv2dBiasAddDense(
      inputPtrFP32, biasPtrFP32, filterPtrFP32, retPtrFP32, (int)N, (int)C, (int)H,
      (int)W, (int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h,
      (int)pad_w, (int)P, (int)Q, true, (int)numThreads);
    delete[] inputPtrFP32;
    delete[] filterPtrFP32;
    delete[] biasPtrFP32;
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] retPtrFP32;
  } else {
  	nnz = conv2dBiasAddDense(
      inputPtr, biasPtr, filterPtr, retPtr, (int)N, (int)C, (int)H,
      (int)W, (int)K, (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h,
      (int)pad_w, (int)P, (int)Q, true, (int)numThreads);
  }

  RELEASE_INPUT_DOUBLE_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, bias, biasPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jint)nnz;
}

JNIEXPORT jint JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dBackwardDataDense(
    JNIEnv* env, jclass, jdoubleArray filter, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q,
    jint numThreads) {
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (doutPtr == NULL || filterPtr == NULL || retPtr == NULL) return (jint)-1;

  int nnz = -1;
  if (isSinglePrecision()) {
    int doutPtrLen = (int)N * (int)K * (int)P * (int)Q;
    int filterPtrLen = (int)K * (int)C * (int)R * (int)S;
    int retPtrLen = (int)N * (int)C * (int)H * (int)W;
    float* doutPtrFP32 = new float[doutPtrLen];
    float* filterPtrFP32 = new float[filterPtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(doutPtr, doutPtrFP32, doutPtrLen);
    copyFP64ToFP32(filterPtr, filterPtrFP32, filterPtrLen);
    nnz = conv2dBackwardDataDense(
      filterPtrFP32, doutPtrFP32, retPtrFP32, (int)N, (int)C, (int)H, (int)W, (int)K,
      (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
      (int)P, (int)Q, (int)numThreads);
    delete[] doutPtrFP32;
    delete[] filterPtrFP32;
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] retPtrFP32;
  } else {
  	nnz = conv2dBackwardDataDense(
      filterPtr, doutPtr, retPtr, (int)N, (int)C, (int)H, (int)W, (int)K,
      (int)R, (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w,
      (int)P, (int)Q, (int)numThreads);
  }
  
  RELEASE_INPUT_DOUBLE_ARRAY(env, filter, filterPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, dout, doutPtr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jint)nnz;
}

JNIEXPORT jint JNICALL
Java_org_apache_sysml_utils_NativeHelper_conv2dBackwardFilterDense(
    JNIEnv* env, jclass, jdoubleArray input, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q,
    jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input, numThreads);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout, numThreads);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret, numThreads);
  if (doutPtr == NULL || inputPtr == NULL || retPtr == NULL) return (jint)-1;

  int nnz = -1;
  if (isSinglePrecision()) {
    int doutPtrLen = (int)N * (int)K * (int)P * (int)Q;
    int inputPtrLen = (int)N * (int)C * (int)H * (int)W;
    int retPtrLen = (int)K * (int)C * (int)R * (int)S;
    float* inputPtrFP32 = new float[inputPtrLen];
    float* doutPtrFP32 = new float[doutPtrLen];
    float* retPtrFP32 = new float[retPtrLen];
    copyFP64ToFP32(doutPtr, doutPtrFP32, doutPtrLen);
    copyFP64ToFP32(inputPtr, inputPtrFP32, inputPtrLen);
    nnz = conv2dBackwardFilterDense(
      inputPtrFP32, doutPtrFP32, retPtrFP32, (int)N, (int)C, (int)H, (int)W, (int)K, (int)R,
      (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P,
      (int)Q, (int)numThreads);
    delete[] doutPtrFP32;
    delete[] inputPtrFP32;
    copyFP32ToFP64(retPtrFP32, retPtr, retPtrLen);
    delete[] retPtrFP32;
  } else {
  	nnz = conv2dBackwardFilterDense(
      inputPtr, doutPtr, retPtr, (int)N, (int)C, (int)H, (int)W, (int)K, (int)R,
      (int)S, (int)stride_h, (int)stride_w, (int)pad_h, (int)pad_w, (int)P,
      (int)Q, (int)numThreads);
  }

  RELEASE_INPUT_DOUBLE_ARRAY(env, input, inputPtr, numThreads);
  RELEASE_INPUT_DOUBLE_ARRAY(env, dout, doutPtr, numThreads);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr, numThreads);
  return (jint)nnz;
}
