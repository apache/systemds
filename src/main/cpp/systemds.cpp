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

#ifdef _WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif

#include "common.h"
#include "libmatrixdnn.h"
#include "libmatrixmult.h"
#include "systemds.h"
#include "imgUtils.h"
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
/*JNIEXPORT void JNICALL Java_org_apache_sysds_utils_NativeHelper_setMaxNumThreads
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
*/
JNIEXPORT void JNICALL Java_org_apache_sysds_utils_NativeHelper_testNativeBindingWithDgemm
    (JNIEnv* env, jclass cls, jchar transa, jchar transb, jint m, jint n, jint k, jdouble alpha, jdoubleArray A, jint lda, jdoubleArray B, jint ldb, jdouble beta, jdoubleArray C, jint ldc) {

    // Obtain native pointers to Java arrays
    jdouble* nativeA = env->GetDoubleArrayElements(A, NULL);
    jdouble* nativeB = env->GetDoubleArrayElements(B, NULL);
    jdouble* nativeC = env->GetDoubleArrayElements(C, NULL);

    // Perform matrix multiplication using Intel MKL library
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, nativeA, lda, nativeB, ldb, beta, nativeC, ldc);

    // Release the native pointers
    env->ReleaseDoubleArrayElements(A, nativeA, 0);
    env->ReleaseDoubleArrayElements(B, nativeB, 0);
    env->ReleaseDoubleArrayElements(C, nativeC, 0);
}

JNIEXPORT void JNICALL Java_org_apache_sysds_utils_NativeHelper_imageRotate
    (JNIEnv* env, jclass clazz, jdoubleArray img_in, jint rows, jint cols, jdouble radians, jdouble fill_value, jdoubleArray img_out) {
    // Get input image data

    jsize num_pixels = env->GetArrayLength(img_in);
    jdouble* img_in_data = env->GetDoubleArrayElements(img_in, NULL);

    // Create output image array
    jdouble* img_out_data = new jdouble[num_pixels];

    // Rotate the image
    imageRotate(img_in_data, rows, cols, radians, fill_value, img_out_data);

    // Set the output image data
    env->SetDoubleArrayRegion(img_out, 0, num_pixels, img_out_data);

    // Clean up
    delete[] img_out_data;
    env->ReleaseDoubleArrayElements(img_in, img_in_data, JNI_ABORT);
}

JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_NativeHelper_imageCutout
    (JNIEnv* env, jclass cls, jdoubleArray img_in, jint rows, jint cols, jint x, jint y, jint width, jint height, jdouble fill_value) {
    // Convert the Java 1D array to a double*
    jdouble* img_in_arr = env->GetDoubleArrayElements(img_in, nullptr);

    // Call the C++ implementation of the Image Cutout function
    jdouble* img_out_arr = imageCutout(img_in_arr, rows, cols, x, y, width, height, fill_value);

    // Convert the double* back to a Java 1D array
    jdoubleArray img_out = env->NewDoubleArray(rows * cols);
    env->SetDoubleArrayRegion(img_out, 0, rows * cols, img_out_arr);

    // Release the native array reference
    env->ReleaseDoubleArrayElements(img_in, img_in_arr, JNI_ABORT);
    delete[] img_out_arr;

    return img_out;
}

JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_NativeHelper_cropImage(JNIEnv *env, jclass,
    jdoubleArray img_in, jint orig_w, jint orig_h, jint w, jint h, jint x_offset, jint y_offset) {
    jsize length = env->GetArrayLength(img_in);
    double *img_in_array = env->GetDoubleArrayElements(img_in, 0);

    int start_h = (ceil((orig_h - h) / 2)) + y_offset - 1;
    int end_h = (start_h + h - 1);
    int start_w = (ceil((orig_w - w) / 2)) + x_offset - 1;
    int end_w = (start_w + w - 1);

    jdoubleArray img_out_java;
    if (start_h < 0 || end_h >= orig_h || start_w < 0 || end_w >= orig_w) {
      img_out_java = env->NewDoubleArray(orig_w * orig_h);
      env->SetDoubleArrayRegion(img_out_java, 0, orig_w * orig_h, img_in_array);

    }else {
        double *img_out = imageCrop(img_in_array, orig_w, orig_h, w, h, x_offset, y_offset);

         if (img_out == nullptr) {
                 return nullptr;
             }

         img_out_java = env->NewDoubleArray(w * h);
         env->SetDoubleArrayRegion(img_out_java, 0, w * h, img_out);
         delete[] img_out;
     }

     env->ReleaseDoubleArrayElements(img_in, img_in_array, 0);

    return img_out_java;
}

/*JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_NativeHelper_shearImage(JNIEnv *env, jclass,
    jdoubleArray img_in, jint width, jint height, jdouble shear_x, jdouble shear_y, jdouble fill_value) {

    // Convert the Java input double array to a C++ double array
    jsize length = env->GetArrayLength(img_in);
    double *img_in_array = env->GetDoubleArrayElements(img_in, 0);

    // Call the C++ m_img_shear function (assuming it is implemented)
    double* img_out = img_transform(img_in_array, width, height, shear_x, shear_y, fill_value);

    // Release the input double array elements
    env->ReleaseDoubleArrayElements(img_in, img_in_array, 0);

    if (img_out == nullptr) {
        return nullptr;
    }

    // Create a new Java double array and copy the result into it
    jdoubleArray img_out_java = env->NewDoubleArray(width * height);
    env->SetDoubleArrayRegion(img_out_java, 0, width * height, img_out);

    // Free memory for the output image
    delete[] img_out;

    return img_out_java;
}*/

JNIEXPORT void JNICALL Java_org_apache_sysds_utils_NativeHelper_imgTranslate(JNIEnv *env, jclass cls,
                                                          jdoubleArray img_in, jdouble offset_x, jdouble offset_y,
                                                          jint in_w, jint in_h, jint out_w, jint out_h,
                                                          jdouble fill_value, jdoubleArray img_out) {
    // Convert Java arrays to C++ arrays
    jdouble* j_img_in = env->GetDoubleArrayElements(img_in, nullptr);
    jdouble* j_img_out = env->GetDoubleArrayElements(img_out, nullptr);

    // Call your C++ ImageTranslate function
    img_translate(j_img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, j_img_out);

    // Release Java arrays
    env->ReleaseDoubleArrayElements(img_in, j_img_in, 0);
    env->ReleaseDoubleArrayElements(img_out, j_img_out, 0);
}