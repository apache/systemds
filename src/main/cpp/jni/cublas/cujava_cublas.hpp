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

#include <jni.h>

#ifndef _Included_org_apache_sysds_cujava_cublas_CuJavaCublas
#define _Included_org_apache_sysds_cujava_cublas_CuJavaCublas
#ifdef __cplusplus
extern "C" {
#endif


/*
 * Class:  org.apache.sysds.cujava.cublas.CuJavaCublas
 * Methods:
 *  - cublasCreate
 *  - cublasDestroy
 *  - cublasDgeam
 *  - cublasDdot
 *  - cublasDgemv
 *  - cublasDgemm
 *  - cublasDsyrk
 *  - cublasDaxpy
 *  - cublasDtrsm
 */



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasCreateNative(JNIEnv *env, jclass cls, jobject handle);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDestroyNative(JNIEnv *env, jclass cls, jobject handle);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgeamNative
    (JNIEnv *env, jclass cls, jobject handle, jint transa, jint transb, jint m, jint n, jobject alpha, jobject A,
     jint lda, jobject beta, jobject B, jint ldb, jobject C, jint ldc);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDdotNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject x, jint incx, jobject y, jint incy, jobject result);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgemvNative
    (JNIEnv *env, jclass cls, jobject handle, jint trans, jint m, jint n, jobject alpha, jobject A, jint lda,
     jobject x, jint incx, jobject beta, jobject y, jint incy);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgemmNative
    (JNIEnv *env, jclass cls, jobject handle, jint transa, jint transb, jint m, jint n, jint k, jobject alpha,
     jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDsyrkNative
    (JNIEnv *env, jclass cls, jobject handle, jint uplo, jint trans, jint n, jint k, jobject alpha,
     jobject A, jint lda, jobject beta, jobject C, jint ldc);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDaxpyNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDtrsmNative
    (JNIEnv *env, jclass cls, jobject handle, jint side, jint uplo, jint trans, jint diag, jint m,
     jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb);


#ifdef __cplusplus
}
#endif
#endif
