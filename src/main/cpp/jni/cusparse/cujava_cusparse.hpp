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

#ifndef _Included_org_apache_sysds_cujava_cusparse_CuJavaCusparse
#define _Included_org_apache_sysds_cujava_cusparse_CuJavaCusparse
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:  org.apache.sysds.cujava.cusparse.CuJavaCusparse
 * Methods:
 *  - cusparseSpGEMM_copyNative
 *  - cusparseGetMatIndexBase
 *  - cusparseCreateCsr
 *  - cusparseCreateDnVec
 *  - cusparseSpMV_bufferSize
 *  - cusparseSpMV
 *  - cusparseDestroy
 *  - cusparseDestroyDnVec
 *  - cusparseDestroyDnMat
 *  - cusparseDestroySpMat
 *  - cusparseSpMM
 *  - cusparseSpMM_bufferSize
 */

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1copyNative
    (JNIEnv *env, jclass, jobject handle, jint opA, jint opB,
     jobject alpha, jobject matA, jobject matB, jobject beta, jobject matC,
     jint computeType, jint alg, jobject spgemmDescr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseGetMatIndexBaseNative
  (JNIEnv *env, jclass cls, jobject descrA);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateCsrNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jlong rows, jlong cols, jlong nnz, jobject csrRowOffsets,
     jobject csrColInd, jobject csrValues, jint csrRowOffsetsType, jint csrColIndType, jint idxBase, jint valueType);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnVecNative
    (JNIEnv *env, jclass cls, jobject dnVecDescr, jlong size, jobject values, jint valueType);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMV_1bufferSizeNative
  (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta,
   jobject vecY, jint computeType, jint alg, jlongArray bufferSize);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMVNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta,
     jobject vecY, jint computeType, jint alg, jobject externalBuffer);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyNative
    (JNIEnv *env, jclass cls, jobject handle);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnVecNative
    (JNIEnv *env, jclass cls, jobject dnVecDescr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnMatNative
    (JNIEnv *env, jclass cls, jobject dnMatDescr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroySpMatNative
    (JNIEnv *env, jclass cls, jobject spMatDescr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMMNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jobject externalBuffer);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMM_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jlongArray bufferSize);

#ifdef __cplusplus
}
#endif
#endif
