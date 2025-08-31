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
 *  - cusparseCreateDnMat
 *  - cusparseCsrSetPointers
 *  - cusparseCsr2cscEx2
 *  - cusparseCsr2cscEx2_bufferSize
 *  - cusparseDcsrgeam2
 *  - cusparseDcsrgeam2_bufferSizeEx
 *  - cusparseSparseToDense
 *  - cusparseSparseToDense_bufferSize
 *  - cusparseDenseToSparse_bufferSize
 *  - cusparseDenseToSparse_analysis
 *  - cusparseDenseToSparse_convert
 *  - cusparseDnnz
 *  - cusparseSetMatType
 *  - cusparseSetMatIndexBase
 *  - cusparseSetPointerMode
 *  - cusparseXcsrgeam2Nnz
 *  - cusparseSpGEMM_workEstimation
 *  - cusparseSpGEMM_compute
 *  - cusparseSpMatGetSize
 *  - cusparseXcsrsort
 *  - cusparseXcsrsort_bufferSizeExt
 *  - cusparseCreate
 *  - cusparseCreateIdentityPermutation
 */


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1copyNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB,
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


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnMatNative
    (JNIEnv *env, jclass cls, jobject dnMatDescr, jlong rows, jlong cols, jlong ld, jobject values, jint valueType, jint order);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsrSetPointersNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jobject csrRowOffsets, jobject csrColInd, jobject csrValues);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr,
     jobject csrColInd, jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jobject buffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr, jobject csrColInd,
     jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jlongArray bufferSize);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB,
     jobject csrSortedRowPtrB, jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jobject pBuffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2_1bufferSizeExtNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB, jobject csrSortedRowPtrB,
     jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jlongArray pBufferSizeInBytes);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSparseToDenseNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSparseToDense_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jlongArray bufferSize);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jlongArray bufferSize);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1analysisNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1convertNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDnnzNative
    (JNIEnv *env, jclass cls, jobject handle, jint dirA, jint m, jint n, jobject descrA, jobject A, jint lda, jobject nnzPerRowCol, jobject nnzTotalDevHostPtr);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetMatTypeNative
    (JNIEnv *env, jclass cls, jobject descrA, jint type);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetMatIndexBaseNative
    (JNIEnv *env, jclass cls, jobject descrA, jint base);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetPointerModeNative
    (JNIEnv *env, jclass cls, jobject handle, jint mode);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrgeam2NnzNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject descrA, jint nnzA, jobject csrSortedRowPtrA, jobject csrSortedColIndA,
     jobject descrB, jint nnzB, jobject csrSortedRowPtrB, jobject csrSortedColIndB, jobject descrC, jobject csrSortedRowPtrC, jobject nnzTotalDevHostPtr, jobject workspace);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1workEstimationNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jobject spgemmDescr, jlongArray bufferSize1, jobject externalBuffer1);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1computeNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jobject spgemmDescr, jlongArray bufferSize2, jobject externalBuffer2);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMatGetSizeNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jlongArray rows, jlongArray cols, jlongArray nnz);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrsortNative
(JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject descrA, jobject csrRowPtrA, jobject csrColIndA, jobject P, jobject pBuffer);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrsort_1bufferSizeExtNative
(JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrRowPtrA, jobject csrColIndA, jlongArray pBufferSizeInBytes);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateNative(JNIEnv *env, jclass cls, jobject handle);


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateIdentityPermutationNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject p);

#ifdef __cplusplus
}
#endif
#endif
