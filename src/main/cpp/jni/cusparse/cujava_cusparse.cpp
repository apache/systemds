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


#include "cujava_cusparse.hpp"
#include "cujava_cusparse_common.hpp"

#define CUJAVA_REQUIRE_NONNULL(env, obj, name, method)                           \
    do {                                                                          \
        if ((obj) == nullptr) {                                                   \
            ThrowByName((env), "java/lang/NullPointerException",                  \
                        "Parameter '" name "' is null for " method);              \
            return CUJAVA_CUSPARSE_INTERNAL_ERROR;                                         \
        }                                                                         \
    } while (0)


JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv *env = nullptr;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4)) {
        return JNI_ERR;
    }

    // Only what we need so far
    if (initJNIUtils(env) == JNI_ERR)      return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR)  return JNI_ERR;

    return JNI_VERSION_1_4;
}



JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1copyNative
  (JNIEnv *env, jclass, jobject handle, jint opA, jint opB,jobject alpha, jobject matA, jobject matB, jobject beta, jobject matC,
   jint computeType, jint alg, jobject spgemmDescr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, matC, "matC", "cusparseSpGEMM_copy");
    CUJAVA_REQUIRE_NONNULL(env, spgemmDescr, "spgemmDescr", "cusparseSpGEMM_copy");

    Logger::log(LOG_TRACE, "Executing cusparseSpGEMM_copy\n");

    // Copy Java inputs into native locals
    cusparseHandle_t h = (cusparseHandle_t)getNativePointerValue(env, handle);
    cusparseOperation_t aOp = (cusparseOperation_t)opA;
    cusparseOperation_t bOp = (cusparseOperation_t)opB;
    PointerData* alphaPD = initPointerData(env, alpha); if (!alphaPD) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    void* alphaPtr = alphaPD->getPointer(env);
    cusparseConstSpMatDescr_t A = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    cusparseConstSpMatDescr_t B = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matB);
    PointerData* betaPD = initPointerData(env, beta);  if (!betaPD)  { releasePointerData(env, alphaPD, JNI_ABORT); return CUJAVA_CUSPARSE_INTERNAL_ERROR; }
    void* betaPtr = betaPD->getPointer(env);
    cusparseSpMatDescr_t C = (cusparseSpMatDescr_t)getNativePointerValue(env, matC);
    cudaDataType ct = (cudaDataType)computeType;
    cusparseSpGEMMAlg_t al = (cusparseSpGEMMAlg_t)alg;
    cusparseSpGEMMDescr_t D  = (cusparseSpGEMMDescr_t)getNativePointerValue(env, spgemmDescr);

    // Cusparse API call
    cusparseStatus_t st = cusparseSpGEMM_copy(h, aOp, bOp, alphaPtr, A, B, betaPtr, C, ct, al, D);

    // alpha/beta are inputs → no commit
    if (!releasePointerData(env, alphaPD, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, betaPD,  JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    return (jint)st;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseGetMatIndexBaseNative(JNIEnv *env, jclass cls, jobject descrA) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseGetMatIndexBase");

    Logger::log(LOG_TRACE, "Executing cusparseGetMatIndexBase(descrA=%p)\n", descrA);

    // Declare native variables
    cusparseMatDescr_t descrA_native;

    // Copy Java inputs into native locals
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);

    // Cusparse API call
    cusparseIndexBase_t jniResult_native = cusparseGetMatIndexBase(descrA_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateCsrNative
  (JNIEnv *env, jclass cls, jobject spMatDescr, jlong rows, jlong cols, jlong nnz, jobject csrRowOffsets,
   jobject csrColInd, jobject csrValues, jint csrRowOffsetsType, jint csrColIndType, jint idxBase, jint valueType) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, spMatDescr, "spMatDescr", "cusparseCreateCsr");

    // Log message
    Logger::log(LOG_TRACE, "Executing cusparseCreateCsr(spMatDescr=%p, rows=%ld, cols=%ld, nnz=%ld, csrRowOffsets=%p, csrColInd=%p, csrValues=%p, csrRowOffsetsType=%d, csrColIndType=%d, idxBase=%d, valueType=%d)\n",
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);

    // Declare native variables
    cusparseSpMatDescr_t spMatDescr_native;
    int64_t rows_native = 0;
    int64_t cols_native = 0;
    int64_t nnz_native = 0;
    void * csrRowOffsets_native = nullptr;
    void * csrColInd_native = nullptr;
    void * csrValues_native = nullptr;
    cusparseIndexType_t csrRowOffsetsType_native;
    cusparseIndexType_t csrColIndType_native;
    cusparseIndexBase_t idxBase_native;
    cudaDataType valueType_native;

    // Copy Java inputs into native locals
    rows_native = (int64_t)rows;
    cols_native = (int64_t)cols;
    nnz_native = (int64_t)nnz;
    csrRowOffsets_native = (void *)getPointer(env, csrRowOffsets);
    csrColInd_native = (void *)getPointer(env, csrColInd);
    csrValues_native = (void *)getPointer(env, csrValues);
    csrRowOffsetsType_native = (cusparseIndexType_t)csrRowOffsetsType;
    csrColIndType_native = (cusparseIndexType_t)csrColIndType;
    idxBase_native = (cusparseIndexBase_t)idxBase;
    valueType_native = (cudaDataType)valueType;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCreateCsr(&spMatDescr_native, rows_native, cols_native, nnz_native, csrRowOffsets_native,
        csrColInd_native, csrValues_native, csrRowOffsetsType_native, csrColIndType_native, idxBase_native, valueType_native);
    setNativePointerValue(env, spMatDescr, (jlong)spMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnVecNative
    (JNIEnv *env, jclass cls, jobject dnVecDescr, jlong size, jobject values, jint valueType) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, dnVecDescr, "dnVecDescr", "cusparseCreateDnVec");
    CUJAVA_REQUIRE_NONNULL(env, values, "values", "cusparseCreateDnVec");

    Logger::log(LOG_TRACE, "Executing cusparseCreateDnVec(dnVecDescr=%p, size=%ld, values=%p, valueType=%d)\n",
        dnVecDescr, size, values, valueType);

    // Declare native variables
    cusparseDnVecDescr_t dnVecDescr_native;
    int64_t size_native = 0;
    void * values_native = nullptr;
    cudaDataType valueType_native;

    // Copy Java inputs into native locals
    size_native = (int64_t)size;
    values_native = (void *)getPointer(env, values);
    valueType_native = (cudaDataType)valueType;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCreateDnVec(&dnVecDescr_native, size_native, values_native, valueType_native);
    setNativePointerValue(env, dnVecDescr, (jlong)dnVecDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMV_1bufferSizeNative
  (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta, jobject vecY, jint computeType, jint alg, jlongArray bufferSize) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, vecX, "vecX", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, vecY, "vecY", "cusparseSpMV_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize, "bufferSize", "cusparseSpMV_bufferSize");

    Logger::log(LOG_TRACE, "Executing cusparseSpMV_bufferSize(handle=%p, opA=%d, alpha=%p, matA=%p, vecX=%p, beta=%p, vecY=%p, computeType=%d, alg=%d, bufferSize=%p)\n",
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnVecDescr_t vecX_native;
    void * beta_native = nullptr;
    cusparseDnVecDescr_t vecY_native;
    cudaDataType computeType_native;
    cusparseSpMVAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    vecX_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, vecX);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    beta_native = (void *)beta_pointerData->getPointer(env);
    vecY_native = (cusparseDnVecDescr_t)getNativePointerValue(env, vecY);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMVAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSpMV_bufferSize(handle_native, opA_native, alpha_native, matA_native,
        vecX_native, beta_native, vecY_native, computeType_native, alg_native, bufferSize_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMVNative
 (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta, jobject vecY, jint computeType, jint alg, jobject externalBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpMV");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpMV");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpMV");
    CUJAVA_REQUIRE_NONNULL(env, vecX, "vecX", "cusparseSpMV");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpMV");
    CUJAVA_REQUIRE_NONNULL(env, vecY, "vecY", "cusparseSpMV");

    Logger::log(LOG_TRACE, "Executing cusparseSpMV(handle=%p, opA=%d, alpha=%p, matA=%p, vecX=%p, beta=%p, vecY=%p, computeType=%d, alg=%d, externalBuffer=%p)\n",
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnVecDescr_t vecX_native;
    void * beta_native = nullptr;
    cusparseDnVecDescr_t vecY_native;
    cudaDataType computeType_native;
    cusparseSpMVAlg_t alg_native;
    void * externalBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    vecX_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, vecX);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    vecY_native = (cusparseDnVecDescr_t)getNativePointerValue(env, vecY);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMVAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSpMV(handle_native, opA_native, alpha_native, matA_native, vecX_native, beta_native, vecY_native, computeType_native, alg_native, externalBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyNative(JNIEnv *env, jclass cls, jobject handle) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDestroy");

    Logger::log(LOG_TRACE, "Executing cusparseDestroy(handle=%p)\n", handle);

    // Declare native variables
    cusparseHandle_t handle_native;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDestroy(handle_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnVecNative(JNIEnv *env, jclass cls, jobject dnVecDescr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, dnVecDescr, "dnVecDescr", "cusparseDestroyDnVec");

    Logger::log(LOG_TRACE, "Executing cusparseDestroyDnVec(dnVecDescr=%p)\n", dnVecDescr);

    // Declare native variables
    cusparseConstDnVecDescr_t dnVecDescr_native;

    // Copy Java inputs into native locals
    dnVecDescr_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, dnVecDescr);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDestroyDnVec(dnVecDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnMatNative(JNIEnv *env, jclass cls, jobject dnMatDescr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, dnMatDescr, "dnMatDescr", "cusparseDestroyDnMat");

    Logger::log(LOG_TRACE, "Executing cusparseDestroyDnMat(dnMatDescr=%p)\n", dnMatDescr);

    // Declare native variables
    cusparseConstDnMatDescr_t dnMatDescr_native;

    // Copy Java inputs into native locals
    dnMatDescr_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, dnMatDescr);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDestroyDnMat(dnMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroySpMatNative(JNIEnv *env, jclass cls, jobject spMatDescr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, spMatDescr, "spMatDescr", "cusparseDestroySpMat");

    Logger::log(LOG_TRACE, "Executing cusparseDestroySpMat(spMatDescr=%p)\n", spMatDescr);

    // Declare native variables
    cusparseConstSpMatDescr_t spMatDescr_native;

    // Copy Java inputs into native locals
    spMatDescr_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, spMatDescr);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDestroySpMat(spMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMMNative
(JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta, jobject matC, jint computeType, jint alg, jobject externalBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpMM");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpMM");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpMM");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSpMM");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpMM");
    CUJAVA_REQUIRE_NONNULL(env, matC, "matC", "cusparseSpMM");

    Logger::log(LOG_TRACE, "Executing cusparseSpMM(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, externalBuffer=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    cusparseOperation_t opB_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnMatDescr_t matB_native;
    void * beta_native = nullptr;
    cusparseDnMatDescr_t matC_native;
    cudaDataType computeType_native;
    cusparseSpMMAlg_t alg_native;
    void * externalBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMMAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSpMM(handle_native, opA_native, opB_native, alpha_native, matA_native,
        matB_native, beta_native, matC_native, computeType_native, alg_native, externalBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMM_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jlongArray bufferSize) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matC, "matC", "cusparseSpMM_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize, "bufferSize", "cusparseSpMM_bufferSize");

    Logger::log(LOG_TRACE, "Executing cusparseSpMM_bufferSize(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, bufferSize=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    cusparseOperation_t opB_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnMatDescr_t matB_native;
    void * beta_native = nullptr;
    cusparseDnMatDescr_t matC_native;
    cudaDataType computeType_native;
    cusparseSpMMAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr)
    {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMMAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSpMM_bufferSize(handle_native, opA_native, opB_native, alpha_native,
        matA_native, matB_native, beta_native, matC_native, computeType_native, alg_native, bufferSize_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnMatNative
    (JNIEnv *env, jclass cls, jobject dnMatDescr, jlong rows, jlong cols, jlong ld, jobject values, jint valueType, jint order) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, dnMatDescr, "dnMatDescr", "cusparseCreateDnMat");
    CUJAVA_REQUIRE_NONNULL(env, values, "values", "cusparseCreateDnMat");

    Logger::log(LOG_TRACE, "Executing cusparseCreateDnMat(dnMatDescr=%p, rows=%ld, cols=%ld, ld=%ld, values=%p, valueType=%d, order=%d)\n",
        dnMatDescr, rows, cols, ld, values, valueType, order);

    // Declare native variables
    cusparseDnMatDescr_t dnMatDescr_native;
    int64_t rows_native = 0;
    int64_t cols_native = 0;
    int64_t ld_native = 0;
    void * values_native = nullptr;
    cudaDataType valueType_native;
    cusparseOrder_t order_native;

    // Copy Java inputs into native locals
    rows_native = (int64_t)rows;
    cols_native = (int64_t)cols;
    ld_native = (int64_t)ld;
    values_native = (void *)getPointer(env, values);
    valueType_native = (cudaDataType)valueType;
    order_native = (cusparseOrder_t)order;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCreateDnMat(&dnMatDescr_native, rows_native, cols_native, ld_native,
        values_native, valueType_native, order_native);
    setNativePointerValue(env, dnMatDescr, (jlong)dnMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsrSetPointersNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jobject csrRowOffsets, jobject csrColInd, jobject csrValues) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, spMatDescr, "spMatDescr", "cusparseCsrSetPointers");
    CUJAVA_REQUIRE_NONNULL(env, csrRowOffsets, "csrRowOffsets", "cusparseCsrSetPointers");
    CUJAVA_REQUIRE_NONNULL(env, csrColInd, "csrColInd", "cusparseCsrSetPointers");
    CUJAVA_REQUIRE_NONNULL(env, csrValues, "csrValues", "cusparseCsrSetPointers");

    Logger::log(LOG_TRACE, "Executing cusparseCsrSetPointers(spMatDescr=%p, csrRowOffsets=%p, csrColInd=%p, csrValues=%p)\n",
        spMatDescr, csrRowOffsets, csrColInd, csrValues);

    // Declare native variables
    cusparseSpMatDescr_t spMatDescr_native;
    void * csrRowOffsets_native = nullptr;
    void * csrColInd_native = nullptr;
    void * csrValues_native = nullptr;

    // Copy Java inputs into native locals
    spMatDescr_native = (cusparseSpMatDescr_t)getNativePointerValue(env, spMatDescr);
    csrRowOffsets_native = (void *)getPointer(env, csrRowOffsets);
    csrColInd_native = (void *)getPointer(env, csrColInd);
    csrValues_native = (void *)getPointer(env, csrValues);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCsrSetPointers(spMatDescr_native, csrRowOffsets_native, csrColInd_native, csrValues_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr,
     jobject csrColInd, jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jobject buffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, csrVal, "csrVal", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, csrRowPtr, "csrRowPtr", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, csrColInd, "csrColInd", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, cscVal, "cscVal", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, cscColPtr, "cscColPtr", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, cscRowInd, "cscRowInd", "cusparseCsr2cscEx2");
    CUJAVA_REQUIRE_NONNULL(env, buffer, "buffer", "cusparseCsr2cscEx2");

    Logger::log(LOG_TRACE, "Executing cusparseCsr2cscEx2(handle=%p, m=%d, n=%d, nnz=%d, csrVal=%p, csrRowPtr=%p, csrColInd=%p, cscVal=%p, cscColPtr=%p, cscRowInd=%p, valType=%d, copyValues=%d, idxBase=%d, alg=%d, buffer=%p)\n",
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    int nnz_native = 0;
    void * csrVal_native = nullptr;
    int * csrRowPtr_native = nullptr;
    int * csrColInd_native = nullptr;
    void * cscVal_native = nullptr;
    int * cscColPtr_native = nullptr;
    int * cscRowInd_native = nullptr;
    cudaDataType valType_native;
    cusparseAction_t copyValues_native;
    cusparseIndexBase_t idxBase_native;
    cusparseCsr2CscAlg_t alg_native;
    void * buffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    nnz_native = (int)nnz;
    csrVal_native = (void *)getPointer(env, csrVal);
    csrRowPtr_native = (int *)getPointer(env, csrRowPtr);
    csrColInd_native = (int *)getPointer(env, csrColInd);
    cscVal_native = (void *)getPointer(env, cscVal);
    cscColPtr_native = (int *)getPointer(env, cscColPtr);
    cscRowInd_native = (int *)getPointer(env, cscRowInd);
    valType_native = (cudaDataType)valType;
    copyValues_native = (cusparseAction_t)copyValues;
    idxBase_native = (cusparseIndexBase_t)idxBase;
    alg_native = (cusparseCsr2CscAlg_t)alg;
    buffer_native = (void *)getPointer(env, buffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCsr2cscEx2(handle_native, m_native, n_native, nnz_native,
        csrVal_native, csrRowPtr_native, csrColInd_native, cscVal_native, cscColPtr_native, cscRowInd_native,
        valType_native, copyValues_native, idxBase_native, alg_native, buffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr, jobject csrColInd,
     jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jlongArray bufferSize) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, csrVal, "csrVal", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, csrRowPtr, "csrRowPtr", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, csrColInd, "csrColInd", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, cscVal, "cscVal", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, cscColPtr, "cscColPtr", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, cscRowInd, "cscRowInd", "cusparseCsr2cscEx2_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize, "bufferSize", "cusparseCsr2cscEx2_bufferSize");

    Logger::log(LOG_TRACE, "Executing cusparseCsr2cscEx2_bufferSize(handle=%p, m=%d, n=%d, nnz=%d, csrVal=%p, csrRowPtr=%p, csrColInd=%p, cscVal=%p, cscColPtr=%p, cscRowInd=%p, valType=%d, copyValues=%d, idxBase=%d, alg=%d, bufferSize=%p)\n",
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    int nnz_native = 0;
    void * csrVal_native = nullptr;
    int * csrRowPtr_native = nullptr;
    int * csrColInd_native = nullptr;
    void * cscVal_native = nullptr;
    int * cscColPtr_native = nullptr;
    int * cscRowInd_native = nullptr;
    cudaDataType valType_native;
    cusparseAction_t copyValues_native;
    cusparseIndexBase_t idxBase_native;
    cusparseCsr2CscAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    nnz_native = (int)nnz;
    csrVal_native = (void *)getPointer(env, csrVal);
    csrRowPtr_native = (int *)getPointer(env, csrRowPtr);
    csrColInd_native = (int *)getPointer(env, csrColInd);
    cscVal_native = (void *)getPointer(env, cscVal);
    cscColPtr_native = (int *)getPointer(env, cscColPtr);
    cscRowInd_native = (int *)getPointer(env, cscRowInd);
    valType_native = (cudaDataType)valType;
    copyValues_native = (cusparseAction_t)copyValues;
    idxBase_native = (cusparseIndexBase_t)idxBase;
    alg_native = (cusparseCsr2CscAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCsr2cscEx2_bufferSize
        (handle_native, m_native, n_native, nnz_native, csrVal_native, csrRowPtr_native, csrColInd_native, cscVal_native,
         cscColPtr_native, cscRowInd_native, valType_native, copyValues_native, idxBase_native, alg_native, bufferSize_native);
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB,
     jobject csrSortedRowPtrB, jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jobject pBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValA, "csrSortedValA", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrA, "csrSortedRowPtrA", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndA, "csrSortedColIndA", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, descrB, "descrB", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValB, "csrSortedValB", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrB, "csrSortedRowPtrB", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndB, "csrSortedColIndB", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, descrC, "descrC", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValC, "csrSortedValC", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrC, "csrSortedRowPtrC", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndC, "csrSortedColIndC", "cusparseDcsrgeam2");
    CUJAVA_REQUIRE_NONNULL(env, pBuffer, "pBuffer", "cusparseDcsrgeam2");

    Logger::log(LOG_TRACE, "Executing cusparseDcsrgeam2(handle=%p, m=%d, n=%d, alpha=%p, descrA=%p, nnzA=%d, csrSortedValA=%p, csrSortedRowPtrA=%p, csrSortedColIndA=%p, beta=%p, descrB=%p, nnzB=%d, csrSortedValB=%p, csrSortedRowPtrB=%p, csrSortedColIndB=%p, descrC=%p, csrSortedValC=%p, csrSortedRowPtrC=%p, csrSortedColIndC=%p, pBuffer=%p)\n",
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    cusparseMatDescr_t descrA_native;
    int nnzA_native = 0;
    double * csrSortedValA_native = nullptr;
    int * csrSortedRowPtrA_native = nullptr;
    int * csrSortedColIndA_native = nullptr;
    double * beta_native = nullptr;
    cusparseMatDescr_t descrB_native;
    int nnzB_native = 0;
    double * csrSortedValB_native = nullptr;
    int * csrSortedRowPtrB_native = nullptr;
    int * csrSortedColIndB_native = nullptr;
    cusparseMatDescr_t descrC_native;
    double * csrSortedValC_native = nullptr;
    int * csrSortedRowPtrC_native = nullptr;
    int * csrSortedColIndC_native = nullptr;
    void * pBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    nnzA_native = (int)nnzA;
    csrSortedValA_native = (double *)getPointer(env, csrSortedValA);
    csrSortedRowPtrA_native = (int *)getPointer(env, csrSortedRowPtrA);
    csrSortedColIndA_native = (int *)getPointer(env, csrSortedColIndA);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    descrB_native = (cusparseMatDescr_t)getNativePointerValue(env, descrB);
    nnzB_native = (int)nnzB;
    csrSortedValB_native = (double *)getPointer(env, csrSortedValB);
    csrSortedRowPtrB_native = (int *)getPointer(env, csrSortedRowPtrB);
    csrSortedColIndB_native = (int *)getPointer(env, csrSortedColIndB);
    descrC_native = (cusparseMatDescr_t)getNativePointerValue(env, descrC);
    csrSortedValC_native = (double *)getPointer(env, csrSortedValC);
    csrSortedRowPtrC_native = (int *)getPointer(env, csrSortedRowPtrC);
    csrSortedColIndC_native = (int *)getPointer(env, csrSortedColIndC);
    pBuffer_native = (void *)getPointer(env, pBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDcsrgeam2(handle_native, m_native, n_native, alpha_native, descrA_native,
        nnzA_native, csrSortedValA_native, csrSortedRowPtrA_native, csrSortedColIndA_native, beta_native, descrB_native,
         nnzB_native, csrSortedValB_native, csrSortedRowPtrB_native, csrSortedColIndB_native, descrC_native, csrSortedValC_native,
         csrSortedRowPtrC_native, csrSortedColIndC_native, pBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2_1bufferSizeExtNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB, jobject csrSortedRowPtrB,
     jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jlongArray pBufferSizeInBytes) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValA, "csrSortedValA", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrA, "csrSortedRowPtrA", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndA, "csrSortedColIndA", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, descrB, "descrB", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValB, "csrSortedValB", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrB, "csrSortedRowPtrB", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndB, "csrSortedColIndB", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, descrC, "descrC", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedValC, "csrSortedValC", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrC, "csrSortedRowPtrC", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndC, "csrSortedColIndC", "cusparseDcsrgeam2_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, pBufferSizeInBytes, "pBufferSizeInBytes", "cusparseDcsrgeam2_bufferSizeExt");

    Logger::log(LOG_TRACE, "Executing cusparseDcsrgeam2_bufferSizeExt(handle=%p, m=%d, n=%d, alpha=%p, descrA=%p, nnzA=%d, csrSortedValA=%p, csrSortedRowPtrA=%p, csrSortedColIndA=%p, beta=%p, descrB=%p, nnzB=%d, csrSortedValB=%p, csrSortedRowPtrB=%p, csrSortedColIndB=%p, descrC=%p, csrSortedValC=%p, csrSortedRowPtrC=%p, csrSortedColIndC=%p, pBufferSizeInBytes=%p)\n",
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    cusparseMatDescr_t descrA_native;
    int nnzA_native = 0;
    double * csrSortedValA_native = nullptr;
    int * csrSortedRowPtrA_native = nullptr;
    int * csrSortedColIndA_native = nullptr;
    double * beta_native = nullptr;
    cusparseMatDescr_t descrB_native;
    int nnzB_native = 0;
    double * csrSortedValB_native = nullptr;
    int * csrSortedRowPtrB_native = nullptr;
    int * csrSortedColIndB_native = nullptr;
    cusparseMatDescr_t descrC_native;
    double * csrSortedValC_native = nullptr;
    int * csrSortedRowPtrC_native = nullptr;
    int * csrSortedColIndC_native = nullptr;
    size_t * pBufferSizeInBytes_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    nnzA_native = (int)nnzA;
    csrSortedValA_native = (double *)getPointer(env, csrSortedValA);
    csrSortedRowPtrA_native = (int *)getPointer(env, csrSortedRowPtrA);
    csrSortedColIndA_native = (int *)getPointer(env, csrSortedColIndA);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    descrB_native = (cusparseMatDescr_t)getNativePointerValue(env, descrB);
    nnzB_native = (int)nnzB;
    csrSortedValB_native = (double *)getPointer(env, csrSortedValB);
    csrSortedRowPtrB_native = (int *)getPointer(env, csrSortedRowPtrB);
    csrSortedColIndB_native = (int *)getPointer(env, csrSortedColIndB);
    descrC_native = (cusparseMatDescr_t)getNativePointerValue(env, descrC);
    csrSortedValC_native = (double *)getPointer(env, csrSortedValC);
    csrSortedRowPtrC_native = (int *)getPointer(env, csrSortedRowPtrC);
    csrSortedColIndC_native = (int *)getPointer(env, csrSortedColIndC);
    if (!initNative(env, pBufferSizeInBytes, pBufferSizeInBytes_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDcsrgeam2_bufferSizeExt(handle_native, m_native, n_native, alpha_native,
        descrA_native, nnzA_native, csrSortedValA_native, csrSortedRowPtrA_native, csrSortedColIndA_native, beta_native,
        descrB_native, nnzB_native, csrSortedValB_native, csrSortedRowPtrB_native, csrSortedColIndB_native, descrC_native,
        csrSortedValC_native, csrSortedRowPtrC_native, csrSortedColIndC_native, pBufferSizeInBytes_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releaseNative(env, pBufferSizeInBytes_native, pBufferSizeInBytes, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSparseToDenseNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSparseToDense");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSparseToDense");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSparseToDense");

    Logger::log(LOG_TRACE, "Executing cusparseSparseToDense(handle=%p, matA=%p, matB=%p, alg=%d, externalBuffer=%p)\n",
        handle, matA, matB, alg, externalBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseConstSpMatDescr_t matA_native;
    cusparseDnMatDescr_t matB_native;
    cusparseSparseToDenseAlg_t alg_native;
    void * externalBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matB);
    alg_native = (cusparseSparseToDenseAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSparseToDense(handle_native, matA_native, matB_native, alg_native, externalBuffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSparseToDense_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jlongArray bufferSize) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSparseToDense_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSparseToDense_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSparseToDense_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize, "bufferSize", "cusparseSparseToDense_bufferSize");

    Logger::log(LOG_TRACE, "Executing cusparseSparseToDense_bufferSize(handle=%p, matA=%p, matB=%p, alg=%d, bufferSize=%p)\n",
        handle, matA, matB, alg, bufferSize);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseConstSpMatDescr_t matA_native;
    cusparseDnMatDescr_t matB_native;
    cusparseSparseToDenseAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matB);
    alg_native = (cusparseSparseToDenseAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSparseToDense_bufferSize(handle_native, matA_native, matB_native, alg_native, bufferSize_native);

    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jlongArray bufferSize) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDenseToSparse_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseDenseToSparse_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseDenseToSparse_bufferSize");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize, "bufferSize", "cusparseDenseToSparse_bufferSize");

    Logger::log(LOG_TRACE, "Executing cusparseDenseToSparse_bufferSize(handle=%p, matA=%p, matB=%p, alg=%d, bufferSize=%p)\n", handle, matA, matB, alg, bufferSize);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseConstDnMatDescr_t matA_native;
    cusparseSpMatDescr_t matB_native;
    cusparseDenseToSparseAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    matA_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseSpMatDescr_t)getNativePointerValue(env, matB);
    alg_native = (cusparseDenseToSparseAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDenseToSparse_bufferSize(handle_native, matA_native, matB_native, alg_native, bufferSize_native);

    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1analysisNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDenseToSparse_analysis");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseDenseToSparse_analysis");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseDenseToSparse_analysis");

    Logger::log(LOG_TRACE, "Executing cusparseDenseToSparse_analysis(handle=%p, matA=%p, matB=%p, alg=%d, externalBuffer=%p)\n",
        handle, matA, matB, alg, externalBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseConstDnMatDescr_t matA_native;
    cusparseSpMatDescr_t matB_native;
    cusparseDenseToSparseAlg_t alg_native;
    void * externalBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    matA_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseSpMatDescr_t)getNativePointerValue(env, matB);
    alg_native = (cusparseDenseToSparseAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDenseToSparse_analysis(handle_native, matA_native, matB_native, alg_native, externalBuffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDenseToSparse_1convertNative
    (JNIEnv *env, jclass cls, jobject handle, jobject matA, jobject matB, jint alg, jobject externalBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDenseToSparse_convert");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseDenseToSparse_convert");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseDenseToSparse_convert");

    Logger::log(LOG_TRACE, "Executing cusparseDenseToSparse_convert(handle=%p, matA=%p, matB=%p, alg=%d, externalBuffer=%p)\n",
        handle, matA, matB, alg, externalBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseConstDnMatDescr_t matA_native;
    cusparseSpMatDescr_t matB_native;
    cusparseDenseToSparseAlg_t alg_native;
    void * externalBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    matA_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseSpMatDescr_t)getNativePointerValue(env, matB);
    alg_native = (cusparseDenseToSparseAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDenseToSparse_convert(handle_native, matA_native, matB_native, alg_native, externalBuffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDnnzNative
    (JNIEnv *env, jclass cls, jobject handle, jint dirA, jint m, jint n, jobject descrA, jobject A, jint lda, jobject nnzPerRowCol, jobject nnzTotalDevHostPtr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseDnnz");
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseDnnz");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cusparseDnnz");
    CUJAVA_REQUIRE_NONNULL(env, nnzPerRowCol, "nnzPerRowCol", "cusparseDnnz");
    CUJAVA_REQUIRE_NONNULL(env, nnzTotalDevHostPtr, "nnzTotalDevHostPtr", "cusparseDnnz");

    Logger::log(LOG_TRACE, "Executing cusparseDnnz(handle=%p, dirA=%d, m=%d, n=%d, descrA=%p, A=%p, lda=%d, nnzPerRowCol=%p, nnzTotalDevHostPtr=%p)\n",
        handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseDirection_t dirA_native;
    int m_native = 0;
    int n_native = 0;
    cusparseMatDescr_t descrA_native;
    double * A_native = nullptr;
    int lda_native = 0;
    int * nnzPerRowCol_native = nullptr;
    int * nnzTotalDevHostPtr_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    dirA_native = (cusparseDirection_t)dirA;
    m_native = (int)m;
    n_native = (int)n;
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    nnzPerRowCol_native = (int *)getPointer(env, nnzPerRowCol);
    PointerData *nnzTotalDevHostPtr_pointerData = initPointerData(env, nnzTotalDevHostPtr);
    if (nnzTotalDevHostPtr_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    nnzTotalDevHostPtr_native = (int *)nnzTotalDevHostPtr_pointerData->getPointer(env);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseDnnz(handle_native, dirA_native, m_native, n_native, descrA_native, A_native,
        lda_native, nnzPerRowCol_native, nnzTotalDevHostPtr_native);

    if (!isPointerBackedByNativeMemory(env, nnzTotalDevHostPtr)) {
        cudaDeviceSynchronize();
    }
    if (!releasePointerData(env, nnzTotalDevHostPtr_pointerData, 0)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetMatTypeNative
    (JNIEnv *env, jclass cls, jobject descrA, jint type) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseSetMatType");

    Logger::log(LOG_TRACE, "Executing cusparseSetMatType(descrA=%p, type=%d)\n", descrA, type);

    // Declare native variables
    cusparseMatDescr_t descrA_native;
    cusparseMatrixType_t type_native;

    // Copy Java inputs into native locals
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    type_native = (cusparseMatrixType_t)type;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSetMatType(descrA_native, type_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetMatIndexBaseNative
    (JNIEnv *env, jclass cls, jobject descrA, jint base) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseSetMatIndexBase");

    Logger::log(LOG_TRACE, "Executing cusparseSetMatIndexBase(descrA=%p, base=%d)\n", descrA, base);

    // Declare native variables
    cusparseMatDescr_t descrA_native;
    cusparseIndexBase_t base_native;

    // Copy Java inputs into native locals
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    base_native = (cusparseIndexBase_t)base;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSetMatIndexBase(descrA_native, base_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSetPointerModeNative
    (JNIEnv *env, jclass cls, jobject handle, jint mode) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSetPointerMode");

    Logger::log(LOG_TRACE, "Executing cusparseSetPointerMode(handle=%p, mode=%d)\n", handle, mode);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparsePointerMode_t mode_native;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    mode_native = (cusparsePointerMode_t)mode;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseSetPointerMode(handle_native, mode_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrgeam2NnzNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject descrA, jint nnzA, jobject csrSortedRowPtrA, jobject csrSortedColIndA,
     jobject descrB, jint nnzB, jobject csrSortedRowPtrB, jobject csrSortedColIndB, jobject descrC, jobject csrSortedRowPtrC, jobject nnzTotalDevHostPtr, jobject workspace) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrA, "csrSortedRowPtrA", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndA, "csrSortedColIndA", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, descrB, "descrB", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrB, "csrSortedRowPtrB", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedColIndB, "csrSortedColIndB", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, descrC, "descrC", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, csrSortedRowPtrC, "csrSortedRowPtrC", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, nnzTotalDevHostPtr, "nnzTotalDevHostPtr", "cusparseXcsrgeam2Nnz");
    CUJAVA_REQUIRE_NONNULL(env, workspace, "workspace", "cusparseXcsrgeam2Nnz");

    // Log message
    Logger::log(LOG_TRACE, "Executing cusparseXcsrgeam2Nnz(handle=%p, m=%d, n=%d, descrA=%p, nnzA=%d, csrSortedRowPtrA=%p, csrSortedColIndA=%p, descrB=%p, nnzB=%d, csrSortedRowPtrB=%p, csrSortedColIndB=%p, descrC=%p, csrSortedRowPtrC=%p, nnzTotalDevHostPtr=%p, workspace=%p)\n",
        handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    cusparseMatDescr_t descrA_native;
    int nnzA_native = 0;
    int * csrSortedRowPtrA_native = nullptr;
    int * csrSortedColIndA_native = nullptr;
    cusparseMatDescr_t descrB_native;
    int nnzB_native = 0;
    int * csrSortedRowPtrB_native = nullptr;
    int * csrSortedColIndB_native = nullptr;
    cusparseMatDescr_t descrC_native;
    int * csrSortedRowPtrC_native = nullptr;
    int * nnzTotalDevHostPtr_native = nullptr;
    void * workspace_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    nnzA_native = (int)nnzA;
    csrSortedRowPtrA_native = (int *)getPointer(env, csrSortedRowPtrA);
    csrSortedColIndA_native = (int *)getPointer(env, csrSortedColIndA);
    descrB_native = (cusparseMatDescr_t)getNativePointerValue(env, descrB);
    nnzB_native = (int)nnzB;
    csrSortedRowPtrB_native = (int *)getPointer(env, csrSortedRowPtrB);
    csrSortedColIndB_native = (int *)getPointer(env, csrSortedColIndB);
    descrC_native = (cusparseMatDescr_t)getNativePointerValue(env, descrC);
    csrSortedRowPtrC_native = (int *)getPointer(env, csrSortedRowPtrC);
    PointerData *nnzTotalDevHostPtr_pointerData = initPointerData(env, nnzTotalDevHostPtr);

    if (nnzTotalDevHostPtr_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    nnzTotalDevHostPtr_native = (int *)nnzTotalDevHostPtr_pointerData->getPointer(env);
    workspace_native = (void *)getPointer(env, workspace);

    cusparseStatus_t jniResult_native = cusparseXcsrgeam2Nnz(handle_native, m_native, n_native, descrA_native, nnzA_native,
        csrSortedRowPtrA_native, csrSortedColIndA_native, descrB_native, nnzB_native, csrSortedRowPtrB_native, csrSortedColIndB_native,
        descrC_native, csrSortedRowPtrC_native, nnzTotalDevHostPtr_native, workspace_native);

    if (!isPointerBackedByNativeMemory(env, nnzTotalDevHostPtr)) {
        cudaDeviceSynchronize();
    }
    if (!releasePointerData(env, nnzTotalDevHostPtr_pointerData, 0)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1workEstimationNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jobject spgemmDescr, jlongArray bufferSize1, jobject externalBuffer1) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, matC, "matC", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, spgemmDescr, "spgemmDescr", "cusparseSpGEMM_workEstimation");
    CUJAVA_REQUIRE_NONNULL(env, bufferSize1, "bufferSize1", "cusparseSpGEMM_workEstimation");

    Logger::log(LOG_TRACE, "Executing cusparseSpGEMM_workEstimation(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, spgemmDescr=%p, bufferSize1=%p, externalBuffer1=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    cusparseOperation_t opB_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstSpMatDescr_t matB_native;
    void * beta_native = nullptr;
    cusparseSpMatDescr_t matC_native;
    cudaDataType computeType_native;
    cusparseSpGEMMAlg_t alg_native;
    cusparseSpGEMMDescr_t spgemmDescr_native;
    size_t * bufferSize1_native = nullptr;
    void * externalBuffer1_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseSpMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpGEMMAlg_t)alg;
    spgemmDescr_native = (cusparseSpGEMMDescr_t)getNativePointerValue(env, spgemmDescr);
    if (!initNative(env, bufferSize1, bufferSize1_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    externalBuffer1_native = (void *)getPointer(env, externalBuffer1);

    cusparseStatus_t jniResult_native = cusparseSpGEMM_workEstimation(handle_native, opA_native, opB_native, alpha_native,
        matA_native, matB_native, beta_native, matC_native, computeType_native, alg_native, spgemmDescr_native, bufferSize1_native, externalBuffer1_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize1_native, bufferSize1, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpGEMM_1computeNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jobject spgemmDescr, jlongArray bufferSize2, jobject externalBuffer2) {

     // Validate: all jobject parameters must be non-null
     CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, matA, "matA", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, matB, "matB", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, matC, "matC", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, spgemmDescr, "spgemmDescr", "cusparseSpGEMM_compute");
     CUJAVA_REQUIRE_NONNULL(env, bufferSize2, "bufferSize2", "cusparseSpGEMM_compute");

    Logger::log(LOG_TRACE, "Executing cusparseSpGEMM_compute(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, spgemmDescr=%p, bufferSize2=%p, externalBuffer2=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2);

    // Declare native variables
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    cusparseOperation_t opB_native;
    void * alpha_native = nullptr;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstSpMatDescr_t matB_native;
    void * beta_native = nullptr;
    cusparseSpMatDescr_t matC_native;
    cudaDataType computeType_native;
    cusparseSpGEMMAlg_t alg_native;
    cusparseSpGEMMDescr_t spgemmDescr_native;
    size_t * bufferSize2_native = nullptr;
    void * externalBuffer2_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseSpMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpGEMMAlg_t)alg;
    spgemmDescr_native = (cusparseSpGEMMDescr_t)getNativePointerValue(env, spgemmDescr);
    if (!initNative(env, bufferSize2, bufferSize2_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    externalBuffer2_native = (void *)getPointer(env, externalBuffer2);

    cusparseStatus_t jniResult_native = cusparseSpGEMM_compute(handle_native, opA_native, opB_native, alpha_native, matA_native,
        matB_native, beta_native, matC_native, computeType_native, alg_native, spgemmDescr_native, bufferSize2_native, externalBuffer2_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize2_native, bufferSize2, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMatGetSizeNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jlongArray rows, jlongArray cols, jlongArray nnz) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, spMatDescr, "spMatDescr", "cusparseSpMatGetSize");
    CUJAVA_REQUIRE_NONNULL(env, rows, "rows", "cusparseSpMatGetSize");
    CUJAVA_REQUIRE_NONNULL(env, cols, "cols", "cusparseSpMatGetSize");
    CUJAVA_REQUIRE_NONNULL(env, nnz, "nnz", "cusparseSpMatGetSize");

    Logger::log(LOG_TRACE, "Executing cusparseSpMatGetSize(spMatDescr=%p, rows=%p, cols=%p, nnz=%p)\n", spMatDescr, rows, cols, nnz);

    // Declare native variables
    cusparseConstSpMatDescr_t spMatDescr_native;
    int64_t rows_native;
    int64_t cols_native;
    int64_t nnz_native;

    // Copy Java inputs into native locals
    spMatDescr_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, spMatDescr);

    cusparseStatus_t jniResult_native = cusparseSpMatGetSize(spMatDescr_native, &rows_native, &cols_native, &nnz_native);

    if (!set(env, rows, 0, (jlong)rows_native)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!set(env, cols, 0, (jlong)cols_native)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;
    if (!set(env, nnz, 0, (jlong)nnz_native)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrsortNative
(JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject descrA, jobject csrRowPtrA, jobject csrColIndA, jobject P, jobject pBuffer) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseXcsrsort");
    CUJAVA_REQUIRE_NONNULL(env, descrA, "descrA", "cusparseXcsrsort");
    CUJAVA_REQUIRE_NONNULL(env, csrRowPtrA, "csrRowPtrA", "cusparseXcsrsort");
    CUJAVA_REQUIRE_NONNULL(env, csrColIndA, "csrColIndA", "cusparseXcsrsort");
    CUJAVA_REQUIRE_NONNULL(env, P, "P", "cusparseXcsrsort");
    CUJAVA_REQUIRE_NONNULL(env, pBuffer, "pBuffer", "cusparseXcsrsort");

    Logger::log(LOG_TRACE, "Executing cusparseXcsrsort(handle=%p, m=%d, n=%d, nnz=%d, descrA=%p, csrRowPtrA=%p, csrColIndA=%p, P=%p, pBuffer=%p)\n",
        handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    int nnz_native = 0;
    cusparseMatDescr_t descrA_native;
    int * csrRowPtrA_native = nullptr;
    int * csrColIndA_native = nullptr;
    int * P_native = nullptr;
    void * pBuffer_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    nnz_native = (int)nnz;
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    csrRowPtrA_native = (int *)getPointer(env, csrRowPtrA);
    csrColIndA_native = (int *)getPointer(env, csrColIndA);
    P_native = (int *)getPointer(env, P);
    pBuffer_native = (void *)getPointer(env, pBuffer);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseXcsrsort(handle_native, m_native, n_native, nnz_native, descrA_native,
        csrRowPtrA_native, csrColIndA_native, P_native, pBuffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseXcsrsort_1bufferSizeExtNative
(JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrRowPtrA, jobject csrColIndA, jlongArray pBufferSizeInBytes) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseXcsrsort_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrRowPtrA, "csrRowPtrA", "cusparseXcsrsort_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, csrColIndA, "csrColIndA", "cusparseXcsrsort_bufferSizeExt");
    CUJAVA_REQUIRE_NONNULL(env, pBufferSizeInBytes, "pBufferSizeInBytes", "cusparseXcsrsort_bufferSizeExt");

    Logger::log(LOG_TRACE, "Executing cusparseXcsrsort_bufferSizeExt(handle=%p, m=%d, n=%d, nnz=%d, csrRowPtrA=%p, csrColIndA=%p, pBufferSizeInBytes=%p)\n",
        handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes);

    // Declare native variables
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    int nnz_native = 0;
    int * csrRowPtrA_native = nullptr;
    int * csrColIndA_native = nullptr;
    size_t * pBufferSizeInBytes_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    nnz_native = (int)nnz;
    csrRowPtrA_native = (int *)getPointer(env, csrRowPtrA);
    csrColIndA_native = (int *)getPointer(env, csrColIndA);
    if (!initNative(env, pBufferSizeInBytes, pBufferSizeInBytes_native, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseXcsrsort_bufferSizeExt(handle_native, m_native, n_native, nnz_native,
        csrRowPtrA_native, csrColIndA_native, pBufferSizeInBytes_native);

    if (!releaseNative(env, pBufferSizeInBytes_native, pBufferSizeInBytes, true)) return CUJAVA_CUSPARSE_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateNative(JNIEnv *env, jclass cls, jobject handle) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseCreate");

    Logger::log(LOG_TRACE, "Executing cusparseCreate(handle=%p)\n", handle);

    // Declare native variables
    cusparseHandle_t handle_native;

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCreate(&handle_native);
    setNativePointerValue(env, handle, (jlong)handle_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateIdentityPermutationNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject p) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cusparseCreateIdentityPermutation");
    CUJAVA_REQUIRE_NONNULL(env, p, "p", "cusparseCreateIdentityPermutation");

    Logger::log(LOG_TRACE, "Executing cusparseCreateIdentityPermutation(handle=%p, n=%d, p=%p)\n", handle, n, p);

    // Declare native variables
    cusparseHandle_t handle_native;
    int n_native = 0;
    int * p_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    n_native = (int)n;
    p_native = (int *)getPointer(env, p);

    // Cusparse API call
    cusparseStatus_t jniResult_native = cusparseCreateIdentityPermutation(handle_native, n_native, p_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}
