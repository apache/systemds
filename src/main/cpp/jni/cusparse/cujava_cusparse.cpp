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
  (JNIEnv *env, jclass, jobject handle, jint opA, jint opB,
   jobject alpha, jobject matA, jobject matB, jobject beta, jobject matC,
   jint computeType, jint alg, jobject spgemmDescr) {
    if (!handle || !alpha || !matA || !matB || !beta || !matC || !spgemmDescr) {
        ThrowByName(env, "java/lang/NullPointerException", "Null argument in cusparseSpGEMM_copy");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cusparseSpGEMM_copy\n");

    cusparseHandle_t h = (cusparseHandle_t)getNativePointerValue(env, handle);
    cusparseOperation_t aOp = (cusparseOperation_t)opA;
    cusparseOperation_t bOp = (cusparseOperation_t)opB;
    PointerData* alphaPD = initPointerData(env, alpha); if (!alphaPD) return CUJAVA_INTERNAL_ERROR;
    void* alphaPtr = alphaPD->getPointer(env);

    cusparseConstSpMatDescr_t A = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    cusparseConstSpMatDescr_t B = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matB);
    PointerData* betaPD = initPointerData(env, beta);  if (!betaPD)  { releasePointerData(env, alphaPD, JNI_ABORT); return CUJAVA_INTERNAL_ERROR; }
    void* betaPtr = betaPD->getPointer(env);

    cusparseSpMatDescr_t C = (cusparseSpMatDescr_t)getNativePointerValue(env, matC);
    cudaDataType ct = (cudaDataType)computeType;
    cusparseSpGEMMAlg_t al = (cusparseSpGEMMAlg_t)alg;
    cusparseSpGEMMDescr_t D  = (cusparseSpGEMMDescr_t)getNativePointerValue(env, spgemmDescr);

    cusparseStatus_t st = cusparseSpGEMM_copy(h, aOp, bOp, alphaPtr, A, B, betaPtr, C, ct, al, D);

    // alpha/beta are inputs → no commit
    if (!releasePointerData(env, alphaPD, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, betaPD,  JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;

    return (jint)st;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseGetMatIndexBaseNative(JNIEnv *env, jclass cls, jobject descrA) {
    // Null-checks for non-primitive arguments
    if (descrA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrA' is null for cusparseGetMatIndexBase");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseGetMatIndexBase(descrA=%p)\n", descrA);

    cusparseMatDescr_t descrA_native;
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    cusparseIndexBase_t jniResult_native = cusparseGetMatIndexBase(descrA_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateCsrNative
  (JNIEnv *env, jclass cls, jobject spMatDescr, jlong rows, jlong cols, jlong nnz, jobject csrRowOffsets,
   jobject csrColInd, jobject csrValues, jint csrRowOffsetsType, jint csrColIndType, jint idxBase, jint valueType) {
    // Null-checks for non-primitive arguments
    if (spMatDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'spMatDescr' is null for cusparseCreateCsr");
        return CUJAVA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cusparseCreateCsr(spMatDescr=%p, rows=%ld, cols=%ld, nnz=%ld, csrRowOffsets=%p, csrColInd=%p, csrValues=%p, csrRowOffsetsType=%d, csrColIndType=%d, idxBase=%d, valueType=%d)\n",
        spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType);

    // Native variable declarations
    cusparseSpMatDescr_t spMatDescr_native;
    int64_t rows_native = 0;
    int64_t cols_native = 0;
    int64_t nnz_native = 0;
    void * csrRowOffsets_native = NULL;
    void * csrColInd_native = NULL;
    void * csrValues_native = NULL;
    cusparseIndexType_t csrRowOffsetsType_native;
    cusparseIndexType_t csrColIndType_native;
    cusparseIndexBase_t idxBase_native;
    cudaDataType valueType_native;

    // Obtain native variable values
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

    // Native function call
    cusparseStatus_t jniResult_native = cusparseCreateCsr(&spMatDescr_native, rows_native, cols_native, nnz_native, csrRowOffsets_native, csrColInd_native, csrValues_native, csrRowOffsetsType_native, csrColIndType_native, idxBase_native, valueType_native);

    // Write back native variable values
    setNativePointerValue(env, spMatDescr, (jlong)spMatDescr_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnVecNative(JNIEnv *env, jclass cls, jobject dnVecDescr, jlong size, jobject values, jint valueType) {
    if (dnVecDescr == nullptr){
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dnVecDescr' is null for cusparseCreateDnVec");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (values == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'values' is null for cusparseCreateDnVec");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseCreateDnVec(dnVecDescr=%p, size=%ld, values=%p, valueType=%d)\n",
        dnVecDescr, size, values, valueType);

    // Native variable declarations
    cusparseDnVecDescr_t dnVecDescr_native;
    int64_t size_native = 0;
    void * values_native = NULL;
    cudaDataType valueType_native;

    size_native = (int64_t)size;
    values_native = (void *)getPointer(env, values);
    valueType_native = (cudaDataType)valueType;

    // Native function call
    cusparseStatus_t jniResult_native = cusparseCreateDnVec(&dnVecDescr_native, size_native, values_native, valueType_native);
    setNativePointerValue(env, dnVecDescr, (jlong)dnVecDescr_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMV_1bufferSizeNative
  (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta, jobject vecY, jint computeType, jint alg, jlongArray bufferSize) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matA' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (vecX == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'vecX' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (vecY == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'vecY' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (bufferSize == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bufferSize' is null for cusparseSpMV_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseSpMV_bufferSize(handle=%p, opA=%d, alpha=%p, matA=%p, vecX=%p, beta=%p, vecY=%p, computeType=%d, alg=%d, bufferSize=%p)\n",
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);

    // Native variable declarations
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    void * alpha_native = NULL;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnVecDescr_t vecX_native;
    void * beta_native = NULL;
    cusparseDnVecDescr_t vecY_native;
    cudaDataType computeType_native;
    cusparseSpMVAlg_t alg_native;
    size_t * bufferSize_native = NULL;

    // Obtain native variable values
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) return CUJAVA_INTERNAL_ERROR;

    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    vecX_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, vecX);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) return CUJAVA_INTERNAL_ERROR;

    beta_native = (void *)beta_pointerData->getPointer(env);
    vecY_native = (cusparseDnVecDescr_t)getNativePointerValue(env, vecY);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMVAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_INTERNAL_ERROR;

    // Native function call
    cusparseStatus_t jniResult_native = cusparseSpMV_bufferSize(handle_native, opA_native, alpha_native, matA_native, vecX_native, beta_native, vecY_native, computeType_native, alg_native, bufferSize_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMVNative
 (JNIEnv *env, jclass cls, jobject handle, jint opA, jobject alpha, jobject matA, jobject vecX, jobject beta, jobject vecY, jint computeType, jint alg, jobject externalBuffer) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matA' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (vecX == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'vecX' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (vecY == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'vecY' is null for cusparseSpMV");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseSpMV(handle=%p, opA=%d, alpha=%p, matA=%p, vecX=%p, beta=%p, vecY=%p, computeType=%d, alg=%d, externalBuffer=%p)\n",
        handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);

    // Native variable declarations
    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    void * alpha_native = NULL;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnVecDescr_t vecX_native;
    void * beta_native = NULL;
    cusparseDnVecDescr_t vecY_native;
    cudaDataType computeType_native;
    cusparseSpMVAlg_t alg_native;
    void * externalBuffer_native = NULL;

    // Obtain native variable values
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    vecX_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, vecX);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    vecY_native = (cusparseDnVecDescr_t)getNativePointerValue(env, vecY);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMVAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    // Native function call
    cusparseStatus_t jniResult_native = cusparseSpMV(handle_native, opA_native, alpha_native, matA_native, vecX_native, beta_native, vecY_native, computeType_native, alg_native, externalBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyNative(JNIEnv *env, jclass cls, jobject handle) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseDestroy");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseDestroy(handle=%p)\n", handle);

    cusparseHandle_t handle_native;
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    cusparseStatus_t jniResult_native = cusparseDestroy(handle_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnVecNative(JNIEnv *env, jclass cls, jobject dnVecDescr) {
    if (dnVecDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dnVecDescr' is null for cusparseDestroyDnVec");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cusparseDestroyDnVec(dnVecDescr=%p)\n", dnVecDescr);

    cusparseConstDnVecDescr_t dnVecDescr_native;
    dnVecDescr_native = (cusparseConstDnVecDescr_t)getNativePointerValue(env, dnVecDescr);
    cusparseStatus_t jniResult_native = cusparseDestroyDnVec(dnVecDescr_native);

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroyDnMatNative(JNIEnv *env, jclass cls, jobject dnMatDescr) {
    if (dnMatDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dnMatDescr' is null for cusparseDestroyDnMat");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cusparseDestroyDnMat(dnMatDescr=%p)\n", dnMatDescr);

    cusparseConstDnMatDescr_t dnMatDescr_native;
    dnMatDescr_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, dnMatDescr);
    cusparseStatus_t jniResult_native = cusparseDestroyDnMat(dnMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDestroySpMatNative(JNIEnv *env, jclass cls, jobject spMatDescr) {
    if (spMatDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'spMatDescr' is null for cusparseDestroySpMat");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cusparseDestroySpMat(spMatDescr=%p)\n", spMatDescr);

    cusparseConstSpMatDescr_t spMatDescr_native;
    spMatDescr_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, spMatDescr);
    cusparseStatus_t jniResult_native = cusparseDestroySpMat(spMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMMNative
(JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta, jobject matC, jint computeType, jint alg, jobject externalBuffer) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matA' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matB' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matC' is null for cusparseSpMM");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseSpMM(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, externalBuffer=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);

    // Native variable declarations
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

    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMMAlg_t)alg;
    externalBuffer_native = (void *)getPointer(env, externalBuffer);

    cusparseStatus_t jniResult_native = cusparseSpMM(handle_native, opA_native, opB_native, alpha_native, matA_native,
        matB_native, beta_native, matC_native, computeType_native, alg_native, externalBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseSpMM_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint opA, jint opB, jobject alpha, jobject matA, jobject matB, jobject beta,
     jobject matC, jint computeType, jint alg, jlongArray bufferSize) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matA' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matB' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (matC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'matC' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (bufferSize == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bufferSize' is null for cusparseSpMM_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseSpMM_bufferSize(handle=%p, opA=%d, opB=%d, alpha=%p, matA=%p, matB=%p, beta=%p, matC=%p, computeType=%d, alg=%d, bufferSize=%p)\n",
        handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize);

    cusparseHandle_t handle_native;
    cusparseOperation_t opA_native;
    cusparseOperation_t opB_native;
    void * alpha_native = NULL;
    cusparseConstSpMatDescr_t matA_native;
    cusparseConstDnMatDescr_t matB_native;
    void * beta_native = nullptr;
    cusparseDnMatDescr_t matC_native;
    cudaDataType computeType_native;
    cusparseSpMMAlg_t alg_native;
    size_t * bufferSize_native = nullptr;

    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    opA_native = (cusparseOperation_t)opA;
    opB_native = (cusparseOperation_t)opB;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr)
    {
        return CUJAVA_INTERNAL_ERROR;
    }
    alpha_native = (void *)alpha_pointerData->getPointer(env);
    matA_native = (cusparseConstSpMatDescr_t)getNativePointerValue(env, matA);
    matB_native = (cusparseConstDnMatDescr_t)getNativePointerValue(env, matB);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    beta_native = (void *)beta_pointerData->getPointer(env);
    matC_native = (cusparseDnMatDescr_t)getNativePointerValue(env, matC);
    computeType_native = (cudaDataType)computeType;
    alg_native = (cusparseSpMMAlg_t)alg;
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_INTERNAL_ERROR;

    cusparseStatus_t jniResult_native = cusparseSpMM_bufferSize(handle_native, opA_native, opB_native, alpha_native,
        matA_native, matB_native, beta_native, matC_native, computeType_native, alg_native, bufferSize_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCreateDnMatNative
    (JNIEnv *env, jclass cls, jobject dnMatDescr, jlong rows, jlong cols, jlong ld, jobject values, jint valueType, jint order) {

    if (dnMatDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dnMatDescr' is null for cusparseCreateDnMat");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (values == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'values' is null for cusparseCreateDnMat");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseCreateDnMat(dnMatDescr=%p, rows=%ld, cols=%ld, ld=%ld, values=%p, valueType=%d, order=%d)\n",
        dnMatDescr, rows, cols, ld, values, valueType, order);

    cusparseDnMatDescr_t dnMatDescr_native;
    int64_t rows_native = 0;
    int64_t cols_native = 0;
    int64_t ld_native = 0;
    void * values_native = nullptr;
    cudaDataType valueType_native;
    cusparseOrder_t order_native;

    rows_native = (int64_t)rows;
    cols_native = (int64_t)cols;
    ld_native = (int64_t)ld;
    values_native = (void *)getPointer(env, values);
    valueType_native = (cudaDataType)valueType;
    order_native = (cusparseOrder_t)order;

    cusparseStatus_t jniResult_native = cusparseCreateDnMat(&dnMatDescr_native, rows_native, cols_native, ld_native,
        values_native, valueType_native, order_native);
    setNativePointerValue(env, dnMatDescr, (jlong)dnMatDescr_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsrSetPointersNative
    (JNIEnv *env, jclass cls, jobject spMatDescr, jobject csrRowOffsets, jobject csrColInd, jobject csrValues) {
    if (spMatDescr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'spMatDescr' is null for cusparseCsrSetPointers");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrRowOffsets == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrRowOffsets' is null for cusparseCsrSetPointers");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrColInd == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrColInd' is null for cusparseCsrSetPointers");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrValues == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrValues' is null for cusparseCsrSetPointers");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseCsrSetPointers(spMatDescr=%p, csrRowOffsets=%p, csrColInd=%p, csrValues=%p)\n",
        spMatDescr, csrRowOffsets, csrColInd, csrValues);

    cusparseSpMatDescr_t spMatDescr_native;
    void * csrRowOffsets_native = nullptr;
    void * csrColInd_native = nullptr;
    void * csrValues_native = nullptr;

    spMatDescr_native = (cusparseSpMatDescr_t)getNativePointerValue(env, spMatDescr);
    csrRowOffsets_native = (void *)getPointer(env, csrRowOffsets);
    csrColInd_native = (void *)getPointer(env, csrColInd);
    csrValues_native = (void *)getPointer(env, csrValues);
    cusparseStatus_t jniResult_native = cusparseCsrSetPointers(spMatDescr_native, csrRowOffsets_native, csrColInd_native, csrValues_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr,
     jobject csrColInd, jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jobject buffer) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }

    if (csrVal == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrVal' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrRowPtr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrRowPtr' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrColInd == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrColInd' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscVal == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscVal' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscColPtr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscColPtr' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscRowInd == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscRowInd' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (buffer == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'buffer' is null for cusparseCsr2cscEx2");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseCsr2cscEx2(handle=%p, m=%d, n=%d, nnz=%d, csrVal=%p, csrRowPtr=%p, csrColInd=%p, cscVal=%p, cscColPtr=%p, cscRowInd=%p, valType=%d, copyValues=%d, idxBase=%d, alg=%d, buffer=%p)\n",
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer);

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

    // Native arguments
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

    cusparseStatus_t jniResult_native = cusparseCsr2cscEx2(handle_native, m_native, n_native, nnz_native,
        csrVal_native, csrRowPtr_native, csrColInd_native, cscVal_native, cscColPtr_native, cscRowInd_native,
        valType_native, copyValues_native, idxBase_native, alg_native, buffer_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseCsr2cscEx2_1bufferSizeNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jint nnz, jobject csrVal, jobject csrRowPtr, jobject csrColInd,
     jobject cscVal, jobject cscColPtr, jobject cscRowInd, jint valType, jint copyValues, jint idxBase, jint alg, jlongArray bufferSize) {

    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrVal == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrVal' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrRowPtr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrRowPtr' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrColInd == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrColInd' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscVal == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscVal' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscColPtr == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscColPtr' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (cscRowInd == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cscRowInd' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (bufferSize == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'bufferSize' is null for cusparseCsr2cscEx2_bufferSize");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseCsr2cscEx2_bufferSize(handle=%p, m=%d, n=%d, nnz=%d, csrVal=%p, csrRowPtr=%p, csrColInd=%p, cscVal=%p, cscColPtr=%p, cscRowInd=%p, valType=%d, copyValues=%d, idxBase=%d, alg=%d, bufferSize=%p)\n",
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize);

    // declare native arguments
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    int nnz_native = 0;
    void * csrVal_native = NULL;
    int * csrRowPtr_native = NULL;
    int * csrColInd_native = NULL;
    void * cscVal_native = NULL;
    int * cscColPtr_native = NULL;
    int * cscRowInd_native = NULL;
    cudaDataType valType_native;
    cusparseAction_t copyValues_native;
    cusparseIndexBase_t idxBase_native;
    cusparseCsr2CscAlg_t alg_native;
    size_t * bufferSize_native = NULL;

    // get Native arguments
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
    if (!initNative(env, bufferSize, bufferSize_native, true)) return CUJAVA_INTERNAL_ERROR;

    cusparseStatus_t jniResult_native = cusparseCsr2cscEx2_bufferSize
        (handle_native, m_native, n_native, nnz_native, csrVal_native, csrRowPtr_native, csrColInd_native, cscVal_native,
         cscColPtr_native, cscRowInd_native, valType_native, copyValues_native, idxBase_native, alg_native, bufferSize_native);
    if (!releaseNative(env, bufferSize_native, bufferSize, true)) return CUJAVA_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2Native
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB,
     jobject csrSortedRowPtrB, jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jobject pBuffer) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrA' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    // nnzA is primitive
    if (csrSortedValA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValA' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrA' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndA' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrB' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    // nnzB is primitive
    if (csrSortedValB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValB' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrB' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndB' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrC' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedValC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValC' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrC' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndC' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (pBuffer == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pBuffer' is null for cusparseDcsrgeam2");
        return CUJAVA_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cusparseDcsrgeam2(handle=%p, m=%d, n=%d, alpha=%p, descrA=%p, nnzA=%d, csrSortedValA=%p, csrSortedRowPtrA=%p, csrSortedColIndA=%p, beta=%p, descrB=%p, nnzB=%d, csrSortedValB=%p, csrSortedRowPtrB=%p, csrSortedColIndB=%p, descrC=%p, csrSortedValC=%p, csrSortedRowPtrC=%p, csrSortedColIndC=%p, pBuffer=%p)\n",
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);

    // Native variable declarations
    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    cusparseMatDescr_t descrA_native;
    int nnzA_native = 0;
    double * csrSortedValA_native = nullptr;
    int * csrSortedRowPtrA_native = NULL;
    int * csrSortedColIndA_native = NULL;
    double * beta_native = NULL;
    cusparseMatDescr_t descrB_native;
    int nnzB_native = 0;
    double * csrSortedValB_native = NULL;
    int * csrSortedRowPtrB_native = NULL;
    int * csrSortedColIndB_native = NULL;
    cusparseMatDescr_t descrC_native;
    double * csrSortedValC_native = NULL;
    int * csrSortedRowPtrC_native = NULL;
    int * csrSortedColIndC_native = NULL;
    void * pBuffer_native = NULL;

    // Obtain native variable values
    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    nnzA_native = (int)nnzA;
    csrSortedValA_native = (double *)getPointer(env, csrSortedValA);
    csrSortedRowPtrA_native = (int *)getPointer(env, csrSortedRowPtrA);
    csrSortedColIndA_native = (int *)getPointer(env, csrSortedColIndA);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
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

    cusparseStatus_t jniResult_native = cusparseDcsrgeam2(handle_native, m_native, n_native, alpha_native, descrA_native,
        nnzA_native, csrSortedValA_native, csrSortedRowPtrA_native, csrSortedColIndA_native, beta_native, descrB_native,
         nnzB_native, csrSortedValB_native, csrSortedRowPtrB_native, csrSortedColIndB_native, descrC_native, csrSortedValC_native,
         csrSortedRowPtrC_native, csrSortedColIndC_native, pBuffer_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cusparse_CuJavaCusparse_cusparseDcsrgeam2_1bufferSizeExtNative
    (JNIEnv *env, jclass cls, jobject handle, jint m, jint n, jobject alpha, jobject descrA, jint nnzA, jobject csrSortedValA,
     jobject csrSortedRowPtrA, jobject csrSortedColIndA, jobject beta, jobject descrB, jint nnzB, jobject csrSortedValB, jobject csrSortedRowPtrB,
     jobject csrSortedColIndB, jobject descrC, jobject csrSortedValC, jobject csrSortedRowPtrC, jobject csrSortedColIndC, jlongArray pBufferSizeInBytes) {
    if (handle == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (alpha == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'alpha' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrA' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    // nnzA is primitive
    if (csrSortedValA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValA' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrA' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndA == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndA' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (beta == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'beta' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrB' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    // nnzB is primitive
    if (csrSortedValB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValB' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrB' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndB == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndB' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (descrC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'descrC' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedValC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedValC' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedRowPtrC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedRowPtrC' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (csrSortedColIndC == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'csrSortedColIndC' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (pBufferSizeInBytes == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pBufferSizeInBytes' is null for cusparseDcsrgeam2_bufferSizeExt");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cusparseDcsrgeam2_bufferSizeExt(handle=%p, m=%d, n=%d, alpha=%p, descrA=%p, nnzA=%d, csrSortedValA=%p, csrSortedRowPtrA=%p, csrSortedColIndA=%p, beta=%p, descrB=%p, nnzB=%d, csrSortedValB=%p, csrSortedRowPtrB=%p, csrSortedColIndB=%p, descrC=%p, csrSortedValC=%p, csrSortedRowPtrC=%p, csrSortedColIndC=%p, pBufferSizeInBytes=%p)\n",
        handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);

    cusparseHandle_t handle_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = NULL;
    cusparseMatDescr_t descrA_native;
    int nnzA_native = 0;
    double * csrSortedValA_native = NULL;
    int * csrSortedRowPtrA_native = NULL;
    int * csrSortedColIndA_native = NULL;
    double * beta_native = NULL;
    cusparseMatDescr_t descrB_native;
    int nnzB_native = 0;
    double * csrSortedValB_native = NULL;
    int * csrSortedRowPtrB_native = NULL;
    int * csrSortedColIndB_native = NULL;
    cusparseMatDescr_t descrC_native;
    double * csrSortedValC_native = NULL;
    int * csrSortedRowPtrC_native = NULL;
    int * csrSortedColIndC_native = NULL;
    size_t * pBufferSizeInBytes_native = NULL;

    handle_native = (cusparseHandle_t)getNativePointerValue(env, handle);
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    descrA_native = (cusparseMatDescr_t)getNativePointerValue(env, descrA);
    nnzA_native = (int)nnzA;
    csrSortedValA_native = (double *)getPointer(env, csrSortedValA);
    csrSortedRowPtrA_native = (int *)getPointer(env, csrSortedRowPtrA);
    csrSortedColIndA_native = (int *)getPointer(env, csrSortedColIndA);
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
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
    if (!initNative(env, pBufferSizeInBytes, pBufferSizeInBytes_native, true)) return CUJAVA_INTERNAL_ERROR;

    cusparseStatus_t jniResult_native = cusparseDcsrgeam2_bufferSizeExt(handle_native, m_native, n_native, alpha_native,
        descrA_native, nnzA_native, csrSortedValA_native, csrSortedRowPtrA_native, csrSortedColIndA_native, beta_native,
        descrB_native, nnzB_native, csrSortedValB_native, csrSortedRowPtrB_native, csrSortedColIndB_native, descrC_native,
        csrSortedValC_native, csrSortedRowPtrC_native, csrSortedColIndC_native, pBufferSizeInBytes_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releaseNative(env, pBufferSizeInBytes_native, pBufferSizeInBytes, true)) return CUJAVA_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}
