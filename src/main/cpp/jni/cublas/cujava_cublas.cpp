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


#include "cujava_cublas.hpp"
#include "cujava_cublas_common.hpp"

#define CUJAVA_REQUIRE_NONNULL(env, obj, name, method)                           \
    do {                                                                          \
        if ((obj) == nullptr) {                                                   \
            ThrowByName((env), "java/lang/NullPointerException",                  \
                        "Parameter '" name "' is null for " method);              \
            return CUJAVA_CUBLAS_INTERNAL_ERROR;                                  \
        }                                                                         \
    } while (0)



JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv *env = nullptr;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4)) {
        return JNI_ERR;
    }

    // Only what we need so far
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    return JNI_VERSION_1_4;
}



JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasCreateNative(JNIEnv *env, jclass cls, jobject handle) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasCreate");

    Logger::log(LOG_TRACE, "Executing cublasCreate(handle=%p)\n", handle);

    // Declare native variables
    cublasHandle_t handle_native;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasCreate(&handle_native);
    setNativePointerValue(env, handle, (jlong)handle_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDestroyNative(JNIEnv *env, jclass cls, jobject handle) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDestroy");

    Logger::log(LOG_TRACE, "Executing cublasDestroy(handle=%p)\n", handle);

    // Declare native variables
    cublasHandle_t handle_native;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDestroy(handle_native);

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgeamNative
    (JNIEnv *env, jclass cls, jobject handle, jint transa, jint transb, jint m, jint n, jobject alpha, jobject A,
     jint lda, jobject beta, jobject B, jint ldb, jobject C, jint ldc) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDgeam");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDgeam");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cublasDgeam");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cublasDgeam");
    CUJAVA_REQUIRE_NONNULL(env, B, "B", "cublasDgeam");
    CUJAVA_REQUIRE_NONNULL(env, C, "C", "cublasDgeam");

    Logger::log(LOG_TRACE, "Executing cublasDgeam(handle=%p, transa=%d, transb=%d, m=%d, n=%d, alpha=%p, A=%p, lda=%d, beta=%p, B=%p, ldb=%d, C=%p, ldc=%d)\n",
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

    // Declare native variables
    cublasHandle_t handle_native;
    cublasOperation_t transa_native;
    cublasOperation_t transb_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    double * A_native = nullptr;
    int lda_native = 0;
    double * beta_native = nullptr;
    double * B_native = nullptr;
    int ldb_native = 0;
    double * C_native = nullptr;
    int ldc_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    transa_native = (cublasOperation_t)transa;
    transb_native = (cublasOperation_t)transb;
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    B_native = (double *)getPointer(env, B);
    ldb_native = (int)ldb;
    C_native = (double *)getPointer(env, C);
    ldc_native = (int)ldc;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDgeam(handle_native, transa_native, transb_native, m_native, n_native, alpha_native,
        A_native, lda_native, beta_native, B_native, ldb_native, C_native, ldc_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDdotNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject x, jint incx, jobject y, jint incy, jobject result) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDdot");
    CUJAVA_REQUIRE_NONNULL(env, x, "x", "cublasDdot");
    CUJAVA_REQUIRE_NONNULL(env, y, "y", "cublasDdot");
    CUJAVA_REQUIRE_NONNULL(env, result, "result", "cublasDdot");

    Logger::log(LOG_TRACE, "Executing cublasDdot(handle=%p, n=%d, x=%p, incx=%d, y=%p, incy=%d, result=%p)\n",
        handle, n, x, incx, y, incy, result);

    // Declare native variables
    cublasHandle_t handle_native;
    int n_native = 0;
    double * x_native = nullptr;
    int incx_native = 0;
    double * y_native = nullptr;
    int incy_native = 0;
    double * result_native = nullptr;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    n_native = (int)n;
    x_native = (double *)getPointer(env, x);
    incx_native = (int)incx;
    y_native = (double *)getPointer(env, y);
    incy_native = (int)incy;
    PointerData *result_pointerData = initPointerData(env, result);
    if (result_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    result_native = (double *)result_pointerData->getPointer(env);

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDdot(handle_native, n_native, x_native, incx_native, y_native, incy_native, result_native);

    if (!isPointerBackedByNativeMemory(env, result)) {
        cudaDeviceSynchronize();                        // add cudart to CMake to cover runtime call
    }
    if (!releasePointerData(env, result_pointerData, 0)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgemvNative
    (JNIEnv *env, jclass cls, jobject handle, jint trans, jint m, jint n, jobject alpha, jobject A, jint lda,
     jobject x, jint incx, jobject beta, jobject y, jint incy) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDgemv");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDgemv");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cublasDgemv");
    CUJAVA_REQUIRE_NONNULL(env, x, "x", "cublasDgemv");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cublasDgemv");
    CUJAVA_REQUIRE_NONNULL(env, y, "y", "cublasDgemv");

    Logger::log(LOG_TRACE, "Executing cublasDgemv(handle=%p, trans=%d, m=%d, n=%d, alpha=%p, A=%p, lda=%d, x=%p, incx=%d, beta=%p, y=%p, incy=%d)\n",
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

    // Declare native variables
    cublasHandle_t handle_native;
    cublasOperation_t trans_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    double * A_native = nullptr;
    int lda_native = 0;
    double * x_native = nullptr;
    int incx_native = 0;
    double * beta_native = nullptr;
    double * y_native = nullptr;
    int incy_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    trans_native = (cublasOperation_t)trans;
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    x_native = (double *)getPointer(env, x);
    incx_native = (int)incx;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    y_native = (double *)getPointer(env, y);
    incy_native = (int)incy;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDgemv(handle_native, trans_native, m_native, n_native, alpha_native, A_native,
        lda_native, x_native, incx_native, beta_native, y_native, incy_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDgemmNative
    (JNIEnv *env, jclass cls, jobject handle, jint transa, jint transb, jint m, jint n, jint k, jobject alpha,
     jobject A, jint lda, jobject B, jint ldb, jobject beta, jobject C, jint ldc) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDgemm");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDgemm");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cublasDgemm");
    CUJAVA_REQUIRE_NONNULL(env, B, "B", "cublasDgemm");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cublasDgemm");
    CUJAVA_REQUIRE_NONNULL(env, C, "C", "cublasDgemm");

    Logger::log(LOG_TRACE, "Executing cublasDgemm(handle=%p, transa=%d, transb=%d, m=%d, n=%d, k=%d, alpha=%p, A=%p, lda=%d, B=%p, ldb=%d, beta=%p, C=%p, ldc=%d)\n",
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Declare native variables
    cublasHandle_t handle_native;
    cublasOperation_t transa_native;
    cublasOperation_t transb_native;
    int m_native = 0;
    int n_native = 0;
    int k_native = 0;
    double * alpha_native = nullptr;
    double * A_native = nullptr;
    int lda_native = 0;
    double * B_native = nullptr;
    int ldb_native = 0;
    double * beta_native = nullptr;
    double * C_native = nullptr;
    int ldc_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    transa_native = (cublasOperation_t)transa;
    transb_native = (cublasOperation_t)transb;
    m_native = (int)m;
    n_native = (int)n;
    k_native = (int)k;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == NULL) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    B_native = (double *)getPointer(env, B);
    ldb_native = (int)ldb;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    C_native = (double *)getPointer(env, C);
    ldc_native = (int)ldc;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDgemm(handle_native, transa_native, transb_native, m_native, n_native, k_native,
        alpha_native, A_native, lda_native, B_native, ldb_native, beta_native, C_native, ldc_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDsyrkNative
    (JNIEnv *env, jclass cls, jobject handle, jint uplo, jint trans, jint n, jint k, jobject alpha,
     jobject A, jint lda, jobject beta, jobject C, jint ldc) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDsyrk");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDsyrk");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cublasDsyrk");
    CUJAVA_REQUIRE_NONNULL(env, beta, "beta", "cublasDsyrk");
    CUJAVA_REQUIRE_NONNULL(env, C, "C", "cublasDsyrk");

    Logger::log(LOG_TRACE, "Executing cublasDsyrk(handle=%p, uplo=%d, trans=%d, n=%d, k=%d, alpha=%p, A=%p, lda=%d, beta=%p, C=%p, ldc=%d)\n",
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);

    // Declare native variables
    cublasHandle_t handle_native;
    cublasFillMode_t uplo_native;
    cublasOperation_t trans_native;
    int n_native = 0;
    int k_native = 0;
    double * alpha_native = nullptr;
    double * A_native = nullptr;
    int lda_native = 0;
    double * beta_native = nullptr;
    double * C_native = nullptr;
    int ldc_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    uplo_native = (cublasFillMode_t)uplo;
    trans_native = (cublasOperation_t)trans;
    n_native = (int)n;
    k_native = (int)k;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    PointerData *beta_pointerData = initPointerData(env, beta);
    if (beta_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    beta_native = (double *)beta_pointerData->getPointer(env);
    C_native = (double *)getPointer(env, C);
    ldc_native = (int)ldc;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDsyrk(handle_native, uplo_native, trans_native, n_native, k_native,
        alpha_native, A_native, lda_native, beta_native, C_native, ldc_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;
    if (!releasePointerData(env, beta_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDaxpyNative
    (JNIEnv *env, jclass cls, jobject handle, jint n, jobject alpha, jobject x, jint incx, jobject y, jint incy) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDaxpy");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDaxpy");
    CUJAVA_REQUIRE_NONNULL(env, x, "x", "cublasDaxpy");
    CUJAVA_REQUIRE_NONNULL(env, y, "y", "cublasDaxpy");

    Logger::log(LOG_TRACE, "Executing cublasDaxpy(handle=%p, n=%d, alpha=%p, x=%p, incx=%d, y=%p, incy=%d)\n",
        handle, n, alpha, x, incx, y, incy);

    // Declare native variables
    cublasHandle_t handle_native;
    int n_native = 0;
    double * alpha_native = nullptr;
    double * x_native = nullptr;
    int incx_native = 0;
    double * y_native = nullptr;
    int incy_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    x_native = (double *)getPointer(env, x);
    incx_native = (int)incx;
    y_native = (double *)getPointer(env, y);
    incy_native = (int)incy;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDaxpy(handle_native, n_native, alpha_native, x_native, incx_native, y_native, incy_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_cublas_CuJavaCublas_cublasDtrsmNative
    (JNIEnv *env, jclass cls, jobject handle, jint side, jint uplo, jint trans, jint diag, jint m,
     jint n, jobject alpha, jobject A, jint lda, jobject B, jint ldb) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, handle, "handle", "cublasDtrsm");
    CUJAVA_REQUIRE_NONNULL(env, alpha, "alpha", "cublasDtrsm");
    CUJAVA_REQUIRE_NONNULL(env, A, "A", "cublasDtrsm");
    CUJAVA_REQUIRE_NONNULL(env, B, "B", "cublasDtrsm");

    Logger::log(LOG_TRACE, "Executing cublasDtrsm(handle=%p, side=%d, uplo=%d, trans=%d, diag=%d, m=%d, n=%d, alpha=%p, A=%p, lda=%d, B=%p, ldb=%d)\n",
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);

    // Declare native variables
    cublasHandle_t handle_native;
    cublasSideMode_t side_native;
    cublasFillMode_t uplo_native;
    cublasOperation_t trans_native;
    cublasDiagType_t diag_native;
    int m_native = 0;
    int n_native = 0;
    double * alpha_native = nullptr;
    double * A_native = nullptr;
    int lda_native = 0;
    double * B_native = nullptr;
    int ldb_native = 0;

    // Copy Java inputs into native locals
    handle_native = (cublasHandle_t)getNativePointerValue(env, handle);
    side_native = (cublasSideMode_t)side;
    uplo_native = (cublasFillMode_t)uplo;
    trans_native = (cublasOperation_t)trans;
    diag_native = (cublasDiagType_t)diag;
    m_native = (int)m;
    n_native = (int)n;
    PointerData *alpha_pointerData = initPointerData(env, alpha);
    if (alpha_pointerData == nullptr) {
        return CUJAVA_CUBLAS_INTERNAL_ERROR;
    }
    alpha_native = (double *)alpha_pointerData->getPointer(env);
    A_native = (double *)getPointer(env, A);
    lda_native = (int)lda;
    B_native = (double *)getPointer(env, B);
    ldb_native = (int)ldb;

    // Cublas API call
    cublasStatus_t jniResult_native = cublasDtrsm(handle_native, side_native, uplo_native, trans_native, diag_native,
        m_native, n_native, alpha_native, A_native, lda_native, B_native, ldb_native);

    if (!releasePointerData(env, alpha_pointerData, JNI_ABORT)) return CUJAVA_CUBLAS_INTERNAL_ERROR;

    jint jniResult = (jint)jniResult_native;
    return jniResult;
}
