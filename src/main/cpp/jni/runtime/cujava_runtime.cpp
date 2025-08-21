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


#include "cujava_runtime.hpp"
#include "cujava_runtime_common.hpp"


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


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemcpyNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong count, jint kind) {
    if (dst == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dst' is null for cudaMemcpy");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (src == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'src' is null for cudaMemcpy");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemcpy of %ld bytes\n", (long)count);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == NULL) {
        return CUJAVA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == NULL) {
        return CUJAVA_INTERNAL_ERROR;
    }

    // Execute the cudaMemcpy operation
    int result = CUJAVA_INTERNAL_ERROR;
    if (kind == cudaMemcpyHostToHost) {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to host\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToHost);
    }
    else if (kind == cudaMemcpyHostToDevice) {
        Logger::log(LOG_TRACE, "Copying %ld bytes from host to device\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyHostToDevice);
    }
    else if (kind == cudaMemcpyDeviceToHost) {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to host\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToHost);
    }
    else if (kind == cudaMemcpyDeviceToDevice) {
        Logger::log(LOG_TRACE, "Copying %ld bytes from device to device\n", (long)count);
        result = cudaMemcpy((void*)dstPointerData->getPointer(env), (void*)srcPointerData->getPointer(env), (size_t)count, cudaMemcpyDeviceToDevice);
    }
    else {
        Logger::log(LOG_ERROR, "Invalid cudaMemcpyKind given: %d\n", kind);
        return cudaErrorInvalidMemcpyDirection;
    }

    // Release the pointer data
    if (!releasePointerData(env, dstPointerData)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, srcPointerData, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMallocNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size) {
    if (devPtr == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMalloc");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMalloc of %ld bytes\n", (long)size);

    void *nativeDevPtr = NULL;
    int result = cudaMalloc(&nativeDevPtr, (size_t)size);
    setPointer(env, devPtr, (jlong)nativeDevPtr);

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaFreeNative
  (JNIEnv *env, jclass cls, jobject devPtr) {
    if (devPtr == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaFree");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaFree\n");

    void *nativeDevPtr = NULL;
    nativeDevPtr = getPointer(env, devPtr);
    int result = cudaFree(nativeDevPtr);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemsetNative
  (JNIEnv *env, jclass cls, jobject mem, jint c, jlong count) {
    if (mem == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'mem' is null for cudaMemset");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemset\n");

    void *nativeMem = getPointer(env, mem);

    int result = cudaMemset(nativeMem, (int)c, (size_t)count);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaDeviceSynchronizeNative
  (JNIEnv *env, jclass cls) {
    Logger::log(LOG_TRACE, "Executing cudaDeviceSynchronize\n");

    int result = cudaDeviceSynchronize();
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMallocManagedNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size, jint flags) {
    if (devPtr == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'devPtr' is null for cudaMallocManaged");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMallocManaged of %ld bytes\n", (long)size);

    void *nativeDevPtr = NULL;
    int result = cudaMallocManaged(&nativeDevPtr, (size_t)size, (unsigned int)flags);
    if (result == cudaSuccess) {
        if (flags == cudaMemAttachHost) {
            jobject object = env->NewDirectByteBuffer(nativeDevPtr, size);
            env->SetObjectField(devPtr, Pointer_buffer, object);
            env->SetObjectField(devPtr, Pointer_pointers, NULL);
            env->SetLongField(devPtr, Pointer_byteOffset, 0);
        }
        env->SetLongField(devPtr, NativePointerObject_nativePointer, (jlong)nativeDevPtr);
    }

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemGetInfoNative
  (JNIEnv *env, jclass cls, jlongArray freeBytes, jlongArray totalBytes) {
    if (freeBytes == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'freeBytes' is null for cudaMemGetInfo");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (totalBytes == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'freeBytes' is null for cudaMemGetInfo");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaMemGetInfo\n");

    size_t nativeFreeBytes = 0;
    size_t nativeTotalBytes = 0;

    int result = cudaMemGetInfo(&nativeFreeBytes, &nativeTotalBytes);

    if (!set(env, freeBytes, 0, (jlong)nativeFreeBytes)) return CUJAVA_INTERNAL_ERROR;
    if (!set(env, totalBytes, 0, (jlong)nativeTotalBytes)) return CUJAVA_INTERNAL_ERROR;

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaGetDeviceCountNative
  (JNIEnv *env, jclass cls, jintArray count) {
    if (count == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cudaGetDeviceCount");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDeviceCount\n");

    int nativeCount = 0;
    int result = cudaGetDeviceCount(&nativeCount);
    if (!set(env, count, 0, nativeCount)) return CUJAVA_INTERNAL_ERROR;
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaSetDeviceNative
  (JNIEnv *env, jclass cls, jint device) {
    Logger::log(LOG_TRACE, "Executing cudaSetDevice\n");

    return cudaSetDevice(device);
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaSetDeviceFlagsNative
  (JNIEnv *env, jclass cls, jint flags) {
    Logger::log(LOG_TRACE, "Executing cudaSetDeviceFlags\n");

    return cudaSetDeviceFlags((int)flags);
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaGetDeviceNative
  (JNIEnv *env, jclass cls, jintArray device) {
    if (device == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cudaGetDevice");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDevice\n");

    int nativeDevice = 0;
    int result = cudaGetDevice(&nativeDevice);
    if (!set(env, device, 0, nativeDevice)) return CUJAVA_INTERNAL_ERROR;
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaGetDevicePropertiesNative
  (JNIEnv *env, jclass cls, jobject prop, jint device) {
    if (prop == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'prop' is null for cudaGetDeviceProperties");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cudaGetDeviceProperties\n");

    cudaDeviceProp nativeProp;
    int result = cudaGetDeviceProperties(&nativeProp, device);

    setCudaDeviceProp(env, prop, nativeProp);
    return result;
}




