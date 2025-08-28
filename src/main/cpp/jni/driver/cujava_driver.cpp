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

#include "cujava_driver.hpp"
#include "cujava_driver_common.hpp"


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




JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxCreateNative
  (JNIEnv *env, jclass cls, jobject pctx, jint flags, jobject dev) {
    if (pctx == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pctx' is null for cuCtxCreate");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (dev == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuCtxCreate");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxCreate\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    CUcontext nativePctx;
    int result = cuCtxCreate(&nativePctx, (int)flags, nativeDev);
    setNativePointerValue(env, pctx, (jlong)nativePctx);

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetNative
  (JNIEnv *env, jclass cls, jobject device, jint ordinal) {
    if (device == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'device' is null for cuDeviceGet");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuDeviceGet for device %ld\n", ordinal);

    CUdevice nativeDevice;
    int result = cuDeviceGet(&nativeDevice, ordinal);
    setNativePointerValue(env, device, (jlong)nativeDevice);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetCountNative
  (JNIEnv *env, jclass cls, jintArray count) {
    if (count == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'count' is null for cuDeviceGetCount");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuDeviceGetCount\n");

    int nativeCount = 0;
    int result = cuDeviceGetCount(&nativeCount);
    if (!set(env, count, 0, nativeCount)) return CUJAVA_INTERNAL_ERROR;
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuInitNative
  (JNIEnv *env, jclass cls, jint flags) {
    Logger::log(LOG_TRACE, "Executing cuInit\n");

    int result = cuInit((unsigned int)flags);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuLaunchKernelNative
  (JNIEnv *env, jclass, jobject f, jint gridDimX, jint gridDimY, jint gridDimZ,
   jint blockDimX, jint blockDimY, jint blockDimZ, jint sharedMemBytes,
   jobject hStream, jobject kernelParams, jobject extra) {
    if (f == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'f' is null for cuLaunchKernel");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuLaunchKernel\n");

    CUfunction nativeF    = (CUfunction)getNativePointerValue(env, f);
    CUstream   nativeHStr = (CUstream)  getNativePointerValue(env, hStream);

    PointerData *kernelParamsPD = nullptr;
    void **nativeKernelParams   = nullptr;
    if (kernelParams != nullptr) {
        kernelParamsPD = initPointerData(env, kernelParams);
        if (kernelParamsPD == nullptr) return CUJAVA_INTERNAL_ERROR;
        nativeKernelParams = (void**)kernelParamsPD->getPointer(env);
    }

    PointerData *extraPD = nullptr;
    void **nativeExtra   = nullptr;
    if (extra != nullptr) {
        extraPD = initPointerData(env, extra);
        if (extraPD == nullptr) return CUJAVA_INTERNAL_ERROR;
        nativeExtra = (void**)extraPD->getPointer(env);
    }

    int result = cuLaunchKernel(
        nativeF,
        (unsigned int)gridDimX, (unsigned int)gridDimY, (unsigned int)gridDimZ,
        (unsigned int)blockDimX, (unsigned int)blockDimY, (unsigned int)blockDimZ,
        (unsigned int)sharedMemBytes,
        nativeHStr,
        nativeKernelParams,
        nativeExtra);

    if (!releasePointerData(env, kernelParamsPD, 0)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, extraPD,        0)) return CUJAVA_INTERNAL_ERROR;

    return result;
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleGetFunctionNative
  (JNIEnv *env, jclass, jobject hfunc, jobject hmod, jstring name) {
    if (hfunc == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hfunc' is null for cuModuleGetFunction");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (hmod  == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleGetFunction");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (name  == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'name' is null for cuModuleGetFunction");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuModuleGetFunction\n");

    CUmodule   nativeHmod  = (CUmodule)getNativePointerValue(env, hmod);
    char*      nativeName  = toNativeCString(env, name);
    if (!nativeName) return CUJAVA_INTERNAL_ERROR;

    CUfunction nativeHfunc = nullptr;
    int result = cuModuleGetFunction(&nativeHfunc, nativeHmod, nativeName);

    delete[] nativeName;
    setNativePointerValue(env, hfunc, (jlong)nativeHfunc);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleLoadDataExNative
  (JNIEnv *env, jclass, jobject phMod, jobject p, jint numOptions, jintArray options, jobject optionValues)
{
    if (phMod == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'phMod' is null for cuModuleLoadDataEx");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (p == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'p' is null for cuModuleLoadDataEx");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (options == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'options' is null for cuModuleLoadDataEx");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (optionValues == nullptr) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'optionValues' is null for cuModuleLoadDataEx");
        return CUJAVA_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cuModuleLoadDataEx\n");

    CUjit_option *nativeOptions = nullptr;
    {
        jint *opts = env->GetIntArrayElements(options, nullptr);
        if (opts == nullptr) return CUJAVA_INTERNAL_ERROR;

        nativeOptions = new CUjit_option[(size_t)numOptions];
        for (int i = 0; i < numOptions; ++i) nativeOptions[i] = (CUjit_option)opts[i];

        env->ReleaseIntArrayElements(options, opts, JNI_ABORT);
    }

    // Pointers for 'p' (module data) and 'optionValues' (void** for JIT options)
    CUmodule nativeModule;

    PointerData *pPD = initPointerData(env, p);
    if (pPD == nullptr) { delete[] nativeOptions; return CUJAVA_INTERNAL_ERROR; }

    PointerData *ovPD = initPointerData(env, optionValues);
    if (ovPD == nullptr) {
        releasePointerData(env, pPD, JNI_ABORT);
        delete[] nativeOptions;
        return CUJAVA_INTERNAL_ERROR;
    }
    void **nativeOptionValues = (void**)ovPD->getPointer(env);

    int result = cuModuleLoadDataEx(
        &nativeModule,
        (void*)pPD->getPointer(env),
        (unsigned int)numOptions,
        nativeOptions,
        nativeOptionValues);

    delete[] nativeOptions;

    setNativePointerValue(env, phMod, (jlong)nativeModule);

    // p is input-only → no-commit; optionValues may receive outputs → commit
    if (!releasePointerData(env, pPD, JNI_ABORT)) return CUJAVA_INTERNAL_ERROR;
    if (!releasePointerData(env, ovPD, 0))        return CUJAVA_INTERNAL_ERROR;

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemAllocNative
  (JNIEnv *env, jclass cls, jobject dptr, jlong bytesize) {
    if (dptr == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemAlloc");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemAlloc of %ld bytes\n", (long)bytesize);

    CUdeviceptr nativeDptr;
    int result = cuMemAlloc(&nativeDptr, (size_t)bytesize);
    setPointer(env, dptr, (jlong)nativeDptr);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleUnloadNative
  (JNIEnv *env, jclass cls, jobject hmod) {
    if (hmod == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'hmod' is null for cuModuleUnload");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuModuleUnload\n");

    CUmodule nativeHmod = (CUmodule)getNativePointerValue(env, hmod);
    int result = cuModuleUnload(nativeHmod);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxDestroyNative
  (JNIEnv *env, jclass cls, jobject ctx) {
    if (ctx == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'ctx' is null for cuCtxDestroy");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuCtxDestroy\n");

    CUcontext nativeCtx = (CUcontext)getNativePointerValue(env, ctx);
    int result = cuCtxDestroy(nativeCtx);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemFreeNative
  (JNIEnv *env, jclass cls, jobject dptr) {
    if (dptr == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dptr' is null for cuMemFree");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemFree\n");

    CUdeviceptr nativeDptr = (CUdeviceptr)getPointer(env, dptr);
    int result = cuMemFree(nativeDptr);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemcpyDtoHNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcDevice, jlong ByteCount) {
    if (dstHost == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dstHost' is null for cuMemcpyDtoH");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (srcDevice == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'srcDevice' is null for cuMemcpyDtoH");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuMemcpyDtoH of %d bytes\n", (size_t)ByteCount);

    PointerData *dstHostPointerData = initPointerData(env, dstHost);
    if (dstHostPointerData == NULL) {
        return CUJAVA_INTERNAL_ERROR;
    }
    CUdeviceptr nativeSrcDevice = (CUdeviceptr)getPointer(env, srcDevice);
    void *nativeDstHost = (void*)dstHostPointerData->getPointer(env);

    int result = cuMemcpyDtoH(nativeDstHost, nativeSrcDevice, (size_t)ByteCount);

    if (!releasePointerData(env, dstHostPointerData, 0)) return CUJAVA_INTERNAL_ERROR;

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxSynchronizeNative
  (JNIEnv *env, jclass cls) {
    Logger::log(LOG_TRACE, "Executing cuCtxSynchronize\n");

    return cuCtxSynchronize();
}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetAttributeNative
  (JNIEnv *env, jclass cls, jintArray pi, jint CUdevice_attribute_attrib, jobject dev) {
    if (pi == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'pi' is null for cuDeviceGetAttribute");
        return CUJAVA_INTERNAL_ERROR;
    }
    if (dev == NULL) {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'dev' is null for cuDeviceGetAttribute");
        return CUJAVA_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cuDeviceGetAttribute\n");

    CUdevice nativeDev = (CUdevice)(intptr_t)getNativePointerValue(env, dev);
    int nativePi = 0;
    int result = cuDeviceGetAttribute(&nativePi, (CUdevice_attribute)CUdevice_attribute_attrib, nativeDev);
    if (!set(env, pi, 0, nativePi)) return CUJAVA_INTERNAL_ERROR;
    return result;
}
