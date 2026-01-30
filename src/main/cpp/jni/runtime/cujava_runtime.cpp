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

#define CUJAVA_REQUIRE_NONNULL(env, obj, name, method)                           \
    do {                                                                          \
        if ((obj) == nullptr) {                                                   \
            ThrowByName((env), "java/lang/NullPointerException",                  \
                        "Parameter '" name "' is null for " method);              \
            return CUJAVA_INTERNAL_ERROR;                                         \
        }                                                                         \
    } while (0)

// ---- cudaDeviceProp jfieldIDs ----
static jclass  cudaDeviceProp_class = nullptr;

#define F(name) static jfieldID name = nullptr;
F(cudaDeviceProp_accessPolicyMaxWindowSize)
F(cudaDeviceProp_asyncEngineCount)
F(cudaDeviceProp_canMapHostMemory)
F(cudaDeviceProp_canUseHostPointerForRegisteredMem)
F(cudaDeviceProp_clockRate)
F(cudaDeviceProp_clusterLaunch)
F(cudaDeviceProp_computeMode)
F(cudaDeviceProp_computePreemptionSupported)
F(cudaDeviceProp_concurrentKernels)
F(cudaDeviceProp_concurrentManagedAccess)
F(cudaDeviceProp_cooperativeLaunch)
F(cudaDeviceProp_cooperativeMultiDeviceLaunch)
F(cudaDeviceProp_deferredMappingCudaArraySupported)
F(cudaDeviceProp_deviceOverlap)
F(cudaDeviceProp_directManagedMemAccessFromHost)
F(cudaDeviceProp_ECCEnabled)
F(cudaDeviceProp_globalL1CacheSupported)
F(cudaDeviceProp_gpuDirectRDMAFlushWritesOptions)
F(cudaDeviceProp_gpuDirectRDMASupported)
F(cudaDeviceProp_gpuDirectRDMAWritesOrdering)
F(cudaDeviceProp_hostNativeAtomicSupported)
F(cudaDeviceProp_hostRegisterReadOnlySupported)
F(cudaDeviceProp_hostRegisterSupported)
F(cudaDeviceProp_integrated)
F(cudaDeviceProp_ipcEventSupported)
F(cudaDeviceProp_isMultiGpuBoard)
F(cudaDeviceProp_kernelExecTimeoutEnabled)
F(cudaDeviceProp_l2CacheSize)
F(cudaDeviceProp_localL1CacheSupported)
F(cudaDeviceProp_luid)
F(cudaDeviceProp_luidDeviceNodeMask)
F(cudaDeviceProp_major)
F(cudaDeviceProp_managedMemory)
F(cudaDeviceProp_maxBlocksPerMultiProcessor)
F(cudaDeviceProp_maxGridSize)
F(cudaDeviceProp_maxSurface1D)
F(cudaDeviceProp_maxSurface1DLayered)
F(cudaDeviceProp_maxSurface2D)
F(cudaDeviceProp_maxSurface2DLayered)
F(cudaDeviceProp_maxSurface3D)
F(cudaDeviceProp_maxSurfaceCubemap)
F(cudaDeviceProp_maxSurfaceCubemapLayered)
F(cudaDeviceProp_maxTexture1D)
F(cudaDeviceProp_maxTexture1DLayered)
F(cudaDeviceProp_maxTexture1DLinear)
F(cudaDeviceProp_maxTexture1DMipmap)
F(cudaDeviceProp_maxTexture2D)
F(cudaDeviceProp_maxTexture2DGather)
F(cudaDeviceProp_maxTexture2DLayered)
F(cudaDeviceProp_maxTexture2DLinear)
F(cudaDeviceProp_maxTexture2DMipmap)
F(cudaDeviceProp_maxTexture3D)
F(cudaDeviceProp_maxTexture3DAlt)
F(cudaDeviceProp_maxTextureCubemap)
F(cudaDeviceProp_maxTextureCubemapLayered)
F(cudaDeviceProp_maxThreadsDim)
F(cudaDeviceProp_maxThreadsPerBlock)
F(cudaDeviceProp_maxThreadsPerMultiProcessor)
F(cudaDeviceProp_memoryBusWidth)
F(cudaDeviceProp_memoryClockRate)
F(cudaDeviceProp_memoryPoolsSupported)
F(cudaDeviceProp_memoryPoolSupportedHandleTypes)
F(cudaDeviceProp_memPitch)
F(cudaDeviceProp_minor)
F(cudaDeviceProp_multiGpuBoardGroupID)
F(cudaDeviceProp_multiProcessorCount)
F(cudaDeviceProp_name)
F(cudaDeviceProp_pageableMemoryAccess)
F(cudaDeviceProp_pageableMemoryAccessUsesHostPageTables)
F(cudaDeviceProp_pciBusID)
F(cudaDeviceProp_pciDeviceID)
F(cudaDeviceProp_pciDomainID)
F(cudaDeviceProp_persistingL2CacheMaxSize)
F(cudaDeviceProp_regsPerBlock)
F(cudaDeviceProp_regsPerMultiprocessor)
F(cudaDeviceProp_reserved)
F(cudaDeviceProp_reservedSharedMemPerBlock)
F(cudaDeviceProp_sharedMemPerBlock)
F(cudaDeviceProp_sharedMemPerBlockOptin)
F(cudaDeviceProp_sharedMemPerMultiprocessor)
F(cudaDeviceProp_singleToDoublePrecisionPerfRatio)
F(cudaDeviceProp_sparseCudaArraySupported)
F(cudaDeviceProp_streamPrioritiesSupported)
F(cudaDeviceProp_surfaceAlignment)
F(cudaDeviceProp_tccDriver)
F(cudaDeviceProp_textureAlignment)
F(cudaDeviceProp_texturePitchAlignment)
F(cudaDeviceProp_timelineSemaphoreInteropSupported)
F(cudaDeviceProp_totalConstMem)
F(cudaDeviceProp_totalGlobalMem)
F(cudaDeviceProp_unifiedAddressing)
F(cudaDeviceProp_unifiedFunctionPointers)
F(cudaDeviceProp_warpSize)
#undef F



JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    JNIEnv *env = nullptr;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4)) {
        return JNI_ERR;
    }

    // Only what we need so far
    if (initJNIUtils(env) == JNI_ERR)      return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR)  return JNI_ERR;

    // ---- cache all fields of org.apache.sysds.cujava.runtime.cudaDeviceProp ----
    {
        jclass cls = nullptr;
        if (!init(env, cls, "org/apache/sysds/cujava/runtime/CudaDeviceProp")) return JNI_ERR;
        cudaDeviceProp_class = (jclass)env->NewGlobalRef(cls);
        if (!cudaDeviceProp_class) return JNI_ERR;

        struct Spec { const char* name; const char* sig; jfieldID* out; } specs[] = {
            {"accessPolicyMaxWindowSize","I",&cudaDeviceProp_accessPolicyMaxWindowSize},
            {"asyncEngineCount","I",&cudaDeviceProp_asyncEngineCount},
            {"canMapHostMemory","I",&cudaDeviceProp_canMapHostMemory},
            {"canUseHostPointerForRegisteredMem","I",&cudaDeviceProp_canUseHostPointerForRegisteredMem},
            {"clockRate","I",&cudaDeviceProp_clockRate},
            {"clusterLaunch","I",&cudaDeviceProp_clusterLaunch},
            {"computeMode","I",&cudaDeviceProp_computeMode},
            {"computePreemptionSupported","I",&cudaDeviceProp_computePreemptionSupported},
            {"concurrentKernels","I",&cudaDeviceProp_concurrentKernels},
            {"concurrentManagedAccess","I",&cudaDeviceProp_concurrentManagedAccess},
            {"cooperativeLaunch","I",&cudaDeviceProp_cooperativeLaunch},
            {"cooperativeMultiDeviceLaunch","I",&cudaDeviceProp_cooperativeMultiDeviceLaunch},
            {"deferredMappingCudaArraySupported","I",&cudaDeviceProp_deferredMappingCudaArraySupported},
            {"deviceOverlap","I",&cudaDeviceProp_deviceOverlap},
            {"directManagedMemAccessFromHost","I",&cudaDeviceProp_directManagedMemAccessFromHost},
            {"ECCEnabled","I",&cudaDeviceProp_ECCEnabled},
            {"globalL1CacheSupported","I",&cudaDeviceProp_globalL1CacheSupported},
            {"gpuDirectRDMAFlushWritesOptions","I",&cudaDeviceProp_gpuDirectRDMAFlushWritesOptions},
            {"gpuDirectRDMASupported","I",&cudaDeviceProp_gpuDirectRDMASupported},
            {"gpuDirectRDMAWritesOrdering","I",&cudaDeviceProp_gpuDirectRDMAWritesOrdering},
            {"hostNativeAtomicSupported","I",&cudaDeviceProp_hostNativeAtomicSupported},
            {"hostRegisterReadOnlySupported","I",&cudaDeviceProp_hostRegisterReadOnlySupported},
            {"hostRegisterSupported","I",&cudaDeviceProp_hostRegisterSupported},
            {"integrated","I",&cudaDeviceProp_integrated},
            {"ipcEventSupported","I",&cudaDeviceProp_ipcEventSupported},
            {"isMultiGpuBoard","I",&cudaDeviceProp_isMultiGpuBoard},
            {"kernelExecTimeoutEnabled","I",&cudaDeviceProp_kernelExecTimeoutEnabled},
            {"l2CacheSize","I",&cudaDeviceProp_l2CacheSize},
            {"localL1CacheSupported","I",&cudaDeviceProp_localL1CacheSupported},
            {"luid","[B",&cudaDeviceProp_luid},
            {"luidDeviceNodeMask","I",&cudaDeviceProp_luidDeviceNodeMask},
            {"major","I",&cudaDeviceProp_major},
            {"managedMemory","I",&cudaDeviceProp_managedMemory},
            {"maxBlocksPerMultiProcessor","I",&cudaDeviceProp_maxBlocksPerMultiProcessor},
            {"maxGridSize","[I",&cudaDeviceProp_maxGridSize},
            {"maxSurface1D","I",&cudaDeviceProp_maxSurface1D},
            {"maxSurface1DLayered","[I",&cudaDeviceProp_maxSurface1DLayered},
            {"maxSurface2D","[I",&cudaDeviceProp_maxSurface2D},
            {"maxSurface2DLayered","[I",&cudaDeviceProp_maxSurface2DLayered},
            {"maxSurface3D","[I",&cudaDeviceProp_maxSurface3D},
            {"maxSurfaceCubemap","I",&cudaDeviceProp_maxSurfaceCubemap},
            {"maxSurfaceCubemapLayered","[I",&cudaDeviceProp_maxSurfaceCubemapLayered},
            {"maxTexture1D","I",&cudaDeviceProp_maxTexture1D},
            {"maxTexture1DLayered","[I",&cudaDeviceProp_maxTexture1DLayered},
            {"maxTexture1DLinear","I",&cudaDeviceProp_maxTexture1DLinear},
            {"maxTexture1DMipmap","I",&cudaDeviceProp_maxTexture1DMipmap},
            {"maxTexture2D","[I",&cudaDeviceProp_maxTexture2D},
            {"maxTexture2DGather","[I",&cudaDeviceProp_maxTexture2DGather},
            {"maxTexture2DLayered","[I",&cudaDeviceProp_maxTexture2DLayered},
            {"maxTexture2DLinear","[I",&cudaDeviceProp_maxTexture2DLinear},
            {"maxTexture2DMipmap","[I",&cudaDeviceProp_maxTexture2DMipmap},
            {"maxTexture3D","[I",&cudaDeviceProp_maxTexture3D},
            {"maxTexture3DAlt","[I",&cudaDeviceProp_maxTexture3DAlt},
            {"maxTextureCubemap","I",&cudaDeviceProp_maxTextureCubemap},
            {"maxTextureCubemapLayered","[I",&cudaDeviceProp_maxTextureCubemapLayered},
            {"maxThreadsDim","[I",&cudaDeviceProp_maxThreadsDim},
            {"maxThreadsPerBlock","I",&cudaDeviceProp_maxThreadsPerBlock},
            {"maxThreadsPerMultiProcessor","I",&cudaDeviceProp_maxThreadsPerMultiProcessor},
            {"memoryBusWidth","I",&cudaDeviceProp_memoryBusWidth},
            {"memoryClockRate","I",&cudaDeviceProp_memoryClockRate},
            {"memoryPoolsSupported","I",&cudaDeviceProp_memoryPoolsSupported},
            {"memoryPoolSupportedHandleTypes","I",&cudaDeviceProp_memoryPoolSupportedHandleTypes},
            {"memPitch","J",&cudaDeviceProp_memPitch},
            {"minor","I",&cudaDeviceProp_minor},
            {"multiGpuBoardGroupID","I",&cudaDeviceProp_multiGpuBoardGroupID},
            {"multiProcessorCount","I",&cudaDeviceProp_multiProcessorCount},
            {"name","[B",&cudaDeviceProp_name},
            {"pageableMemoryAccess","I",&cudaDeviceProp_pageableMemoryAccess},
            {"pageableMemoryAccessUsesHostPageTables","I",&cudaDeviceProp_pageableMemoryAccessUsesHostPageTables},
            {"pciBusID","I",&cudaDeviceProp_pciBusID},
            {"pciDeviceID","I",&cudaDeviceProp_pciDeviceID},
            {"pciDomainID","I",&cudaDeviceProp_pciDomainID},
            {"persistingL2CacheMaxSize","I",&cudaDeviceProp_persistingL2CacheMaxSize},
            {"regsPerBlock","I",&cudaDeviceProp_regsPerBlock},
            {"regsPerMultiprocessor","I",&cudaDeviceProp_regsPerMultiprocessor},
            {"reserved","I",&cudaDeviceProp_reserved},
            {"reservedSharedMemPerBlock","J",&cudaDeviceProp_reservedSharedMemPerBlock},
            {"sharedMemPerBlock","J",&cudaDeviceProp_sharedMemPerBlock},
            {"sharedMemPerBlockOptin","J",&cudaDeviceProp_sharedMemPerBlockOptin},
            {"sharedMemPerMultiprocessor","J",&cudaDeviceProp_sharedMemPerMultiprocessor},
            {"singleToDoublePrecisionPerfRatio","I",&cudaDeviceProp_singleToDoublePrecisionPerfRatio},
            {"sparseCudaArraySupported","I",&cudaDeviceProp_sparseCudaArraySupported},
            {"streamPrioritiesSupported","I",&cudaDeviceProp_streamPrioritiesSupported},
            {"surfaceAlignment","J",&cudaDeviceProp_surfaceAlignment},
            {"tccDriver","I",&cudaDeviceProp_tccDriver},
            {"textureAlignment","J",&cudaDeviceProp_textureAlignment},
            {"texturePitchAlignment","J",&cudaDeviceProp_texturePitchAlignment},
            {"timelineSemaphoreInteropSupported","I",&cudaDeviceProp_timelineSemaphoreInteropSupported},
            {"totalConstMem","J",&cudaDeviceProp_totalConstMem},
            {"totalGlobalMem","J",&cudaDeviceProp_totalGlobalMem},
            {"unifiedAddressing","I",&cudaDeviceProp_unifiedAddressing},
            {"unifiedFunctionPointers","I",&cudaDeviceProp_unifiedFunctionPointers},
            {"warpSize","I",&cudaDeviceProp_warpSize},
        };

        for (const auto& s : specs) {
            if (!init(env, cls, *s.out, s.name, s.sig)) return JNI_ERR;
        }
    }


    return JNI_VERSION_1_4;
}


JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
}

static void setCudaDeviceProp(JNIEnv* env, jobject prop, const cudaDeviceProp& p) {
    // byte[256] name + byte[8] luid (luid undefined on non-Windows -> zero it)
    setFieldBytes(env, prop, cudaDeviceProp_name,
                  reinterpret_cast<const jbyte*>(p.name), 256);
    { jbyte z8[8] = {0}; setFieldBytes(env, prop, cudaDeviceProp_luid, z8, 8); }

    // int[] fields
    { jint v[3] = { (jint)p.maxThreadsDim[0], (jint)p.maxThreadsDim[1], (jint)p.maxThreadsDim[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxThreadsDim, v, 3); }

    { jint v[3] = { (jint)p.maxGridSize[0], (jint)p.maxGridSize[1], (jint)p.maxGridSize[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxGridSize, v, 3); }

    { jint v[2] = { (jint)p.maxTexture2D[0], (jint)p.maxTexture2D[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture2D, v, 2); }

    { jint v[2] = { (jint)p.maxTexture2DGather[0], (jint)p.maxTexture2DGather[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture2DGather, v, 2); }

    { jint v[3] = { (jint)p.maxTexture2DLinear[0], (jint)p.maxTexture2DLinear[1], (jint)p.maxTexture2DLinear[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture2DLinear, v, 3); }

    { jint v[2] = { (jint)p.maxTexture2DMipmap[0], (jint)p.maxTexture2DMipmap[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture2DMipmap, v, 2); }

    { jint v[3] = { (jint)p.maxTexture3D[0], (jint)p.maxTexture3D[1], (jint)p.maxTexture3D[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture3D, v, 3); }

    { jint v[3] = { (jint)p.maxTexture3DAlt[0], (jint)p.maxTexture3DAlt[1], (jint)p.maxTexture3DAlt[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture3DAlt, v, 3); }

    { jint v[2] = { (jint)p.maxTexture1DLayered[0], (jint)p.maxTexture1DLayered[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture1DLayered, v, 2); }

    { jint v[3] = { (jint)p.maxTexture2DLayered[0], (jint)p.maxTexture2DLayered[1], (jint)p.maxTexture2DLayered[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxTexture2DLayered, v, 3); }

    { jint v[2] = { (jint)p.maxTextureCubemapLayered[0], (jint)p.maxTextureCubemapLayered[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxTextureCubemapLayered, v, 2); }

    { jint v[2] = { (jint)p.maxSurface1DLayered[0], (jint)p.maxSurface1DLayered[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxSurface1DLayered, v, 2); }

    { jint v[2] = { (jint)p.maxSurface2D[0], (jint)p.maxSurface2D[1] };
      setFieldInts(env, prop, cudaDeviceProp_maxSurface2D, v, 2); }

    { jint v[3] = { (jint)p.maxSurface2DLayered[0], (jint)p.maxSurface2DLayered[1], (jint)p.maxSurface2DLayered[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxSurface2DLayered, v, 3); }

    { jint v[3] = { (jint)p.maxSurface3D[0], (jint)p.maxSurface3D[1], (jint)p.maxSurface3D[2] };
      setFieldInts(env, prop, cudaDeviceProp_maxSurface3D, v, 3); }

    // long fields
    env->SetLongField(prop, cudaDeviceProp_totalGlobalMem,           (jlong)p.totalGlobalMem);
    env->SetLongField(prop, cudaDeviceProp_totalConstMem,            (jlong)p.totalConstMem);
    env->SetLongField(prop, cudaDeviceProp_sharedMemPerBlock,        (jlong)p.sharedMemPerBlock);
    env->SetLongField(prop, cudaDeviceProp_sharedMemPerMultiprocessor,(jlong)p.sharedMemPerMultiprocessor);
    env->SetLongField(prop, cudaDeviceProp_reservedSharedMemPerBlock,(jlong)p.reservedSharedMemPerBlock);
    env->SetLongField(prop, cudaDeviceProp_sharedMemPerBlockOptin,   (jlong)p.sharedMemPerBlockOptin);
    env->SetLongField(prop, cudaDeviceProp_memPitch,                 (jlong)p.memPitch);
    env->SetLongField(prop, cudaDeviceProp_surfaceAlignment,         (jlong)p.surfaceAlignment);
    env->SetLongField(prop, cudaDeviceProp_textureAlignment,         (jlong)p.textureAlignment);
    env->SetLongField(prop, cudaDeviceProp_texturePitchAlignment,    (jlong)p.texturePitchAlignment);

    // int fields (available in cudaDeviceProp)
    env->SetIntField(prop,  cudaDeviceProp_regsPerBlock,             (jint)p.regsPerBlock);
    env->SetIntField(prop,  cudaDeviceProp_regsPerMultiprocessor,    (jint)p.regsPerMultiprocessor);
    env->SetIntField(prop,  cudaDeviceProp_warpSize,                 (jint)p.warpSize);
    env->SetIntField(prop,  cudaDeviceProp_maxThreadsPerBlock,       (jint)p.maxThreadsPerBlock);
    env->SetIntField(prop,  cudaDeviceProp_maxThreadsPerMultiProcessor,(jint)p.maxThreadsPerMultiProcessor);
    env->SetIntField(prop,  cudaDeviceProp_clockRate,                (jint)p.clockRate);
    env->SetIntField(prop,  cudaDeviceProp_memoryClockRate,          (jint)p.memoryClockRate);
    env->SetIntField(prop,  cudaDeviceProp_memoryBusWidth,           (jint)p.memoryBusWidth);
    env->SetIntField(prop,  cudaDeviceProp_l2CacheSize,              (jint)p.l2CacheSize);
    env->SetIntField(prop,  cudaDeviceProp_major,                    (jint)p.major);
    env->SetIntField(prop,  cudaDeviceProp_minor,                    (jint)p.minor);
    env->SetIntField(prop,  cudaDeviceProp_multiProcessorCount,      (jint)p.multiProcessorCount);
    env->SetIntField(prop,  cudaDeviceProp_deviceOverlap,            (jint)p.deviceOverlap);
    env->SetIntField(prop,  cudaDeviceProp_kernelExecTimeoutEnabled, (jint)p.kernelExecTimeoutEnabled);
    env->SetIntField(prop,  cudaDeviceProp_integrated,               (jint)p.integrated);
    env->SetIntField(prop,  cudaDeviceProp_canMapHostMemory,         (jint)p.canMapHostMemory);
    env->SetIntField(prop,  cudaDeviceProp_computeMode,              (jint)p.computeMode);
    env->SetIntField(prop,  cudaDeviceProp_maxTexture1D,             (jint)p.maxTexture1D);
    env->SetIntField(prop,  cudaDeviceProp_maxTexture1DMipmap,       (jint)p.maxTexture1DMipmap);
    env->SetIntField(prop,  cudaDeviceProp_maxTexture1DLinear,       (jint)p.maxTexture1DLinear);
    env->SetIntField(prop,  cudaDeviceProp_maxTextureCubemap,        (jint)p.maxTextureCubemap);
    env->SetIntField(prop,  cudaDeviceProp_maxSurface1D,             (jint)p.maxSurface1D);
    env->SetIntField(prop,  cudaDeviceProp_maxSurfaceCubemap,        (jint)p.maxSurfaceCubemap);
    env->SetIntField(prop,  cudaDeviceProp_asyncEngineCount,         (jint)p.asyncEngineCount);
    env->SetIntField(prop,  cudaDeviceProp_concurrentKernels,        (jint)p.concurrentKernels);
    env->SetIntField(prop,  cudaDeviceProp_ECCEnabled,               (jint)p.ECCEnabled);
    env->SetIntField(prop,  cudaDeviceProp_pciBusID,                 (jint)p.pciBusID);
    env->SetIntField(prop,  cudaDeviceProp_pciDeviceID,              (jint)p.pciDeviceID);
    env->SetIntField(prop,  cudaDeviceProp_pciDomainID,              (jint)p.pciDomainID);
    env->SetIntField(prop,  cudaDeviceProp_unifiedAddressing,        (jint)p.unifiedAddressing);

}



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemcpyNative
  (JNIEnv *env, jclass cls, jobject dst, jobject src, jlong count, jint kind) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, dst, "dst", "cudaMemcpy");
    CUJAVA_REQUIRE_NONNULL(env, src, "src", "cudaMemcpy");

    Logger::log(LOG_TRACE, "Executing cudaMemcpy of %ld bytes\n", (long)count);

    // Obtain the destination and source pointers
    PointerData *dstPointerData = initPointerData(env, dst);
    if (dstPointerData == nullptr) {
        return CUJAVA_INTERNAL_ERROR;
    }
    PointerData *srcPointerData = initPointerData(env, src);
    if (srcPointerData == nullptr) {
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

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, devPtr, "devPtr", "cudaMalloc");

    Logger::log(LOG_TRACE, "Executing cudaMalloc of %ld bytes\n", (long)size);

    void *nativeDevPtr = nullptr;
    int result = cudaMalloc(&nativeDevPtr, (size_t)size);
    setPointer(env, devPtr, (jlong)nativeDevPtr);

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaFreeNative
  (JNIEnv *env, jclass cls, jobject devPtr) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, devPtr, "devPtr", "cudaFree");

    Logger::log(LOG_TRACE, "Executing cudaFree\n");

    void *nativeDevPtr = nullptr;
    nativeDevPtr = getPointer(env, devPtr);
    int result = cudaFree(nativeDevPtr);
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemsetNative
  (JNIEnv *env, jclass cls, jobject mem, jint c, jlong count) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, mem, "mem", "cudaMemset");

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

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, devPtr, "devPtr", "cudaMallocManaged");

    Logger::log(LOG_TRACE, "Executing cudaMallocManaged of %ld bytes\n", (long)size);

    void *nativeDevPtr = nullptr;
    int result = cudaMallocManaged(&nativeDevPtr, (size_t)size, (unsigned int)flags);
    if (result == cudaSuccess) {
        if (flags == cudaMemAttachHost) {
            jobject object = env->NewDirectByteBuffer(nativeDevPtr, size);
            env->SetObjectField(devPtr, Pointer_buffer, object);
            env->SetObjectField(devPtr, Pointer_pointers, nullptr);
            env->SetLongField(devPtr, Pointer_byteOffset, 0);
        }
        env->SetLongField(devPtr, NativePointerObject_nativePointer, (jlong)nativeDevPtr);
    }

    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemGetInfoNative
  (JNIEnv *env, jclass cls, jlongArray freeBytes, jlongArray totalBytes) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, freeBytes, "freeBytes", "cudaMemGetInfo");
    CUJAVA_REQUIRE_NONNULL(env, totalBytes, "totalBytes", "cudaMemGetInfo");

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

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, count, "count", "cudaGetDeviceCount");

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

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, device, "device", "cudaGetDevice");

    Logger::log(LOG_TRACE, "Executing cudaGetDevice\n");

    int nativeDevice = 0;
    int result = cudaGetDevice(&nativeDevice);
    if (!set(env, device, 0, nativeDevice)) return CUJAVA_INTERNAL_ERROR;
    return result;
}


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaGetDevicePropertiesNative
  (JNIEnv *env, jclass cls, jobject prop, jint device) {

    // Validate: all jobject parameters must be non-null
    CUJAVA_REQUIRE_NONNULL(env, prop, "prop", "cudaGetDeviceProperties");

    Logger::log(LOG_TRACE, "Executing cudaGetDeviceProperties\n");

    cudaDeviceProp nativeProp;
    int result = cudaGetDeviceProperties(&nativeProp, device);

    setCudaDeviceProp(env, prop, nativeProp);
    return result;
}




