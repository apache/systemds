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

 #ifndef _Included_org_apache_sysds_cujava_driver_CuJavaDriver
 #define _Included_org_apache_sysds_cujava_driver_CuJavaDriver
 #ifdef __cplusplus
 extern "C" {
 #endif

 /*
  * Class:  org.apache.sysds.cujava.driver.CuJavaDriver
  * Methods:
  *  - cudaCtxCreate
  *  - cuDeviceGet
  *  - cuDeviceGetCount
  *  - cuInit
  *  - cuLaunchKernel
  *  - cuModuleGetFunction
  *  - cuModuleLoadDataEx
  *  - cuMemAlloc
  *  - cuModuleUnload
  *  - cuCtxDestroy
  *  - cuMemFree
  *  - cuMemcpyDtoH
  *  - cuCtxSynchronize
  *  - cuDeviceGetAttribute
  */


JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxCreateNative
  (JNIEnv *env, jclass cls, jobject pctx, jint flags, jobject dev);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetNative
  (JNIEnv *env, jclass cls, jobject device, jint ordinal);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetCountNative
  (JNIEnv *env, jclass cls, jintArray count);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuInitNative
  (JNIEnv *env, jclass cls, jint Flags);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuLaunchKernelNative
  (JNIEnv *env, jclass, jobject f, jint gridDimX, jint gridDimY, jint gridDimZ,
   jint blockDimX, jint blockDimY, jint blockDimZ, jint sharedMemBytes,
   jobject hStream, jobject kernelParams, jobject extra);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleGetFunctionNative
  (JNIEnv *env, jclass, jobject hfunc, jobject hmod, jstring name);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleLoadDataExNative
  (JNIEnv *env, jclass, jobject phMod, jobject p, jint numOptions, jintArray options, jobject optionValues);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemAllocNative
  (JNIEnv *env, jclass cls, jobject dptr, jlong bytesize);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuModuleUnloadNative
  (JNIEnv *env, jclass cls, jobject hmod);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxDestroyNative
  (JNIEnv *env, jclass cls, jobject ctx);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemFreeNative
  (JNIEnv *env, jclass cls, jobject dptr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuMemcpyDtoHNative
  (JNIEnv *env, jclass cls, jobject dstHost, jobject srcDevice, jlong ByteCount);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuCtxSynchronizeNative
  (JNIEnv *env, jclass cls);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_driver_CuJavaDriver_cuDeviceGetAttributeNative
  (JNIEnv *env, jclass cls, jintArray pi, jint CUdevice_attribute_attrib, jobject dev);

#ifdef __cplusplus
}
#endif
#endif
