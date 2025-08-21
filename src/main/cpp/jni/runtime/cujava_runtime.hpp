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

#ifndef _Included_org_apache_sysds_cujava_runtime_CuJava
#define _Included_org_apache_sysds_cujava_runtime_CuJava
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org.apache.sysds.cujava.runtime.CuJava
 * Methods:
 *  - cudaMemcpyNative
 *  - cudaMallocNative
 *  - cudaFreeNative
 *  - cudaMemsetnative
 *  - cudaDeviceSynchronizeNative
 *  - cudaMallocManagedNative
 *  - cudaMemGetInfoNative
 */



JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemcpyNative
  (JNIEnv *, jclass, jobject, jobject, jlong, jint);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMallocNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaFreeNative
  (JNIEnv *env, jclass cls, jobject devPtr);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemsetNative
  (JNIEnv *env, jclass cls, jobject mem, jint c, jlong count);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaDeviceSynchronizeNative
  (JNIEnv *env, jclass cls);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMallocManagedNative
  (JNIEnv *env, jclass cls, jobject devPtr, jlong size, jint flags);

JNIEXPORT jint JNICALL Java_org_apache_sysds_cujava_runtime_CuJava_cudaMemGetInfoNative
  (JNIEnv *env, jclass cls, jlongArray freeBytes, jlongArray totalBytes);

#ifdef __cplusplus
}
#endif
#endif
