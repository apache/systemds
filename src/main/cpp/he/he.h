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
/* Header for class org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper */

#ifndef _Included_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper
#define _Included_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    initClient
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_initClient
  (JNIEnv *, jclass, jbyteArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    generatePartialPublicKey
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_generatePartialPublicKey
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    setPublicKey
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_setPublicKey
  (JNIEnv *, jclass, jlong, jbyteArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    encrypt
 * Signature: (J[D)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_encrypt
  (JNIEnv *, jclass, jlong, jdoubleArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    partiallyDecrypt
 * Signature: (J[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_partiallyDecrypt
  (JNIEnv *, jclass, jlong, jbyteArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    initServer
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_initServer
  (JNIEnv *, jclass);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    generateA
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_generateA
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    aggregatePartialPublicKeys
 * Signature: (J[[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_aggregatePartialPublicKeys
  (JNIEnv *, jclass, jlong, jobjectArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    accumulateCiphertexts
 * Signature: (J[[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_accumulateCiphertexts
  (JNIEnv *, jclass, jlong, jobjectArray);

/*
 * Class:     org_apache_sysds_utils_NativeHelper
 * Method:    average
 * Signature: (J[B[[B)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_runtime_controlprogram_paramserv_NativeHEHelper_average
  (JNIEnv *, jclass, jlong, jbyteArray, jobjectArray);

#ifdef __cplusplus
}
#endif
#endif