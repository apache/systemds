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

/* DO EDIT THIS FILE - it is not machine generated */

#pragma once
#ifndef JNI_BRIDGE_H
#define JNI_BRIDGE_H

#include <jni.h>
/* Header for class org_apache_sysds_hops_codegen_SpoofCompiler */

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_apache_sysds_hops_codegen_SpoofCompiler
 * Method:    initialize_cuda_context
 * Signature: (ILjava/lang/String;)J
 */
[[maybe_unused]] JNIEXPORT jlong JNICALL Java_org_apache_sysds_hops_codegen_SpoofCompiler_initialize_1cuda_1context
	(JNIEnv *, [[maybe_unused]] jobject, jint, jstring);

/*
 * Class:     org_apache_sysds_hops_codegen_SpoofCompiler
 * Method:    destroy_cuda_context
 * Signature: (JI)V
 */
[[maybe_unused]] JNIEXPORT void JNICALL Java_org_apache_sysds_hops_codegen_SpoofCompiler_destroy_1cuda_1context
	([[maybe_unused]] JNIEnv *, [[maybe_unused]] jobject, jlong, jint);

/*
 * Class:     org_apache_sysds_runtime_codegen_SpoofOperator
 * Method:    getNativeStagingBuffer
 * Signature: (Ljcuda/Pointer;JI)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofOperator_getNativeStagingBuffer
	(JNIEnv *, jclass, jobject, jlong, jint);

/*
 * Class:     org_apache_sysds_hops_codegen_cplan_CNodeCell
 * Method:    compile_nvrtc
 * Signature: (JLjava/lang/String;Ljava/lang/String;IIZ)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_hops_codegen_cplan_CNodeCell_compile_1nvrtc
		(JNIEnv *, [[maybe_unused]] jobject, jlong, jstring, jstring, jint, jint, jboolean);

/*
 * Class:     org_apache_sysds_hops_codegen_cplan_CNodeRow
 * Method:    compile_nvrtc
 * Signature: (JLjava/lang/String;Ljava/lang/String;IIIZ)I
 */
[[maybe_unused]] [[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_hops_codegen_cplan_CNodeRow_compile_1nvrtc
		(JNIEnv *, [[maybe_unused]] jobject, jlong, jstring, jstring, jint, jint, jint, jboolean);

/*
 * Class:     org_apache_sysds_runtime_codegen_SpoofCUDACellwise
 * Method:    execute_d
 * Signature: (J)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1d
	(JNIEnv *, jclass, jlong);

/*
 * Class:     org_apache_sysds_runtime_codegen_SpoofCUDACellwise
 * Method:    execute_f
 * Signature: (J)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1f
	(JNIEnv *, jclass, jlong);

/*
 * Class:     org_apache_sysds_runtime_codegen_SpoofCUDARowwise
 * Method:    execute_d
 * Signature: (J)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1d
	(JNIEnv *, jclass, jlong);

/*
 * Class:     org_apache_sysds_runtime_codegen_SpoofCUDARowwise
 * Method:    execute_f
 * Signature: (J)I
 */
[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1f
	(JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif

#endif // JNI_BRIDGE_H
