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

#include "jni_bridge.h"
#include "SpoofCUDAContext.h"
#include "Matrix.h"

// JNI Methods to get/release arrays
#define GET_ARRAY(env, input)                                                  \
  ((void *)env->GetPrimitiveArrayCritical(input, nullptr))

#define RELEASE_ARRAY(env, java, cpp)                                                  \
  (env->ReleasePrimitiveArrayCritical(java, cpp, 0))

JNIEXPORT jlong JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_initialize_1cuda_1context(
    JNIEnv *env, jobject jobj, jint device_id, jstring resource_path) {

  const char *cstr_rp = env->GetStringUTFChars(resource_path, NULL);
  size_t ctx = SpoofCUDAContext::initialize_cuda(device_id, cstr_rp);
  env->ReleaseStringUTFChars(resource_path, cstr_rp);
  return ctx;
}

JNIEXPORT void JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_destroy_1cuda_1context(
    JNIEnv *env, jobject jobj, jlong ctx, jint device_id) {
  SpoofCUDAContext::destroy_cuda(reinterpret_cast<SpoofCUDAContext *>(ctx),
                                 device_id);
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_compile_1cuda_1kernel(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jstring src) {
  SpoofCUDAContext *ctx_ = reinterpret_cast<SpoofCUDAContext *>(ctx);
  const char *cstr_name = env->GetStringUTFChars(name, NULL);
  const char *cstr_src = env->GetStringUTFChars(src, NULL);
  bool result = ctx_->compile_cuda(cstr_src, cstr_name);
  env->ReleaseStringUTFChars(src, cstr_src);
  env->ReleaseStringUTFChars(name, cstr_name);
  return result;
}

JNIEXPORT jdouble JNICALL
Java_org_apache_sysds_runtime_codegen_SpoofCUDA_execute_1d(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jlongArray in_ptrs, jint input_offset, jlongArray side_ptrs, jlongArray out_ptrs,
			jdoubleArray scalars_, jlong m, jlong n, jlong out_len, jlong grix, jobject inputs_, jobject out_obj) {
	
	SpoofCUDAContext *ctx_ = reinterpret_cast<SpoofCUDAContext *>(ctx);
	const char *cstr_name = env->GetStringUTFChars(name, NULL);
	
	size_t* inputs = reinterpret_cast<size_t*>(GET_ARRAY(env, in_ptrs));
	size_t* sides = reinterpret_cast<size_t*>(GET_ARRAY(env, side_ptrs));
	size_t *output = reinterpret_cast<size_t*>(GET_ARRAY(env, out_ptrs));
	double *scalars = reinterpret_cast<double*>(GET_ARRAY(env, scalars_));
	
	//ToDo: call once while init
	jclass CacheableData = env->FindClass("org/apache/sysds/runtime/controlprogram/caching/CacheableData");
	if(!CacheableData) {
	  	std::cerr << " JNIEnv -> FindClass(CacheableData) failed" << std::endl;
	  	return -1.0;
	}
	jclass ArrayList = env->FindClass("java/util/ArrayList");
	if(!ArrayList) {
		std::cerr << " JNIEnv -> FindClass(ArrayList) failed" << std::endl;
		return -1.0;
	}
	jmethodID mat_obj_num_rows = env->GetMethodID(CacheableData, "getNumRows", "()J");
	if(!mat_obj_num_rows) {
		std::cerr << " JNIEnv -> GetMethodID() failed" << std::endl;
		return -1.0;
	}
	jmethodID mat_obj_num_cols = env->GetMethodID(CacheableData, "getNumColumns", "()J");
	if(!mat_obj_num_cols) {
		std::cerr << " JNIEnv -> GetMethodID() failed" << std::endl;
		return -1.0;
	}
	jmethodID ArrayList_size = env->GetMethodID(ArrayList, "size", "()I");
	jmethodID ArrayList_get = env->GetMethodID(ArrayList, "get", "(I)Ljava/lang/Object;");

	std::vector<Matrix<double>> in;
	jint num_inputs = env->CallIntMethod(inputs_, ArrayList_size);
	std::cout << "num inputs: " << num_inputs << " offsets: " << input_offset << std::endl;
;
	for(auto ptr_idx = 0, input_idx = 0; input_idx < input_offset; ptr_idx+=4, input_idx++) {
		jobject input_obj = env->CallObjectMethod(inputs_, ArrayList_get, input_idx);
		uint32_t m = static_cast<uint32_t>(env->CallIntMethod(input_obj, mat_obj_num_rows));
		uint32_t n = static_cast<uint32_t>(env->CallIntMethod(input_obj, mat_obj_num_cols));

		in.push_back(Matrix<double>{reinterpret_cast<double*>(inputs[ptr_idx+3]), reinterpret_cast<uint32_t*>(inputs[ptr_idx+1]),
										   reinterpret_cast<uint32_t*>(inputs[ptr_idx+2]), m, n,
										   static_cast<uint32_t>(inputs[ptr_idx])});
		std::cout << "input #" << input_idx << " m=" << m << " n=" << n << std::endl;
	}

	std::vector<Matrix<double>> side_inputs;
	for(auto ptr_idx = 0, input_idx = input_offset; input_idx < num_inputs; ptr_idx+=4, input_idx++) {
		jobject side_input_obj = env->CallObjectMethod(inputs_, ArrayList_get, input_idx);
		uint32_t m = static_cast<uint32_t>(env->CallIntMethod(side_input_obj, mat_obj_num_rows));
		uint32_t n = static_cast<uint32_t>(env->CallIntMethod(side_input_obj, mat_obj_num_cols));

//		std::cout << "sides["<<i << "]=" <<  sides[i] << std::endl;
//		std::cout << "sides["<<i+1 << "]=" <<  sides[i+1] << std::endl;
//		std::cout << "sides["<<i+2 << "]=" <<  sides[i+2] << std::endl;
//		std::cout << "sides["<<i+3 << "]=" <<  sides[i+3] << std::endl;


		side_inputs.push_back(Matrix<double>{reinterpret_cast<double*>(sides[ptr_idx+3]), reinterpret_cast<uint32_t*>(sides[ptr_idx+1]),
									reinterpret_cast<uint32_t*>(sides[ptr_idx+2]), m, n,
									static_cast<uint32_t>(sides[ptr_idx])});
		
		std::cout << "side input #" << input_idx << " m=" << m << " n=" << n << std::endl;
	}

	std::unique_ptr<Matrix<double>> out;
	if(out_obj != nullptr) {
//		std::cout << "out not null" << std::endl;
		out = std::make_unique<Matrix<double>>(Matrix<double>{reinterpret_cast<double*>(output[3]),
														reinterpret_cast<uint32_t*>(output[1]),
															reinterpret_cast<uint32_t*>(output[2]),
															  static_cast<uint32_t>(env->CallIntMethod(out_obj, mat_obj_num_rows)),
															  static_cast<uint32_t>(env->CallIntMethod(out_obj, mat_obj_num_cols)),
															  static_cast<uint32_t>(output[0])});
	}

	double result = ctx_->execute_kernel(cstr_name, in, side_inputs, out.get(), scalars,
									  env->GetArrayLength(scalars_), m, n, out_len, grix);

	RELEASE_ARRAY(env, in_ptrs, inputs);
	RELEASE_ARRAY(env, side_ptrs, sides);
	RELEASE_ARRAY(env, out_ptrs, output);
	RELEASE_ARRAY(env, scalars_, scalars);

	// FIXME: that release causes an error
	//std::cout << "releasing " << name_ << std::endl;
	env->ReleaseStringUTFChars(name, cstr_name);
	return result;
}

//JNIEXPORT jfloat JNICALL
//Java_org_apache_sysds_runtime_codegen_SpoofCUDA_execute_1f(
//    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jlongArray in_ptrs,
//    jlongArray side_ptrs, jlong out_ptr, jfloatArray scalars_, jlong m, jlong n, jlong out_len, jlong grix) {
//
//  SpoofCUDAContext *ctx_ = reinterpret_cast<SpoofCUDAContext *>(ctx);
//
//  const char *cstr_name = env->GetStringUTFChars(name, NULL);
//
//  float **inputs = reinterpret_cast<float**>(GET_ARRAY(env, in_ptrs));
//  float **sides = reinterpret_cast<float **>(GET_ARRAY(env, side_ptrs));
//  float *scalars = reinterpret_cast<float *>(GET_ARRAY(env, scalars_));
//
//  float result = ctx_->execute_kernel(
//      cstr_name, inputs, env->GetArrayLength(in_ptrs), sides, env->GetArrayLength(side_ptrs),
//      reinterpret_cast<float *>(out_ptr), scalars, env->GetArrayLength(scalars_), m, n, out_len, grix);
//
//  RELEASE_ARRAY(env, in_ptrs, inputs);
//  RELEASE_ARRAY(env, side_ptrs, sides);
//  RELEASE_ARRAY(env, scalars_, scalars);
//
//  // FIXME: that release causes an error
//  env->ReleaseStringUTFChars(name, cstr_name);
//  return result;
//}
