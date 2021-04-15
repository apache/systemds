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
#include "SpoofCellwise.h"
#include "SpoofRowwise.h"

// JNI Methods to get/release arrays
//#define GET_ARRAY(env, input)((void *)env->GetPrimitiveArrayCritical(input, nullptr))
//#define RELEASE_ARRAY(env, java, cpp)(env->ReleasePrimitiveArrayCritical(java, cpp, 0))

jclass jcuda_pointer_class;
jclass jcuda_native_pointer_class;
jfieldID pointer_buffer_field;
jfieldID native_pointer_field;

// error output helper
void printException(const std::string &name, const std::exception &e, bool compile = false) {
	std::string type = compile ? "compiling" : "executing";
	std::cerr << "std::exception while " << type << "  SPOOF CUDA operator " << name << ":\n" << e.what() << std::endl;
}


[[maybe_unused]] JNIEXPORT jlong JNICALL Java_org_apache_sysds_hops_codegen_SpoofCompiler_initialize_1cuda_1context
	(JNIEnv *jenv, [[maybe_unused]] jobject jobj, jint device_id, jstring resource_path)
{
	const char *cstr_rp = jenv->GetStringUTFChars(resource_path, nullptr);
	size_t ctx = SpoofCUDAContext::initialize_cuda(device_id, cstr_rp);
	jenv->ReleaseStringUTFChars(resource_path, cstr_rp);

	// fetch some jcuda class handles
	jcuda_pointer_class = jenv->FindClass("jcuda/Pointer");
	jcuda_native_pointer_class = jenv->FindClass("jcuda/NativePointerObject");
	pointer_buffer_field = jenv->GetFieldID(jcuda_pointer_class, "buffer", "Ljava/nio/Buffer;");
	native_pointer_field = jenv->GetFieldID(jcuda_native_pointer_class, "nativePointer", "J");

	// explicit cast to make compiler and linter happy
	return static_cast<jlong>(ctx);
}


[[maybe_unused]] JNIEXPORT void JNICALL Java_org_apache_sysds_hops_codegen_SpoofCompiler_destroy_1cuda_1context
	([[maybe_unused]] JNIEnv *jenv, [[maybe_unused]] jobject jobj, jlong ctx, jint device_id) {
	SpoofCUDAContext::destroy_cuda(reinterpret_cast<SpoofCUDAContext *>(ctx), device_id);
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofOperator_getNativeStagingBuffer
	(JNIEnv *jenv, [[maybe_unused]] jclass jobj, jobject ptr, jlong _ctx, jint size) {
	std::string operator_name("SpoofOperator_getNativeStagingBuffer");
	try {
		// retrieve data handles from JVM
		auto *ctx = reinterpret_cast<SpoofCUDAContext *>(_ctx);
		if (size > ctx->current_mem_size)
			ctx->resize_staging_buffer(size);

		jobject object = jenv->NewDirectByteBuffer(ctx->staging_buffer, size);
		jenv->SetObjectField(ptr, pointer_buffer_field, object);
		jenv->SetLongField(ptr,native_pointer_field, reinterpret_cast<jlong>(ctx->staging_buffer));

		return 0;
	}
	catch (std::exception & e) {
		printException(operator_name, e);
	}
	catch (...) {
		printException(operator_name, std::runtime_error("unknown exception"), true);
	}
	return -1;
}

template<typename TEMPLATE>
int compile_spoof_operator
	(JNIEnv *jenv, [[maybe_unused]] jobject jobj, jlong _ctx, jstring name, jstring src, TEMPLATE op) {
	std::string operator_name;
	try {
		auto *ctx = reinterpret_cast<SpoofCUDAContext *>(_ctx);
		const char *cstr_name = jenv->GetStringUTFChars(name, nullptr);
		const char *cstr_src = jenv->GetStringUTFChars(src, nullptr);
		operator_name = cstr_name;
		op->name = operator_name;

		int status = ctx->compile(std::move(op), cstr_src);

		jenv->ReleaseStringUTFChars(src, cstr_src);
		jenv->ReleaseStringUTFChars(name, cstr_name);
		return status;
	}
	catch (std::exception &e) {
		printException(operator_name, e, true);
	}
	catch (...) {
		printException(operator_name, std::runtime_error("unknown exception"), true);
	}
	return -1;
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_hops_codegen_cplan_CNodeCell_compile_1nvrtc
	(JNIEnv *jenv, jobject jobj, jlong ctx, jstring name, jstring src, jint type, jint agg_op, jboolean sparseSafe) {
	std::unique_ptr<SpoofCellwiseOp> op = std::make_unique<SpoofCellwiseOp>(SpoofOperator::AggType(type),
																			SpoofOperator::AggOp(agg_op), sparseSafe);

	return compile_spoof_operator<std::unique_ptr<SpoofCellwiseOp>>(jenv, jobj, ctx, name, src, std::move(op));
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_hops_codegen_cplan_CNodeRow_compile_1nvrtc
	(JNIEnv *jenv, jobject jobj, jlong ctx, jstring name, jstring src, jint type, jint const_dim2,
	 jint num_vectors, jboolean TB1) {
	std::unique_ptr<SpoofRowwiseOp> op = std::make_unique<SpoofRowwiseOp>(SpoofOperator::RowType(type), TB1,
																		  num_vectors, const_dim2);
	return compile_spoof_operator<std::unique_ptr<SpoofRowwiseOp>>(jenv, jobj, ctx, name, src, std::move(op));
}


template<typename T, typename TEMPLATE>
int launch_spoof_operator([[maybe_unused]] JNIEnv *jenv, [[maybe_unused]] jclass jobj, jlong _ctx) {
	std::string operator_name("launch_spoof_operator jni-bridge");
	try {
		// retrieve data handles from JVM
		auto *ctx = reinterpret_cast<SpoofCUDAContext *>(_ctx);

#ifndef NDEBUG
		uint32_t opID = *reinterpret_cast<uint32_t*>(&ctx->staging_buffer[sizeof(uint32_t)]);
		// this implicitly checks if op exists
		operator_name = ctx->getOperatorName(opID);
		std::cout << "executing op=" << operator_name << " id=" << opID << std::endl;
#endif
		// transfers resource pointers to GPU and calls op->exec()
		ctx->launch<T, TEMPLATE>();

		return 0;
	}
	catch (std::exception &e) {
		printException(operator_name, e);
	}
	catch (...) {
		printException(operator_name, std::runtime_error("unknown exception"));
	}
	return -1;
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1d
	(JNIEnv *jenv, jclass jobj, jlong ctx) {
	return launch_spoof_operator<double, SpoofCellwise<double>>(jenv, jobj, ctx);
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1f
	(JNIEnv *jenv, jclass jobj, jlong ctx) {
	return launch_spoof_operator<double, SpoofCellwise<double>>(jenv, jobj, ctx);
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1d
	(JNIEnv *jenv, jclass jobj, jlong ctx) {
	return launch_spoof_operator<double, SpoofRowwise<double>>(jenv, jobj, ctx);
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1f
	(JNIEnv *jenv, jclass jobj, jlong ctx) {
	return launch_spoof_operator<double, SpoofRowwise<double>>(jenv, jobj, ctx);
}