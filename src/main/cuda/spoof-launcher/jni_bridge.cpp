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
#define GET_ARRAY(env, input)((void *)env->GetPrimitiveArrayCritical(input, nullptr))

#define RELEASE_ARRAY(env, java, cpp)(env->ReleasePrimitiveArrayCritical(java, cpp, 0))

// error output helper
void printException(const std::string& name, const std::exception& e, bool compile = false) {
	std::string type = compile ? "compiling" : "executing";
	std::cout << "std::exception while " << type << "  SPOOF CUDA operator " << name << ":\n" << e.what() << std::endl;
}


// a pod struct to have names for the passed pointers
template<typename T>
struct LaunchMetadata {
	const T& opID;
	const T& grix;
	const size_t& num_inputs;
	const size_t& num_sides;
	
	// num entries describing one matrix (6 entries):
	// {nnz,rows,cols,row_ptr,col_idxs,data}
	const size_t& entry_size;
	const T& num_scalars;
	
	explicit LaunchMetadata(const size_t* jvals) : opID(jvals[0]), grix(jvals[1]), num_inputs(jvals[2]),
			num_sides(jvals[3]), entry_size(jvals[4]), num_scalars(jvals[5]) {}
};


[[maybe_unused]] JNIEXPORT jlong JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_initialize_1cuda_1context(
    	JNIEnv *jenv, [[maybe_unused]] jobject jobj, jint device_id, jstring resource_path) {
	const char *cstr_rp = jenv->GetStringUTFChars(resource_path, nullptr);
	size_t ctx = SpoofCUDAContext::initialize_cuda(device_id, cstr_rp);
	jenv->ReleaseStringUTFChars(resource_path, cstr_rp);
	return static_cast<jlong>(ctx);
}


[[maybe_unused]] JNIEXPORT void JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_destroy_1cuda_1context(
		[[maybe_unused]] JNIEnv *jenv, [[maybe_unused]] jobject jobj, jlong ctx, jint device_id) {
	SpoofCUDAContext::destroy_cuda(reinterpret_cast<SpoofCUDAContext *>(ctx), device_id);
}


template<typename TEMPLATE>
int compile_spoof_operator(JNIEnv *jenv, [[maybe_unused]] jobject jobj, jlong _ctx, jstring name, jstring src, TEMPLATE op) {
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
	catch (std::exception& e) {
		printException(operator_name, e, true);
	}
	catch (...) {
		printException(operator_name, std::runtime_error("unknown exception"), true);
	}
	return -1;
}


[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_hops_codegen_cplan_CNodeCell_compile_1nvrtc
		(JNIEnv *jenv, jobject jobj, jlong ctx, jstring name, jstring src, jint type, jint agg_op,
				jboolean sparseSafe) {
	
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
int launch_spoof_operator(JNIEnv *jenv, [[maybe_unused]] jclass jobj, jlong _ctx, jlongArray _meta, jlongArray in,
		jlongArray _sides, jlongArray out, jlong _scalars) {
	std::string operator_name("unknown");
	try {
		// retrieve data handles from JVM
		auto *metacast = reinterpret_cast<size_t *>(GET_ARRAY(jenv, _meta));
		auto *ctx = reinterpret_cast<SpoofCUDAContext *>(_ctx);
		auto *inputs = reinterpret_cast<size_t *>(GET_ARRAY(jenv, in));
		auto *sides = reinterpret_cast<size_t *>(GET_ARRAY(jenv, _sides));
		auto *output = reinterpret_cast<size_t *>(GET_ARRAY(jenv, out));
//		auto *scalars = reinterpret_cast<T *>(GET_ARRAY(jenv, _scalars));
		auto *scalars = reinterpret_cast<T *>(_scalars);
		LaunchMetadata<size_t> meta(metacast);
		
		// this implicitly checks if op exists
		operator_name = ctx->getOperatorName(meta.opID);
		
		// wrap/cast inputs
		std::vector<Matrix<T>> mats_in;
		for(auto i = 0ul; i < meta.num_inputs; i+=meta.entry_size)
			mats_in.emplace_back(&inputs[i]);
		
		// wrap/cast sides
		std::vector<Matrix<T>> mats_sides;
		for(auto i = 0ul; i < meta.num_sides; i+=meta.entry_size)
			mats_sides.emplace_back(&sides[i]);
		
		// wrap/cast output
		Matrix<T> mat_out(output);
		
		// wrap/cast scalars
//		std::unique_ptr<Matrix<T>> mat_scalars = scalars == nullptr ? 0 : std::make_unique<Matrix<T>>(scalars);
		
		// transfers resource pointers to GPU and calls op->exec()
		ctx->launch<T, TEMPLATE>(meta.opID, mats_in, mats_sides, mat_out, scalars, meta.grix);
		
		// release data handles from JVM
		RELEASE_ARRAY(jenv, _meta, metacast);
		RELEASE_ARRAY(jenv, in, inputs);
		RELEASE_ARRAY(jenv, _sides, sides);
		RELEASE_ARRAY(jenv, out, output);
//		RELEASE_ARRAY(jenv, _scalars, scalars);
		
		return 0;
	}
	catch (std::exception& e) {
		printException(operator_name, e);
	}
	catch (...) {
		printException(operator_name, std::runtime_error("unknown exception"));
	}
	return -1;
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1f
		(JNIEnv *jenv, jclass jobj, jlong ctx, jlongArray meta, jlongArray in, jlongArray sides, jlongArray out,
		 jlong scalars) {
	return launch_spoof_operator<float, SpoofCellwise<float>>(jenv, jobj, ctx, meta, in, sides, out, scalars);
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDACellwise_execute_1d
		(JNIEnv *jenv, jclass jobj, jlong ctx, jlongArray meta, jlongArray in, jlongArray sides, jlongArray out,
		 jlong scalars) {
	return launch_spoof_operator<double, SpoofCellwise<double>>(jenv, jobj, ctx, meta, in, sides, out, scalars);
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1f
		(JNIEnv *jenv, jclass jobj, jlong ctx, jlongArray meta, jlongArray in, jlongArray sides, jlongArray out,
		 jlong scalars) {
	return launch_spoof_operator<float, SpoofRowwise<float>>(jenv, jobj, ctx, meta, in, sides, out, scalars);
}

[[maybe_unused]] JNIEXPORT jint JNICALL Java_org_apache_sysds_runtime_codegen_SpoofCUDARowwise_execute_1d
		(JNIEnv *jenv, jclass jobj, jlong ctx, jlongArray meta, jlongArray in, jlongArray sides, jlongArray out,
		 jlong scalars) {
	return launch_spoof_operator<double, SpoofRowwise<double>>(jenv, jobj, ctx, meta, in, sides, out, scalars);
}
