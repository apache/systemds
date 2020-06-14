
#include "spoof_native_jni.h"
#include "spoof_native_cuda.h"

// JNI Methods to get/release arrays
#define GET_ARRAY(env, input)                                                  \
  ((void *)env->GetPrimitiveArrayCritical(input, NULL))

#define RELEASE_ARRAY(env, java, cpp)                                                  \
  (env->ReleasePrimitiveArrayCritical(java, cpp, NULL))

JNIEXPORT jlong JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_initialize_1cuda_1context(
    JNIEnv *env, jobject jobj, jint device_id) {
  return SpoofCudaContext::initialize_cuda(device_id);
}

JNIEXPORT void JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_destroy_1cuda_1context(
    JNIEnv *env, jobject jobj, jlong ctx, jint device_id) {
  SpoofCudaContext::destroy_cuda(reinterpret_cast<SpoofCudaContext *>(ctx),
                                 device_id);
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysds_hops_codegen_SpoofCompiler_compile_1cuda_1kernel(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jstring src) {
  SpoofCudaContext *ctx_ = reinterpret_cast<SpoofCudaContext *>(ctx);
  std::string src_(env->GetStringUTFChars(src, NULL));
  std::string name_(env->GetStringUTFChars(name, NULL));
  return ctx_->compile_cuda(src_, name_);
  env->ReleaseStringUTFChars(src, src_.c_str());
  env->ReleaseStringUTFChars(name, name_.c_str());
}

JNIEXPORT jdouble JNICALL
Java_org_apache_sysds_runtime_codegen_SpoofNativeCUDA_execute_1d(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jlongArray in_ptrs,
    jlong num_inputs, jlongArray side_ptrs, jlong num_sides, jlong out_ptr,
    jdoubleArray scalars_, jlong num_scalars, jlong m, jlong n, jlong grix) {

  SpoofCudaContext *ctx_ = reinterpret_cast<SpoofCudaContext *>(ctx);

  std::string name_(env->GetStringUTFChars(name, NULL));

  long *inputs_ = reinterpret_cast<long *>(GET_ARRAY(env, in_ptrs));
  double **inputs = reinterpret_cast<double **>(&inputs_[0]);
  double **sides = reinterpret_cast<double **>(GET_ARRAY(env, side_ptrs));
  double *scalars = reinterpret_cast<double *>(GET_ARRAY(env, scalars_));

  double result = ctx_->execute_kernel(
      name_, inputs, num_inputs, sides, num_sides,
      reinterpret_cast<double *>(out_ptr), scalars, num_scalars, m, n, grix);

  env->ReleaseStringUTFChars(name, name_.c_str());
  RELEASE_ARRAY(env, in_ptrs, inputs_);
  RELEASE_ARRAY(env, side_ptrs, sides);
  RELEASE_ARRAY(env, scalars_, scalars);
  return result;
}

JNIEXPORT jfloat JNICALL
Java_org_apache_sysds_runtime_codegen_SpoofNativeCUDA_execute_1f(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jlongArray in_ptrs,
    jlong num_inputs, jlongArray side_ptrs, jlong num_sides, jlong out_ptr,
    jfloatArray scalars_, jlong num_scalars, jlong m, jlong n, jlong grix) {

  SpoofCudaContext *ctx_ = reinterpret_cast<SpoofCudaContext *>(ctx);

  std::string name_(env->GetStringUTFChars(name, NULL));

  long *inputs_ = reinterpret_cast<long *>(GET_ARRAY(env, in_ptrs));
  float **inputs = reinterpret_cast<float **>(&inputs_[0]);
  float **sides = reinterpret_cast<float **>(GET_ARRAY(env, side_ptrs));
  float *scalars = reinterpret_cast<float *>(GET_ARRAY(env, scalars_));

  float result = ctx_->execute_kernel(
      name_, inputs, num_inputs, sides, num_sides,
      reinterpret_cast<float *>(out_ptr), scalars, num_scalars, m, n, grix);
  env->ReleaseStringUTFChars(name, name_.c_str());
  RELEASE_ARRAY(env, in_ptrs, inputs_);
  RELEASE_ARRAY(env, side_ptrs, sides);
  RELEASE_ARRAY(env, scalars_, scalars);
  return result;
}