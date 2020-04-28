
#include "spoof_native_jni.h"
#include "spoof_native_cuda.h"

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
}

JNIEXPORT jboolean JNICALL
Java_org_apache_sysds_runtime_codegen_SpoofNativeCUDA_execute_1d(
    JNIEnv *env, jobject jobj, jlong ctx, jstring name, jlong in_ptr,
    jlong side_ptr, jlong out_ptr, jdoubleArray scalars, jlong num_scalars,
    jlong m, jlong n, jlong grix) {

  std::cout << "bla" << std::endl;
  SpoofCudaContext *ctx_ = reinterpret_cast<SpoofCudaContext *>(ctx);

  std::string name_(env->GetStringUTFChars(name, NULL));
  return ctx_->execute_kernel(name_, reinterpret_cast<double **>(in_ptr),
                              reinterpret_cast<double **>(side_ptr),
                              reinterpret_cast<double **>(out_ptr),
                              reinterpret_cast<double **>(scalars), num_scalars,
                              m, n, grix);
}
