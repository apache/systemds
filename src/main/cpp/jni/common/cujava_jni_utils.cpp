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

#include "cujava_jni_utils.hpp"
#include "cujava_logger.hpp"

// Cached method ID (same as JCuda; useful for convertString if you add it later)
jmethodID String_getBytes = nullptr;


int initJNIUtils(JNIEnv *env) {
    jclass cls = nullptr;

    // java.lang.String#getBytes()[B
    if (!init(env, cls, "java/lang/String")) return JNI_ERR;
    if (!init(env, cls, String_getBytes, "getBytes", "()[B")) return JNI_ERR;

    return JNI_VERSION_1_4;
}

/** Find a class by name. */
bool init(JNIEnv *env, jclass& cls, const char *name) {
    cls = env->FindClass(name);
    if (cls == nullptr) {
        Logger::log(LOG_ERROR, "Failed to access class '%s'\n", name);
        return false;
    }
    return true;
}

/** Create a global ref to a class. */
bool initGlobal(JNIEnv *env, jclass &globalCls, const char *className) {
    jclass cls = nullptr;
    if (!init(env, cls, className)) return false;
    globalCls = (jclass)env->NewGlobalRef(cls);
    if (globalCls == nullptr) {
        Logger::log(LOG_ERROR, "Failed to create reference to class %s\n", className);
        return false;
    }
    return true;
}

/** Resolve a field ID. */
bool init(JNIEnv *env, jclass cls, jfieldID& field, const char *name, const char *signature) {
    field = env->GetFieldID(cls, name, signature);
    if (field == nullptr) {
        Logger::log(LOG_ERROR, "Failed to access field '%s' with signature '%s'\n", name, signature);
        return false;
    }
    return true;
}

/** Resolve a method ID. */
bool init(JNIEnv *env, jclass cls, jmethodID& method, const char *name, const char *signature) {
    method = env->GetMethodID(cls, name, signature);
    if (method == nullptr) {
        Logger::log(LOG_ERROR, "Failed to access method '%s' with signature '%s'\n", name, signature);
        return false;
    }
    return true;
}

/** Global class + no-args constructor, convenient helper. */
bool init(JNIEnv *env, jclass &globalCls, jmethodID &constructor, const char *className) {
    jclass cls = nullptr;
    if (!init(env, cls, className)) return false;
    if (!init(env, cls, constructor, "<init>", "()V")) return false;

    globalCls = (jclass)env->NewGlobalRef(cls);
    if (globalCls == nullptr) {
        Logger::log(LOG_ERROR, "Failed to create reference to class %s\n", className);
        return false;
    }
    return true;
}

/** Resolve the standard 'long nativePointer' field for a class. */
bool initNativePointer(JNIEnv *env, jfieldID& field, const char *className) {
    jclass cls = env->FindClass(className);
    if (cls == nullptr) {
        Logger::log(LOG_ERROR, "Failed to access class %s\n", className);
        return false;
    }
    return init(env, cls, field, "nativePointer", "J");
}

/** Throw a Java exception by FQN. */
void ThrowByName(JNIEnv *env, const char *name, const char *msg) {
    jclass cls = env->FindClass(name);
    if (cls != nullptr) {
        env->ThrowNew(cls, msg ? msg : "");
        env->DeleteLocalRef(cls);
    }
}

/** Utility to set one element of a long[] array. */
bool set(JNIEnv *env, jlongArray ja, int index, jlong value) {
    if (ja == nullptr) return true;

    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len) {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
                    "Array index out of bounds");
        return false;
    }

    jlong *a = (jlong*)env->GetPrimitiveArrayCritical(ja, nullptr);
    if (a == nullptr) return false;

    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}

/** Utility to set one element of an int[] array. */
bool set(JNIEnv *env, jintArray ja, int index, jint value) {
    if (ja == nullptr) {
        return true;
    }
    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len) {
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array index out of bounds");
        return false;
    }
    jint *a = (jint*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == nullptr) {
        return false;
    }
    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}

/** Helpers for setting cudaDeviceProperties. */
bool setFieldBytes(JNIEnv* env, jobject obj, jfieldID fid, const jbyte* src, jsize n) {
    jbyteArray arr = (jbyteArray)env->GetObjectField(obj, fid);
    if (arr == nullptr || env->GetArrayLength(arr) < n) {
        jbyteArray tmp = env->NewByteArray(n);
        if (tmp == nullptr) return false;
        env->SetObjectField(obj, fid, tmp);
        arr = tmp;
    }
    env->SetByteArrayRegion(arr, 0, n, src);
    return !env->ExceptionCheck();
}

bool setFieldInts(JNIEnv* env, jobject obj, jfieldID fid, const jint* src, jsize n) {
    jintArray arr = (jintArray)env->GetObjectField(obj, fid);
    if (arr == nullptr || env->GetArrayLength(arr) < n) {
        jintArray tmp = env->NewIntArray(n);
        if (tmp == nullptr) return false;
        env->SetObjectField(obj, fid, tmp);
        arr = tmp;
    }
    env->SetIntArrayRegion(arr, 0, n, src);
    return !env->ExceptionCheck();
}

bool zeroFieldInts(JNIEnv* env, jobject obj, jfieldID fid) {
    jintArray arr = (jintArray)env->GetObjectField(obj, fid);
    if (arr == nullptr) return true;
    jsize n = env->GetArrayLength(arr);
    if (n <= 0) return true;
    jint* zeros = new (std::nothrow) jint[n]();
    if (!zeros) {
        ThrowByName(env, "java/lang/OutOfMemoryError", "Out of memory zeroing int array");
        return false;
    }
    env->SetIntArrayRegion(arr, 0, n, zeros);
    delete[] zeros;
    return !env->ExceptionCheck();
}


char* toNativeCString(JNIEnv* env, jstring js, int* length) {
    if (js == nullptr) return nullptr;

    if (env->EnsureLocalCapacity(2) < 0) {
        ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory during string reference creation");
        return nullptr;
    }

    jbyteArray bytes = (jbyteArray)env->CallObjectMethod(js, String_getBytes);
    if (env->ExceptionCheck() || bytes == nullptr) {
        return nullptr;
    }

    jint len = env->GetArrayLength(bytes);
    if (length) *length = (int)len;

    char* out = new char[len + 1];
    if (out == nullptr) {
        ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory during string creation");
        env->DeleteLocalRef(bytes);
        return nullptr;
    }

    env->GetByteArrayRegion(bytes, 0, len, (jbyte*)out);
    out[len] = '\0';
    env->DeleteLocalRef(bytes);
    return out;
}


bool allocNativeArrayFromJLongs(JNIEnv* env, jlongArray javaArr, size_t*& nativeArr, bool copyFromJava) {
    if (javaArr == nullptr) {
        nativeArr = nullptr;
        return true;
    }
    jsize n = env->GetArrayLength(javaArr);

    size_t* tmp = new (std::nothrow) size_t[(size_t)n];
    if (!tmp) {
        ThrowByName(env, "java/lang/OutOfMemoryError", "Out of memory during array creation");
        nativeArr = nullptr;
        return false;
    }

    if (copyFromJava) {
        jlong* jptr = (jlong*)env->GetPrimitiveArrayCritical(javaArr, nullptr);
        if (!jptr) {
            delete[] tmp; nativeArr = nullptr; return false;
        }
        for (jsize i = 0; i < n; ++i) tmp[i] = (size_t)jptr[i];
        env->ReleasePrimitiveArrayCritical(javaArr, jptr, JNI_ABORT); // input-only
    }

    nativeArr = tmp;
    return true;
}

bool commitAndFreeNativeArrayToJLongs(JNIEnv* env, size_t*& nativeArr, jlongArray javaArr, bool copyToJava) {
    if (javaArr == nullptr) {
        delete[] nativeArr; nativeArr = nullptr; return true;
    }
    if (copyToJava && nativeArr) {
        jsize n = env->GetArrayLength(javaArr);
        jlong* jptr = (jlong*)env->GetPrimitiveArrayCritical(javaArr, nullptr);
        if (!jptr) {
            delete[] nativeArr; nativeArr = nullptr;
            return false;
        }
        for (jsize i = 0; i < n; ++i) jptr[i] = (jlong)nativeArr[i];
        env->ReleasePrimitiveArrayCritical(javaArr, jptr, 0); // commit
    }
    delete[] nativeArr;
    nativeArr = nullptr;
    return true;
}

// Back-compat wrappers
bool initNative(JNIEnv* env, jlongArray javaArr, size_t*& nativeArr, bool fill) {
    return allocNativeArrayFromJLongs(env, javaArr, nativeArr, fill);
}
bool releaseNative(JNIEnv* env, size_t*& nativeArr, jlongArray javaArr, bool writeBack) {
    return commitAndFreeNativeArrayToJLongs(env, nativeArr, javaArr, writeBack);
}
