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


 int initJNIUtils(JNIEnv *env)
 {
     jclass cls = nullptr;

     // java.lang.String#getBytes()[B
     if (!init(env, cls, "java/lang/String")) return JNI_ERR;
     if (!init(env, cls, String_getBytes, "getBytes", "()[B")) return JNI_ERR;

     return JNI_VERSION_1_4;
 }

 /** Find a class by name. */
 bool init(JNIEnv *env, jclass& cls, const char *name)
 {
     cls = env->FindClass(name);
     if (cls == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to access class '%s'\n", name);
         return false;
     }
     return true;
 }

 /** Create a global ref to a class. */
 bool initGlobal(JNIEnv *env, jclass &globalCls, const char *className)
 {
     jclass cls = nullptr;
     if (!init(env, cls, className)) return false;
     globalCls = (jclass)env->NewGlobalRef(cls);
     if (globalCls == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to create reference to class %s\n", className);
         return false;
     }
     return true;
 }

 /** Resolve a field ID. */
 bool init(JNIEnv *env, jclass cls, jfieldID& field, const char *name, const char *signature)
 {
     field = env->GetFieldID(cls, name, signature);
     if (field == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to access field '%s' with signature '%s'\n", name, signature);
         return false;
     }
     return true;
 }

 /** Resolve a method ID. */
 bool init(JNIEnv *env, jclass cls, jmethodID& method, const char *name, const char *signature)
 {
     method = env->GetMethodID(cls, name, signature);
     if (method == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to access method '%s' with signature '%s'\n", name, signature);
         return false;
     }
     return true;
 }

 /** Global class + no-args constructor, convenient helper. */
 bool init(JNIEnv *env, jclass &globalCls, jmethodID &constructor, const char *className)
 {
     jclass cls = nullptr;
     if (!init(env, cls, className)) return false;
     if (!init(env, cls, constructor, "<init>", "()V")) return false;

     globalCls = (jclass)env->NewGlobalRef(cls);
     if (globalCls == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to create reference to class %s\n", className);
         return false;
     }
     return true;
 }

 /** Resolve the standard 'long nativePointer' field for a class. */
 bool initNativePointer(JNIEnv *env, jfieldID& field, const char *className)
 {
     jclass cls = env->FindClass(className);
     if (cls == nullptr)
     {
         Logger::log(LOG_ERROR, "Failed to access class %s\n", className);
         return false;
     }
     return init(env, cls, field, "nativePointer", "J");
 }

 /** Throw a Java exception by FQN. */
 void ThrowByName(JNIEnv *env, const char *name, const char *msg)
 {
     jclass cls = env->FindClass(name);
     if (cls != nullptr)
     {
         env->ThrowNew(cls, msg ? msg : "");
         env->DeleteLocalRef(cls);
     }
 }

