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

#ifndef CUJAVA_JNI_UTILS_HPP
#define CUJAVA_JNI_UTILS_HPP

#include <jni.h>

bool init(JNIEnv* env, jclass& cls, const char* name);
bool initGlobal(JNIEnv* env, jclass& globalCls, const char* className);
bool init(JNIEnv* env, jclass cls, jfieldID& field, const char* name, const char* signature);
bool init(JNIEnv* env, jclass cls, jmethodID& method, const char* name, const char* signature);
bool init(JNIEnv* env, jclass& globalCls, jmethodID& constructor, const char* className);
bool initNativePointer(JNIEnv* env, jfieldID& field, const char* className);
bool set(JNIEnv *env, jlongArray ja, int index, jlong value);
bool set(JNIEnv *env, jintArray ja, int index, jint value);

// ---- Exceptions ----
void ThrowByName(JNIEnv* env, const char* name, const char* msg);

// ---- Module init (optional; keep if called from JNI_OnLoad) ----
int initJNIUtils(JNIEnv* env);

// ---- Cached IDs (minimal) ----
extern jmethodID String_getBytes; // ()[B

#endif // CUJAVA_JNI_UTILS_HPP

