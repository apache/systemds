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
#include <cstdint>
#include "cujava_logger.hpp"
#include "cujava_jni_utils.hpp"
#include "cujava_pointer_utils.hpp"

// ---- cached IDs / classes (definitions; headers should declare them 'extern') ----
jmethodID Object_getClass = nullptr;            // ()Ljava/lang/Class;
jmethodID Class_getComponentType = nullptr;     // ()Ljava/lang/Class;
jmethodID Class_newInstance = nullptr;          // ()Ljava/lang/Object;

jmethodID Buffer_isDirect = nullptr;            // ()Z
jmethodID Buffer_hasArray = nullptr;            // ()Z
jmethodID Buffer_array    = nullptr;            // ()Ljava/lang/Object;

jfieldID NativePointerObject_nativePointer = nullptr; // long

jclass   Pointer_class    = nullptr;            // org.apache.sysds.cujava.Pointer (global ref)
jfieldID Pointer_buffer   = nullptr;            // Ljava/nio/Buffer;
jfieldID Pointer_pointers = nullptr;            // [Lorg/apache/sysds/cujava/NativePointerObject;
jfieldID Pointer_byteOffset = nullptr;          // long

jmethodID Pointer_constructor = nullptr;        // ()V

// -----------------------------------------------------------------------------
// Initialize field- and method IDs for Pointer/Buffer plumbing
// -----------------------------------------------------------------------------
int initPointerUtils(JNIEnv *env)
{
    jclass cls = nullptr;

    // java.lang.Object#getClass()
    if (!init(env, cls, "java/lang/Object")) return JNI_ERR;
    if (!init(env, cls, Object_getClass, "getClass", "()Ljava/lang/Class;")) return JNI_ERR;

    // java.lang.Class methods we may need later (kept to match JCuda shape)
    if (!init(env, cls, "java/lang/Class")) return JNI_ERR;
    if (!init(env, cls, Class_getComponentType, "getComponentType", "()Ljava/lang/Class;")) return JNI_ERR;
    if (!init(env, cls, Class_newInstance,      "newInstance",      "()Ljava/lang/Object;")) return JNI_ERR;

    // java.nio.Buffer: isDirect/hasArray/array
    if (!init(env, cls, "java/nio/Buffer")) return JNI_ERR;
    if (!init(env, cls, Buffer_isDirect, "isDirect", "()Z")) return JNI_ERR;
    if (!init(env, cls, Buffer_hasArray, "hasArray", "()Z")) return JNI_ERR;
    if (!init(env, cls, Buffer_array,    "array",    "()Ljava/lang/Object;")) return JNI_ERR;

    // org.apache.sysds.cujava.NativePointerObject.nativePointer (long)
    if (!init(env, cls, "org/apache/sysds/cujava/NativePointerObject")) return JNI_ERR;
    if (!init(env, cls, NativePointerObject_nativePointer, "nativePointer", "J")) return JNI_ERR;

    // org.apache.sysds.cujava.Pointer
    if (!init(env, cls, "org/apache/sysds/cujava/Pointer")) return JNI_ERR;
    Pointer_class = (jclass)env->NewGlobalRef(cls);
    if (Pointer_class == nullptr) return JNI_ERR;

    if (!init(env, cls, Pointer_buffer,     "buffer",     "Ljava/nio/Buffer;")) return JNI_ERR;
    if (!init(env, cls, Pointer_pointers,   "pointers",   "[Lorg/apache/sysds/cujava/NativePointerObject;")) return JNI_ERR;
    if (!init(env, cls, Pointer_byteOffset, "byteOffset", "J")) return JNI_ERR;
    if (!init(env, cls, Pointer_constructor, "<init>", "()V")) return JNI_ERR;

    return JNI_VERSION_1_4;
}

// -----------------------------------------------------------------------------
// Helper: validate newly created PointerData
// -----------------------------------------------------------------------------
static PointerData* validatePointerData(JNIEnv *env, jobject nativePointerObject, PointerData *pointerData)
{
    if (pointerData == nullptr)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError",
            "Out of memory while creating pointer data");
        return nullptr;
    }
    if (!pointerData->init(env, nativePointerObject))
    {
        delete pointerData;
        return nullptr;
    }
    return pointerData;
}

// -----------------------------------------------------------------------------
// Factory: create a PointerData matching the Java-side object
// (mirrors JCuda: Pointer array -> PointersArrayPointerData,
//  Buffer direct -> DirectBufferPointerData,
//  Buffer with array -> ArrayBufferPointerData,
//  else Pointer(nativePointer+byteOffset) -> NativePointerData,
//  else non-Pointer/NULL -> NativePointerObjectPointerData)
// -----------------------------------------------------------------------------
PointerData* initPointerData(JNIEnv *env, jobject nativePointerObject)
{
    Logger::log(LOG_DEBUGTRACE, "Initializing pointer data for Java NativePointerObject %p\n", nativePointerObject);

    // NULL -> NativePointerObjectPointerData
    if (nativePointerObject == nullptr)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerObjectPointerData\n");
        auto *pd = new NativePointerObjectPointerData();
        return validatePointerData(env, nativePointerObject, pd);
    }

    // If not an instance of Pointer -> NativePointerObjectPointerData
    jboolean isPointer = env->IsInstanceOf(nativePointerObject, Pointer_class);
    if (!isPointer)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerObjectPointerData\n");
        auto *pd = new NativePointerObjectPointerData();
        return validatePointerData(env, nativePointerObject, pd);
    }

    // If Pointer.pointers != null -> PointersArrayPointerData
    jobjectArray pointersArray = (jobjectArray)env->GetObjectField(nativePointerObject, Pointer_pointers);
    if (pointersArray != nullptr)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing PointersArrayPointerData\n");
        auto *pd = new PointersArrayPointerData();
        return validatePointerData(env, nativePointerObject, pd);
    }

    // If Pointer.buffer != null -> Buffer paths
    jobject buffer = env->GetObjectField(nativePointerObject, Pointer_buffer);
    if (buffer != nullptr)
    {
        // Direct buffer?
        jboolean isDirect = env->CallBooleanMethod(buffer, Buffer_isDirect);
        if (env->ExceptionCheck()) return nullptr;
        if (isDirect == JNI_TRUE)
        {
            Logger::log(LOG_DEBUGTRACE, "Initializing DirectBufferPointerData\n");
            auto *pd = new DirectBufferPointerData();
            return validatePointerData(env, nativePointerObject, pd);
        }

        // Backed by primitive array?
        jboolean hasArray = env->CallBooleanMethod(buffer, Buffer_hasArray);
        if (env->ExceptionCheck()) return nullptr;
        if (hasArray == JNI_TRUE)
        {
            Logger::log(LOG_DEBUGTRACE, "Initializing ArrayBufferPointerData\n");
            auto *pd = new ArrayBufferPointerData();
            return validatePointerData(env, nativePointerObject, pd);
        }

        // Neither direct nor array-backed -> error (should have been checked in Java)
        Logger::log(LOG_ERROR, "Buffer is neither direct nor has an array\n");
        ThrowByName(env, "java/lang/IllegalArgumentException",
                    "Buffer is neither direct nor has an array");
        return nullptr;
    }

    // Plain Pointer: nativePointer + byteOffset
    Logger::log(LOG_DEBUGTRACE, "Initializing NativePointerData\n");
    auto *pd = new NativePointerData();
    return validatePointerData(env, nativePointerObject, pd);
}

// -----------------------------------------------------------------------------
// Release helper: calls PointerData::release and deletes the object
// -----------------------------------------------------------------------------
bool releasePointerData(JNIEnv *env, PointerData* &pointerData, jint mode)
{
    if (pointerData == nullptr) return true;
    if (!pointerData->release(env, mode)) return false;
    delete pointerData;
    pointerData = nullptr;
    return true;
}

// -----------------------------------------------------------------------------
// Misc helpers
// -----------------------------------------------------------------------------
bool isDirectByteBuffer(JNIEnv *env, jobject buffer)
{
    if (buffer == nullptr) return false;
    jboolean isDirect = env->CallBooleanMethod(buffer, Buffer_isDirect);
    if (env->ExceptionCheck()) return false;
    return (isDirect == JNI_TRUE);
}

bool isPointerBackedByNativeMemory(JNIEnv *env, jobject object)
{
    if (object == nullptr) return false;

    jlong np = env->GetLongField(object, NativePointerObject_nativePointer);
    if (np != 0) return true;

    jboolean isPtr = env->IsInstanceOf(object, Pointer_class);
    if (isPtr)
    {
        jobject buffer = env->GetObjectField(object, Pointer_buffer);
        return isDirectByteBuffer(env, buffer);
    }
    return false;
}

void setNativePointerValue(JNIEnv *env, jobject nativePointerObject, jlong pointer)
{
    if (nativePointerObject == nullptr) return;
    env->SetLongField(nativePointerObject, NativePointerObject_nativePointer, pointer);
}

void* getNativePointerValue(JNIEnv *env, jobject nativePointerObject)
{
    if (nativePointerObject == nullptr) return nullptr;
    jlong p = env->GetLongField(nativePointerObject, NativePointerObject_nativePointer);
    return (void*)(uintptr_t)p;
}

void setPointer(JNIEnv *env, jobject pointerObject, jlong pointer)
{
    if (pointerObject == nullptr) return;
    env->SetLongField(pointerObject, NativePointerObject_nativePointer, pointer);
    env->SetLongField(pointerObject, Pointer_byteOffset, 0);
}

void* getPointer(JNIEnv *env, jobject pointerObject)
{
    if (pointerObject == nullptr) return nullptr;
    jlong start = env->GetLongField(pointerObject, NativePointerObject_nativePointer);
    jlong off   = env->GetLongField(pointerObject, Pointer_byteOffset);
    jlong p     = start + off;
    return (void*)(uintptr_t)p;
}
