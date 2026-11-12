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

#ifndef CUJAVA_POINTER_UTILS_HPP
#define CUJAVA_POINTER_UTILS_HPP

#include <jni.h>
#include "cujava_jni_utils.hpp"
#include "cujava_logger.hpp"

// -----------------------------------------------------------------------------
// Init + helpers
// -----------------------------------------------------------------------------
int   initPointerUtils(JNIEnv* env);

class PointerData;

PointerData* initPointerData(JNIEnv* env, jobject nativePointerObject);
bool         releasePointerData(JNIEnv* env, PointerData*& pointerData, jint mode = 0);

void  setNativePointerValue(JNIEnv* env, jobject nativePointerObject, jlong pointer);
void* getNativePointerValue(JNIEnv* env, jobject nativePointerObject);

void  setPointer(JNIEnv* env, jobject pointerObject, jlong pointer);
void* getPointer(JNIEnv* env, jobject pointerObject);

bool  isDirectByteBuffer(JNIEnv* env, jobject buffer);
bool  isPointerBackedByNativeMemory(JNIEnv* env, jobject object);

// -----------------------------------------------------------------------------
// Cached JNI IDs / classes (initialized in initPointerUtils)
// -----------------------------------------------------------------------------
extern jmethodID Buffer_isDirect;   // ()Z
extern jmethodID Buffer_hasArray;   // ()Z
extern jmethodID Buffer_array;      // ()Ljava/lang/Object;

extern jfieldID  NativePointerObject_nativePointer; // long

extern jclass    Pointer_class;     // Global ref: org.apache.sysds.cujava.Pointer
extern jfieldID  Pointer_buffer;    // Ljava/nio/Buffer;
extern jfieldID  Pointer_pointers;  // [Lorg/apache/sysds/cujava/NativePointerObject;
extern jfieldID  Pointer_byteOffset;// long

extern jmethodID Pointer_constructor; // ()V

extern jmethodID Object_getClass;          // ()Ljava/lang/Class;
extern jmethodID Class_getComponentType;   // ()Ljava/lang/Class;
extern jmethodID Class_newInstance;        // ()Ljava/lang/Object;

// -----------------------------------------------------------------------------
// PointerData hierarchy
// -----------------------------------------------------------------------------

/**
 * Virtual base class for all possible representations of pointers.
 */
class PointerData
{
public:
    virtual ~PointerData() {}

    virtual bool  init(JNIEnv* env, jobject object) = 0;
    virtual bool  release(JNIEnv* env, jint mode = 0) = 0;

    virtual void* getPointer(JNIEnv* env) = 0;
    virtual void  releasePointer(JNIEnv* env, jint mode = 0) = 0;

    /**
     * For pointers inside pointer arrays that may be updated by native code:
     * write the new native address back into the Java object, if supported.
     */
    virtual bool  setNewNativePointerValue(JNIEnv* env, jlong nativePointerValue) = 0;
};


/**
 * Backed by a Java NativePointerObject that is NOT a Pointer instance.
 * Stores only the nativePointer value.
 */
class NativePointerObjectPointerData : public PointerData
{
private:
    jobject nativePointerObject; // global ref (may be null)
    jlong   nativePointer;

public:
    NativePointerObjectPointerData() : nativePointerObject(NULL), nativePointer(0) {}
    ~NativePointerObjectPointerData() {}

    bool init(JNIEnv* env, jobject object)
    {
        if (object != NULL)
        {
            nativePointerObject = env->NewGlobalRef(object);
            if (nativePointerObject == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while creating global reference for pointer data");
                return false;
            }
            nativePointer = env->GetLongField(object, NativePointerObject_nativePointer);
            if (env->ExceptionCheck()) return false;
        }
        Logger::log(LOG_DEBUGTRACE, "Initialized  NativePointerObjectPointerData %p\n", (void*)nativePointer);
        return true;
    }

    bool release(JNIEnv* env, jint = 0)
    {
        Logger::log(LOG_DEBUGTRACE, "Releasing    NativePointerObjectPointerData %p\n", (void*)nativePointer);
        if (nativePointerObject != NULL)
        {
            env->SetLongField(nativePointerObject, NativePointerObject_nativePointer, nativePointer);
            env->DeleteGlobalRef(nativePointerObject);
        }
        return true;
    }

    void* getPointer(JNIEnv*) { return (void*)nativePointer; }
    void  releasePointer(JNIEnv*, jint = 0) {}

    bool setNewNativePointerValue(JNIEnv*, jlong nativePointerValue)
    {
        nativePointer = nativePointerValue;
        return true;
    }
};


/**
 * Backed by a Java Pointer (nativePointer + byteOffset).
 */
class NativePointerData : public PointerData
{
private:
    jobject pointer;     // global ref
    jlong   nativePointer;
    jlong   byteOffset;

public:
    NativePointerData() : pointer(NULL), nativePointer(0), byteOffset(0) {}
    ~NativePointerData() {}

    bool init(JNIEnv* env, jobject object)
    {
        pointer = env->NewGlobalRef(object);
        if (pointer == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory while creating global reference for pointer data");
            return false;
        }

        nativePointer = env->GetLongField(object, NativePointerObject_nativePointer);
        if (env->ExceptionCheck()) return false;

        byteOffset = env->GetLongField(object, Pointer_byteOffset);
        if (env->ExceptionCheck()) return false;

        Logger::log(LOG_DEBUGTRACE, "Initialized  NativePointerData              %p\n", (void*)nativePointer);
        return true;
    }

    bool release(JNIEnv* env, jint = 0)
    {
        Logger::log(LOG_DEBUGTRACE, "Releasing    NativePointerData              %p\n", (void*)nativePointer);
        env->SetLongField(pointer, NativePointerObject_nativePointer, nativePointer);
        env->SetLongField(pointer, Pointer_byteOffset, byteOffset);
        env->DeleteGlobalRef(pointer);
        return true;
    }

    void* getPointer(JNIEnv*) { return (void*)(((char*)nativePointer) + byteOffset); }
    void  releasePointer(JNIEnv*, jint = 0) {}

    bool setNewNativePointerValue(JNIEnv*, jlong nativePointerValue)
    {
        nativePointer = nativePointerValue;
        byteOffset = 0;
        return true;
    }
};


/**
 * Backed by a Java Pointer that points to an array of NativePointerObjects.
 */
class PointersArrayPointerData : public PointerData
{
private:
    jobject      nativePointerObject; // global ref to the Java Pointer
    PointerData** arrayPointerDatas;  // parallel to Java array
    void*        startPointer;        // native array of void* (one per element)
    jlong        byteOffset;
    bool         localPointersInitialized;

    void initLocalPointers(JNIEnv* env)
    {
        Logger::log(LOG_DEBUGTRACE, "Initializing PointersArrayPointerData local pointers\n");
        jobjectArray pointersArray = (jobjectArray)env->GetObjectField(
            nativePointerObject, Pointer_pointers);
        long size = (long)env->GetArrayLength(pointersArray);
        void** localPointer = (void**)startPointer;
        for (int i = 0; i < size; i++)
        {
            if (arrayPointerDatas[i] != NULL)
                localPointer[i] = arrayPointerDatas[i]->getPointer(env);
            else
                localPointer[i] = NULL;
        }
        localPointersInitialized = true;
        Logger::log(LOG_DEBUGTRACE, "Initialized  PointersArrayPointerData local pointers\n");
    }

public:
    PointersArrayPointerData()
    : nativePointerObject(NULL),
      arrayPointerDatas(NULL),
      startPointer(NULL),
      byteOffset(0),
      localPointersInitialized(false) {}

    ~PointersArrayPointerData() {}

    bool init(JNIEnv* env, jobject object)
    {
        nativePointerObject = env->NewGlobalRef(object);
        if (nativePointerObject == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory while creating global reference for pointer data");
            return false;
        }

        jobjectArray pointersArray = (jobjectArray)env->GetObjectField(object, Pointer_pointers);
        long size = (long)env->GetArrayLength(pointersArray);

        void** localPointer = new void*[size];
        if (localPointer == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory while initializing pointer array");
            return false;
        }
        startPointer = (void*)localPointer;

        arrayPointerDatas = new PointerData*[size];
        if (arrayPointerDatas == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory while initializing pointer data array");
            return false;
        }

        for (int i = 0; i < size; i++)
        {
            jobject p = env->GetObjectArrayElement(pointersArray, i);
            if (env->ExceptionCheck()) return false;

            if (p != NULL)
            {
                PointerData* apd = initPointerData(env, p);
                if (apd == NULL) return false;
                arrayPointerDatas[i] = apd;
            }
            else
            {
                arrayPointerDatas[i] = NULL;
            }
        }

        byteOffset = env->GetLongField(object, Pointer_byteOffset);
        if (env->ExceptionCheck()) return false;

        Logger::log(LOG_DEBUGTRACE, "Initialized  PointersArrayPointerData       %p\n", startPointer);
        return true;
    }

    bool release(JNIEnv* env, jint mode = 0)
    {
        Logger::log(LOG_DEBUGTRACE, "Releasing    PointersArrayPointerData       %p\n", startPointer);

        if (!localPointersInitialized) initLocalPointers(env);

        jobjectArray pointersArray = (jobjectArray)env->GetObjectField(
            nativePointerObject, Pointer_pointers);
        long size = (long)env->GetArrayLength(pointersArray);

        void** localPointer = (void**)startPointer;
        if (mode != JNI_ABORT)
        {
            for (int i = 0; i < size; i++)
            {
                jobject p = env->GetObjectArrayElement(pointersArray, i);
                if (env->ExceptionCheck()) return false;

                if (p != NULL)
                {
                    void* oldLocalPointer = arrayPointerDatas[i]->getPointer(env);

                    Logger::log(LOG_DEBUGTRACE, "About to write back pointer %d in PointersArrayPointerData\n", i);
                    Logger::log(LOG_DEBUGTRACE, "Old local pointer was %p\n", oldLocalPointer);
                    Logger::log(LOG_DEBUGTRACE, "New local pointer is  %p\n", localPointer[i]);

                    if (localPointer[i] != oldLocalPointer)
                    {
                        Logger::log(LOG_DEBUGTRACE, "In pointer %d setting value %p\n", i, localPointer[i]);
                        bool updated = arrayPointerDatas[i]->setNewNativePointerValue(env, (jlong)localPointer[i]);
                        if (!updated) return false; // pending IllegalArgumentException
                    }
                }
                else if (localPointer[i] != NULL)
                {
                    ThrowByName(env, "java/lang/NullPointerException",
                                "Pointer points to an array containing a 'null' entry");
                    return false;
                }
            }
        }

        if (arrayPointerDatas != NULL)
        {
            for (int i = 0; i < size; i++)
            {
                if (arrayPointerDatas[i] != NULL)
                {
                    if (!releasePointerData(env, arrayPointerDatas[i], mode)) return false;
                }
            }
            delete[] arrayPointerDatas;
        }
        delete[] localPointer;

        env->DeleteGlobalRef(nativePointerObject);
        return true;
    }

    void* getPointer(JNIEnv* env)
    {
        if (!localPointersInitialized) initLocalPointers(env);
        return (void*)(((char*)startPointer) + byteOffset);
    }

    void  releasePointer(JNIEnv*, jint = 0) {}

    bool setNewNativePointerValue(JNIEnv* env, jlong)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer to an array of pointers may not be overwritten");
        return false;
    }
};


/**
 * Backed by a direct java.nio.Buffer.
 */
class DirectBufferPointerData : public PointerData
{
private:
    void* startPointer;
    jlong byteOffset;

public:
    DirectBufferPointerData() : startPointer(NULL), byteOffset(0) {}
    ~DirectBufferPointerData() {}

    bool init(JNIEnv* env, jobject object)
    {
        jobject buffer = env->GetObjectField(object, Pointer_buffer);
        startPointer = env->GetDirectBufferAddress(buffer);
        if (startPointer == 0)
        {
            ThrowByName(env, "java/lang/IllegalArgumentException",
                "Failed to obtain direct buffer address");
            return false;
        }

        byteOffset = env->GetLongField(object, Pointer_byteOffset);
        if (env->ExceptionCheck()) return false;

        Logger::log(LOG_DEBUGTRACE, "Initialized  DirectBufferPointerData        %p\n", startPointer);
        return true;
    }

    bool  release(JNIEnv*, jint = 0)
    {
        Logger::log(LOG_DEBUGTRACE, "Releasing    DirectBufferPointerData        %p\n", startPointer);
        return true;
    }

    void* getPointer(JNIEnv*) { return (void*)(((char*)startPointer) + byteOffset); }
    void  releasePointer(JNIEnv*, jint = 0) {}

    bool setNewNativePointerValue(JNIEnv* env, jlong)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer to a direct buffer may not be overwritten");
        return false;
    }
};


/**
 * Backed by a primitive-array-backed Buffer (e.g., ByteBuffer.wrap(...)).
 */
class ArrayBufferPointerData : public PointerData
{
private:
    jarray   array;        // global ref to the primitive array
    void*    startPointer; // set on first getPointer()
    jboolean isCopy;
    jlong    byteOffset;

public:
    ArrayBufferPointerData()
    : array(NULL), startPointer(NULL), isCopy(JNI_FALSE), byteOffset(0) {}
    ~ArrayBufferPointerData() {}

    bool init(JNIEnv* env, jobject object)
    {
        jobject buffer    = env->GetObjectField(object, Pointer_buffer);
        jobject localArray = env->CallObjectMethod(buffer, Buffer_array);
        if (env->ExceptionCheck()) return false;

        array = (jarray)env->NewGlobalRef(localArray);
        if (array == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory while creating array reference");
            return false;
        }

        byteOffset = env->GetLongField(object, Pointer_byteOffset);
        if (env->ExceptionCheck()) return false;

        Logger::log(LOG_DEBUGTRACE, "Initialized  ArrayBufferPointerData         %p (deferred)\n", startPointer);
        return true;
    }

    bool release(JNIEnv* env, jint mode = 0)
    {
        Logger::log(LOG_DEBUGTRACE, "Releasing    ArrayBufferPointerData         %p\n", startPointer);
        releasePointer(env, mode);
        env->DeleteGlobalRef(array);
        return true;
    }

    void* getPointer(JNIEnv* env)
    {
        if (startPointer == NULL)
        {
            Logger::log(LOG_DEBUGTRACE, "Initializing ArrayBufferPointerData critical\n");
            isCopy = JNI_FALSE;
            startPointer = env->GetPrimitiveArrayCritical(array, &isCopy);
            if (startPointer == NULL) return NULL;
            Logger::log(LOG_DEBUGTRACE, "Initialized  ArrayBufferPointerData         %p (isCopy %d)\n", startPointer, (int)isCopy);
        }
        return (void*)(((char*)startPointer) + byteOffset);
    }

    void releasePointer(JNIEnv* env, jint mode = 0)
    {
        if (startPointer != NULL)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    ArrayBufferPointerData critical\n");
            if (!isCopy)
                env->ReleasePrimitiveArrayCritical(array, startPointer, JNI_ABORT);
            else
                env->ReleasePrimitiveArrayCritical(array, startPointer, mode);
            startPointer = NULL;
        }
    }

    bool setNewNativePointerValue(JNIEnv* env, jlong)
    {
        ThrowByName(env, "java/lang/IllegalArgumentException",
            "Pointer to an array may not be overwritten");
        return false;
    }
};

#endif // CUJAVA_POINTER_UTILS_HPP
