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

#ifdef _WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#endif

#include "common.h"
#include "libmatrixdnn.h"
#include "libmatrixmult.h"
#include "systemds.h"
#include "imgUtils.h"

JNIEXPORT void JNICALL Java_org_apache_sysds_utils_ImgNativeHelper_imageRotate
    (JNIEnv* env, jclass clazz, jdoubleArray img_in, jint rows, jint cols, jdouble radians, jdouble fill_value, jdoubleArray img_out) {
    
    if (rows != cols) {
        jclass exceptionClass = env->FindClass("java/lang/IllegalArgumentException");
        if (exceptionClass != NULL) {
            env->ThrowNew(exceptionClass, "Image dimensions must be equal");
            return;
        }
    }

    jsize num_pixels = env->GetArrayLength(img_in);
    jdouble* img_in_data = env->GetDoubleArrayElements(img_in, NULL);
    jdouble* img_out_data = new jdouble[num_pixels];
    imageRotate(img_in_data, rows, cols, radians, fill_value, img_out_data);
    env->SetDoubleArrayRegion(img_out, 0, num_pixels, img_out_data);
    delete[] img_out_data;
    env->ReleaseDoubleArrayElements(img_in, img_in_data, JNI_ABORT);
}

JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_ImgNativeHelper_imageCutout
    (JNIEnv* env, jclass cls, jdoubleArray img_in, jint rows, jint cols, jint x, jint y, jint width, jint height, jdouble fill_value) {
   
    jdouble* img_in_arr = env->GetDoubleArrayElements(img_in, nullptr);
    jdouble* img_out_arr = imageCutout(img_in_arr, rows, cols, x, y, width, height, fill_value);
    jdoubleArray img_out = env->NewDoubleArray(rows * cols);

    env->SetDoubleArrayRegion(img_out, 0, rows * cols, img_out_arr);
    env->ReleaseDoubleArrayElements(img_in, img_in_arr, JNI_ABORT);
    delete[] img_out_arr;

    return img_out;
}

JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_ImgNativeHelper_cropImage(JNIEnv *env, jclass,
    jdoubleArray img_in, jint orig_w, jint orig_h, jint w, jint h, jint x_offset, jint y_offset) {
    jsize length = env->GetArrayLength(img_in);
    double *img_in_array = env->GetDoubleArrayElements(img_in, 0);

    int start_h = (ceil((orig_h - h) / 2)) + y_offset - 1;
    int end_h = (start_h + h - 1);
    int start_w = (ceil((orig_w - w) / 2)) + x_offset - 1;
    int end_w = (start_w + w - 1);

    jdoubleArray img_out_java;
    if (start_h < 0 || end_h >= orig_h || start_w < 0 || end_w >= orig_w) {
      img_out_java = env->NewDoubleArray(orig_w * orig_h);
      env->SetDoubleArrayRegion(img_out_java, 0, orig_w * orig_h, img_in_array);

    }else {
        double *img_out = imageCrop(img_in_array, orig_w, orig_h, w, h, x_offset, y_offset);

         if (img_out == nullptr) {
                 return nullptr;
             }

         img_out_java = env->NewDoubleArray(w * h);
         env->SetDoubleArrayRegion(img_out_java, 0, w * h, img_out);
         delete[] img_out;
     }

     env->ReleaseDoubleArrayElements(img_in, img_in_array, 0);

    return img_out_java;
}

/*JNIEXPORT jdoubleArray JNICALL Java_org_apache_sysds_utils_ImgNativeHelper_shearImage(JNIEnv *env, jclass,
    jdoubleArray img_in, jint width, jint height, jdouble shear_x, jdouble shear_y, jdouble fill_value) {

   
    jsize length = env->GetArrayLength(img_in);
    double *img_in_array = env->GetDoubleArrayElements(img_in, 0);

   
    double* img_out = img_transform(img_in_array, width, height, shear_x, shear_y, fill_value);

   
    env->ReleaseDoubleArrayElements(img_in, img_in_array, 0);

    if (img_out == nullptr) {
        return nullptr;
    }

   
    jdoubleArray img_out_java = env->NewDoubleArray(width * height);
    env->SetDoubleArrayRegion(img_out_java, 0, width * height, img_out);

   
    delete[] img_out;

    return img_out_java;
}*/

JNIEXPORT void JNICALL Java_org_apache_sysds_utils_ImgNativeHelper_imgTranslate(JNIEnv *env, jclass cls,
                                                          jdoubleArray img_in, jdouble offset_x, jdouble offset_y,
                                                          jint in_w, jint in_h, jint out_w, jint out_h,
                                                          jdouble fill_value, jdoubleArray img_out) {
   
    jdouble* j_img_in = env->GetDoubleArrayElements(img_in, nullptr);
    jdouble* j_img_out = env->GetDoubleArrayElements(img_out, nullptr);
    img_translate(j_img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, j_img_out);
    env->ReleaseDoubleArrayElements(img_in, j_img_in, 0);
    env->ReleaseDoubleArrayElements(img_out, j_img_out, 0);
}