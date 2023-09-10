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

package org.apache.sysds.utils;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang.SystemUtils;

import java.io.File;

public class ImgNativeHelper extends NativeHelper {

    static {
        final String blas = "openblas";
        try{
            if(SystemUtils.IS_OS_LINUX) {
                String libname = blas+"-Linux-x86_64.so";
                System.load(System.getProperty("user.dir")
                        + "/src/main/cpp/lib/libsystemds_"+libname);
            }else if (SystemUtils.IS_OS_WINDOWS) {
                String libname = blas+"-Windows-x86_64.dll";
                System.load(System.getProperty("user.dir")
                        + "/src/main/cpp/lib/".replace("/",File.separator)+"libsystemds_"+libname);
            }else {
                /*String libname = "systemds_"+blas+"-Darwin-x86_64";
                System.load(System.getProperty("user.dir")
                        + "/src/main/cpp/lib/libsystemds_"+blas+"-Darwin-x86_64.dylib");*/
                throw new NotImplementedException("OS Currently Not Supported");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    //image rotation
    public static native void imageRotate(double[] img_in, int rows, int cols, double radians, double fill_value, double[] img_out);

    //image cutout
    public static native double[] imageCutout(double[] img_in, int rows, int cols, int x, int y, int width, int height, double fill_value);

    //image crop
    public static native double[] cropImage(double[] img_in, int orig_w, int orig_h, int w, int h, int x_offset, int y_offset);

    //image translate
    public static native void imgTranslate(double[] img_in, double offset_x, double offset_y,
                                           int in_w, int in_h, int out_w, int out_h,
                                           double fill_value, double[] img_out);
}
