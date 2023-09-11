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

import org.apache.commons.lang.SystemUtils;

import java.io.File;

public class ImgNativeHelper extends NativeHelper {
    private String blasType;

    // Constructor that accepts the BLAS type
    public ImgNativeHelper(String blasType) {
        if(blasType != null) this.blasType = blasType;
        else this.blasType = "openblas";
        loadNativeLibrary();
    }

    // Static initialization block is removed

    // Method to load the native library based on the specified BLAS type
    private void loadNativeLibrary() {
        try {
            if (SystemUtils.IS_OS_LINUX) {
                String libname = blasType + "-Linux-x86_64.so";
                System.load("/src/main/cpp/lib/libsystemds_" + libname);
            } else if (SystemUtils.IS_OS_WINDOWS) {
                String libname = blasType + "-Windows-x86_64.dll";
                System.load(System.getProperty("user.dir") + "/src/main/cpp/lib/".replace("/", File.separator) + "libsystemds_" + libname);
            } else if (SystemUtils.IS_OS_MAC || SystemUtils.IS_OS_MAC_OSX) {
               // String libname = "systemds_" + blasType + "-Darwin-x86_64";
                System.load(System.getProperty("user.dir") + "/src/main/cpp/lib/libsystemds_" + blasType + "-Darwin-x86_64.dylib");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Rotate an image by the specified number of radians.
     *
     * @param img_in     Input image represented as a 1D double array.
     * @param rows       Number of rows in the image.
     * @param cols       Number of columns in the image.
     * @param radians    Angle in radians by which to rotate the image.
     * @param fill_value Value to fill empty pixels after rotation.
     * @param img_out    Output image represented as a 1D double array.
     */
    public native void imageRotate(double[] img_in, int rows, int cols, double radians, double fill_value, double[] img_out);

    /**
     * Extract a cutout (subregion) from an image.
     *
     * @param img_in     Input image represented as a 1D double array.
     * @param rows       Number of rows in the image.
     * @param cols       Number of columns in the image.
     * @param x          X-coordinate of the top-left corner of the cutout.
     * @param y          Y-coordinate of the top-left corner of the cutout.
     * @param width      Width of the cutout.
     * @param height     Height of the cutout.
     * @param fill_value Value to fill empty pixels in the cutout.
     * @return A double array representing the cutout.
     */
    public native double[] imageCutout(double[] img_in, int rows, int cols, int x, int y, int width, int height, double fill_value);

    /**
     * Crop an image to a specified width and height, starting from a specified offset.
     *
     * @param img_in    Input image represented as a 1D double array.
     * @param orig_w    Original width of the image.
     * @param orig_h    Original height of the image.
     * @param w         Width of the cropped region.
     * @param h         Height of the cropped region.
     * @param x_offset  X-coordinate offset for cropping.
     * @param y_offset  Y-coordinate offset for cropping.
     * @return A double array representing the cropped image.
     */
    public native double[] cropImage(double[] img_in, int orig_w, int orig_h, int w, int h, int x_offset, int y_offset);

    /**
     * Translate (shift) an image by the specified offset in both X and Y directions.
     *
     * @param img_in     Input image represented as a 1D double array.
     * @param offset_x   X-coordinate offset for translation.
     * @param offset_y   Y-coordinate offset for translation.
     * @param in_w       Input image width.
     * @param in_h       Input image height.
     * @param out_w      Output image width.
     * @param out_h      Output image height.
     * @param fill_value Value to fill empty pixels after translation.
     * @param img_out    Output image represented as a 1D double array.
     */
    public native void imgTranslate(double[] img_in, double offset_x, double offset_y,
                                           int in_w, int in_h, int out_w, int out_h,
                                           double fill_value, double[] img_out);

}
