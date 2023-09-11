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

package org.apache.sysds.test.functions.nativ;

import org.apache.sysds.utils.ImgNativeHelper;
import org.junit.Test;
import static org.junit.Assert.assertArrayEquals;

public class ImgUtilsTest {

    private final String blasType = "mkl";
    private final ImgNativeHelper imgNativeHelper = new ImgNativeHelper(blasType);

    @Test
    public void testImageRotation90And45() {

        // Input image dimensions
        int rows = 3;
        int cols = 3;
        // Input image
        double[] img_in = {
                1,2,3,
                4,5,6,
                7,8,9
        };
        // Rotation angle in radians
        double radians = Math.PI / 2;
        // Fill value for the output image
        double fill_value = 0.0;
        // Expected output image
        double[] expected_img_out_90 = {
                3,6,9,
                2,5,8,
                1,4,7
        };

        double[] expected_img_out_45 = {
                2,3,6,
                1,5,9,
                4,7,8
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];
        imgNativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);
        assertArrayEquals(expected_img_out_90, img_out, 0.0001);
        //rotate by 45
        imgNativeHelper.imageRotate(img_in, rows, cols, radians/2, fill_value, img_out);
        assertArrayEquals(expected_img_out_45, img_out, 0.0001);
    }


    @Test
    public void testImageRotation90And45_4x4() {

        // Input image dimensions
        int rows = 4;
        int cols = 4;
        // Input image
        double[] img_in = {
                1,2,3,5,
                4,5,6,5,
                7,8,9,5,
                5,5,5,5,
        };
        // Rotation angle in radians
        double radians = Math.PI / 2;
        // Fill value for the output image
        double fill_value = 0.0;
        // Expected output image
        double[] expected_img_out_90 = {
                5,5,5,5,
                3,6,9,5,
                2,5,8,5,
                1,4,7,5
        };


        double[] expected_img_out_45 = {
                0,3,5,0,
                2,5,6,5,
                4,5,8,5,
                0,7,5,0
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];
        imgNativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);
        assertArrayEquals(expected_img_out_90, img_out, 0.0001);
        //rotate by 45
        imgNativeHelper.imageRotate(img_in, rows, cols, radians/2, fill_value, img_out);
        assertArrayEquals(expected_img_out_45, img_out, 0.0001);
    }

    @Test
    public void testImageRotation45() {
        // Input image dimensions
        int rows = 6;
        int cols = 6;

        // Input image
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,5.0,6.0,
                5.0, 6.0, 7.0, 8.0,9.0,10.0,
                9.0, 10.0, 11.0, 12.0,13.0,14.0,
                13.0, 14.0, 15.0, 16.0,17.0,18.0,
                17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                21.0,22.0,23.0,24.0,25.0,26.0
        };

        // Rotation angle in radians
        double radians = Math.PI / 4;

        // Fill value for the output image
        double fill_value = 0.0;

        // Expected output image
        double[] expected_img_out = {
                0, 4, 5, 10 ,14, 0,
                3, 4, 8 ,13 ,18 ,18,
                2, 7, 12 ,16 ,17 ,22,
                5, 10, 15 ,16, 20 ,25,
                9, 13, 14, 19, 24 ,24,
                0, 13, 17 ,22, 23 ,0
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];

        // Rotate the image
        imgNativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);

        // Compare the output image with the expected image
        assertArrayEquals(expected_img_out, img_out, 0.0001);
    }

    @Test
    public void testImageRotation180() {
        // Input image dimensions
        int rows = 4;
        int cols = 4;

        // Input image
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        // Rotation angle in radians
        double radians = Math.PI;

        // Fill value for the output image
        double fill_value = 0.0;

        // Expected output image
        double[] expected_img_out = {
                16.0, 15.0, 14.0, 13.0,
                12.0, 11.0, 10.0, 9.0,
                8.0, 7.0, 6.0, 5.0,
                4.0, 3.0, 2.0, 1.0
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];

        // Rotate the image
        imgNativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);

        // Compare the output image with the expected image
        assertArrayEquals(expected_img_out, img_out, 0.0001);
    }

    @Test
    public void testCutoutImage() {
        // Example input 2D matrix
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        int rows = 4;
        int cols = 4;
        int x = 2;
        int y = 2;
        int width = 3;
        int height = 3;
        double fill_value = 0.0;

        // Perform image cutout using JNI
        double[] img_out = imgNativeHelper.imageCutout(img_in, rows, cols, x, y, width, height, fill_value);

        // Expected output image after cutout
        double[] expectedOutput = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 0.0, 0.0, 0.0,
                9.0, 0.0, 0.0, 0.0,
                13.0, 0.0, 0.0, 0.0
        };

        // Check if the output image matches the expected output
        assertArrayEquals(expectedOutput, img_out,0.0001);
    }
    @Test
    public void testImageCutoutInvalidCutout() {
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        int rows = 4;
        int cols = 4;
        int x = 3;
        int y = 3;
        int width = -2;
        int height = 0;
        double fill_value = 0.0;

        double[] expectedOutput = img_in; // Expect no change since the cutout is invalid

        double[] img_out = imgNativeHelper.imageCutout(img_in, rows, cols, x, y, width, height, fill_value);
        assertArrayEquals(expectedOutput, img_out,0.0001);
    }

    @Test
    public void testImageCutoutNoCutout() {
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        int rows = 4;
        int cols = 4;
        int x = 3;
        int y = 3;
        int width = 1;
        int height = 1;
        double fill_value = 0.0;

        double[] expectedOutput = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 0.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        double[] img_out = imgNativeHelper.imageCutout(img_in, rows, cols, x, y, width, height, fill_value);
        assertArrayEquals(expectedOutput, img_out,0.0001);
    }

    @Test
    public void testImageCropValidCrop() {
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        int orig_w = 4;
        int orig_h = 4;
        int w = 2;
        int h = 2;
        int x_offset = 1;
        int y_offset = 1;

        double[] expectedOutput = {
                6.0, 7.0,
                10.0, 11.0
        };

        double[] img_out = imgNativeHelper.cropImage(img_in, orig_w, orig_h, w, h, x_offset, y_offset);

        assertArrayEquals(expectedOutput, img_out,0.0001);     }

    @Test
    public void testImageCropInvalidCrop() {
        double[] img_in = {
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
        };

        int orig_w = 4;
        int orig_h = 4;
        int w = 5;
        int h = 5;
        int x_offset = 1;
        int y_offset = 1;

        double[] expectedOutput = img_in; // Expect no change since the crop is invalid

        double[] img_out = imgNativeHelper.cropImage(img_in, orig_w, orig_h, w, h, x_offset, y_offset);
        assertArrayEquals(expectedOutput, img_out,0.0001);
    }

    @Test
    public void testImgTranslate() {
        int in_w = 5;
        int in_h = 5;
        int out_w = 7;
        int out_h = 7;
        double fill_value = 0.0;

        double[] img_in = new double[in_w * in_h];
        for (int i = 0; i < in_w * in_h; ++i) {
            img_in[i] = i + 1; // Filling input image with sequential values
        }

        double[] img_out = new double[out_w * out_h];

        double offset_x = 1.5;
        double offset_y = 1.5;

        imgNativeHelper.imgTranslate(img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, img_out);

        // Expected output based on the given offsets and fill value
        double[] expectedOutput = {
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,1,2,3,4,5,
                0,0,6,7,8,9,10,
                0,0,11,12,13,14,15,
                0,0,16,17,18,19,20,
                0,0,21,22,23,24,25
        };

        assertArrayEquals(expectedOutput, img_out, 1e-9); // Compare arrays with a small epsilon
    }

    @Test
    public void testImgTranslateNegativeOffsets() {
        int in_w = 5;
        int in_h = 5;
        int out_w = 6;
        int out_h = 6;
        double fill_value = 0.0;

        double[] img_in = new double[in_w * in_h];
        for (int i = 0; i < in_w * in_h; ++i) {
            img_in[i] = i + 1; // Filling input image with sequential values
        }

        double[] img_out = new double[out_w * out_h];

        double offset_x = -0.5; // Negative offset in X direction
        double offset_y = -0.5; // Negative offset in Y direction

        imgNativeHelper.imgTranslate(img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, img_out);

        // Expected output based on the given offsets and fill value
        double[] expectedOutput = {
                1,2,3,4,5,0,
                6,7,8,9,10,0,
                11,12,13,14,15,0,
                16,17,18,19,20,0,
                21,22,23,24,25,0,
                0,0,0,0,0,0,
        };

        assertArrayEquals(expectedOutput, img_out, 1e-9); // Compare arrays with a small epsilon
    }

    @Test
    public void testImgTranslatePositiveAndNegativeOffsets() {
        int in_w = 5;
        int in_h = 5;
        int out_w = 6;
        int out_h = 6;
        double fill_value = 0.0;

        double[] img_in = new double[in_w * in_h];
        for (int i = 0; i < in_w * in_h; ++i) {
            img_in[i] = i + 1; // Filling input image with sequential values
        }

        double[] img_out = new double[out_w * out_h];

        double offset_x = -0.5; // Negative offset in X direction
        double offset_y = 0.5; // Negative offset in Y direction

        imgNativeHelper.imgTranslate(img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, img_out);

        // Expected output based on the given offsets and fill value
        double[] expectedOutput = {
                0,0,0,0,0,0,
                1,2,3,4,5,0,
                6,7,8,9,10,0,
                11,12,13,14,15,0,
                16,17,18,19,20,0,
                21,22,23,24,25,0,
        };

        assertArrayEquals(expectedOutput, img_out, 1e-9); // Compare arrays with a small epsilon
    }

    @Test
    public void testImgTranslatePositiveAndNegativeOffsets2() {
        int in_w = 5;
        int in_h = 5;
        int out_w = 6;
        int out_h = 6;
        double fill_value = 0.0;

        double[] img_in = new double[in_w * in_h];
        for (int i = 0; i < in_w * in_h; ++i) {
            img_in[i] = i + 1; // Filling input image with sequential values
        }

        double[] img_out = new double[out_w * out_h];

        double offset_x = 0.5; // Negative offset in X direction
        double offset_y = -0.5; // Negative offset in Y direction

        imgNativeHelper.imgTranslate(img_in, offset_x, offset_y, in_w, in_h, out_w, out_h, fill_value, img_out);

        // Expected output based on the given offsets and fill value
        double[] expectedOutput = {
                0,1,2,3,4,5,
                0,6,7,8,9,10,
                0,11,12,13,14,15,
                0,16,17,18,19,20,
                0,21,22,23,24,25,
                0,0,0,0,0,0,
        };

        assertArrayEquals(expectedOutput, img_out, 1e-9); // Compare arrays with a small epsilon
    }

}
