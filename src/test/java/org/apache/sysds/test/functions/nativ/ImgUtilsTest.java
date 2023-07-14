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

import org.apache.sysds.utils.NativeHelper;
import org.junit.Assert;
import org.junit.Test;

public class ImgUtilsTest {

    static {
        System.loadLibrary("systemds_mkl-Darwin-x86_64");
    }
    @Test
    public void testImageRotation90() {
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
        double radians = Math.PI / 2;

        // Fill value for the output image
        double fill_value = 0.0;

        // Expected output image
        double[] expected_img_out = {
                4.0, 8.0, 12.0, 16.0,
                3.0, 7.0, 11.0, 15.0,
                2.0, 6.0, 10.0, 14.0,
                1.0, 5.0, 9.0, 13.0
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];
        NativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);
        Assert.assertArrayEquals(expected_img_out, img_out, 0.0001);
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
        NativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);

        // Compare the output image with the expected image
        Assert.assertArrayEquals(expected_img_out, img_out, 0.0001);
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
        NativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);

        // Compare the output image with the expected image
        Assert.assertArrayEquals(expected_img_out, img_out, 0.0001);
    }
}
