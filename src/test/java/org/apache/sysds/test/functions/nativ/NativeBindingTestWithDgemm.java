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

import java.util.Random;
import org.apache.sysds.utils.NativeHelper;
import org.junit.Assert;
import org.junit.Test;

public class NativeBindingTestWithDgemm {
    static {
       System.loadLibrary("systemds_openblas-Darwin-x86_64");
    }
    // Helper method to flatten a 2D matrix into a 1D array
    private double[] flattenMatrix(double[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        double[] flattened = new double[rows * columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                flattened[i * columns + j] = matrix[i][j];
            }
        }

        return flattened;
    }

    // Helper method to generate a random matrix
    private double[][] generateRandomMatrix(int rows, int columns) {
        Random random = new Random();
        double[][] matrix = new double[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }

        return matrix;
    }

    public void testImageRotation() {
        // Input image dimensions
        int rows = 4;
        int cols = 4;

        // Input image
        double[] img_in = {
                1, 1, 1, 0,
                0, 3, 1, 2,
                2, 3, 1, 0,
                1, 0, 2, 1
        };

        // Rotation angle in radians
        double radians = Math.PI / 2;

        // Fill value for the output image
        double fill_value = 0.0;

        // Expected output image
        double[] expected_img_out = {
                0, 2, 0, 1,
                1, 1, 1, 2,
                1, 3, 3, 0,
                1, 0, 2, 1
        };

        // Create the output image array
        double[] img_out = new double[rows * cols];

        // Rotate the image
        NativeHelper.imageRotate(img_in, rows, cols, radians, fill_value, img_out);

        // Compare the output image with the expected image
        Assert.assertArrayEquals(expected_img_out, img_out, 0.0001);
    }

    // Method to test the dgemm function
    public void testDgemm() {
        char transa = 'N';
        char transb = 'N';
        int m = 2000;
        int n = 100;
        int k = 1000;
        double alpha = 1.0;
        double beta = 0.0;

        // Generate random input matrices A, B, and C
        double[][] A = generateRandomMatrix(m, k);
        double[][] B = generateRandomMatrix(k, n);
        double[][] C = new double[m][n];

        // Convert matrices to 1D arrays
        double[] flatA = flattenMatrix(A);
        double[] flatB = flattenMatrix(B);
        double[] flatC = flattenMatrix(C);

        // Call the native dgemm method
        long startTime = System.currentTimeMillis();
        NativeHelper.testNativeBindingWithDgemm(transa, transb, m, n, k, alpha, flatA, k, flatB, n, beta, flatC, n);
        long endTime = System.currentTimeMillis();
        long executionTime = (endTime - startTime);



        // Print the result matrix C
        /*System.out.println("Result Matrix C:");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(flatC[i * n + j] + " ");
            }
            System.out.println();
        }*/
        System.out.println("Execution time: " + executionTime + " ms");
    }

    public static void main(String[] args) {
        new NativeBindingTestWithDgemm().testDgemm();
        new NativeBindingTestWithDgemm().testImageRotation();
    }
}
