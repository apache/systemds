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

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.ImgNativeHelper;
import org.junit.Test;

import java.util.Random;

import static org.apache.sysds.utils.ImgNativeHelper.*;

public class PerformanceComparison extends AutomatedTestBase {

    private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + PerformanceComparison.class.getSimpleName() + "/";

    private final static double eps = 1e-10;
    private final static int rows = 512;
    private final static int cols = 512;
    private final static double spSparse = 0.1;
    private final static double spDense = 0.9;
    private final static int x_offset = 12;
    private final static int y_offset = 24;
    private final static float size = 0.8f;

    private final ImgNativeHelper imgNativeHelper = new ImgNativeHelper("openblas");

    @Override
    public void setUp() {
      //  addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
    }

    private static double randomDouble(double min, double max) {
        return min + (max - min) * new Random().nextDouble();
    }

    // Function to generate a random square matrix of size n x n
    public static double[] generateRandomMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
        double[] matrix = new double[rows * cols];
        Random random = (seed == -1) ? TestUtils.random : new Random(seed);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int index = i * cols + j; // Calculate the index for the 1D array
                if (random.nextDouble() > sparsity) {
                    continue;
                }
                matrix[index] = (random.nextDouble() * (max - min) + min);
            }
        }

        return matrix;
    }

    // Function to print a square matrix
    private static void printMatrix(double[] matrix, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%.4f\t", matrix[i * n + j]);
            }
            System.out.println();
        }
    }

    public void runBlasTests(boolean sparse,int n, int seed) {
    double spSparse = 0.1;
    double spDense = 0.9;
    double sparsity = sparse ? spSparse: spDense;

    double[] img_in = PerformanceComparison.generateRandomMatrix(n, n, 0, 255, sparsity, seed);
    double radians = Math.PI / 4.0;

    // Benchmark imageRotate
    double[] img_rotated = new double[n * n];
    long startTime = System.currentTimeMillis();

    imgNativeHelper.imageRotate(img_in, n, n, radians, 0.0,img_rotated);

    long endTime = System.currentTimeMillis();
        System.out.println("image_rotate\nTotal execution time:"+(endTime -startTime));

    // Benchmark imageCutout
    int x = 100;
    int y = 100;
    int width = 200;
    int height = 200;
    startTime =System.currentTimeMillis();
    double[] img_cutout = imgNativeHelper.imageCutout(img_in, n, n, x, y, width, height, 0.0);
    endTime =System.currentTimeMillis();
        System.out.println("image_cutout\nTotal execution time:"+(endTime -startTime));

    // Benchmark cropImage
    int orig_w = 512;
    int orig_h = 512;
    int crop_w = 256;
    int crop_h = 256;
    int x_offset = 128;
    int y_offset = 128;
    startTime =System.currentTimeMillis();
    double[] cropped_img = imgNativeHelper.cropImage(img_in, orig_w, orig_h, crop_w, crop_h, x_offset, y_offset);
    endTime =System.currentTimeMillis();
        System.out.println("image_crop\nTotal execution time:"+(endTime -startTime));

    // Benchmark imgTranslate
    int in_w = 512;
    int in_h = 512;
    int out_w = 512;
    int out_h = 512;
    double[] img_translated = new double[out_w * out_h];
    startTime =System.currentTimeMillis();

    imgNativeHelper.imgTranslate(img_in, 50.0,50.0,in_w, in_h, out_w, out_h, 0.0,img_translated);

    endTime =System.currentTimeMillis();
        System.out.println("image_translate\nTotal execution time:"+(endTime -startTime));
    }


    public void runDMLTests(boolean sparse,int seed)
    {
        String[] testNames = {"image_rotate","image_cutout","image_crop","image_translate"};
        for(String TEST_NAME : testNames) {
            System.out.println(TEST_NAME+"\n");
            addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));

            try
            {
                loadTestConfiguration(getTestConfiguration(TEST_NAME));
                double sparsity = sparse ? spSparse : spDense;
                String HOME = SCRIPT_DIR + TEST_DIR;
                fullDMLScriptName = HOME + TEST_NAME + ".dml";
                programArgs = new String[]{"-stats","-nvargs",
                        "in_file=" + input("A"), "out_file=" + output("B"),
                        "size=" + size, "x_offset=" + x_offset, "y_offset=" + y_offset, "width=" + cols, "height=" + rows
                };

                //generate actual dataset
                double[][] A = getRandomMatrix(rows, cols, 0, 255, sparsity, seed);
                writeInputMatrixWithMTD("A", A, true);

                runTest(true, false, null, -1);

            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }


    public void benchmarkDMLImgImplementations() {
        for(int i = 0; i < 100; i ++) {
            runDMLTests(true,i);
        }

    }


    public void benchmarkBlasImgImplementations() {
        for(int i = 0; i < 100; i ++) {
            runBlasTests(true,512,7);
        }

    }

}
