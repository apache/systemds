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

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft_one_dim;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class FourierTest {

    @Test
    public void test_fft_one_dim() {

        double[] re = {0, 18, -15, 3};
        double[] im = {0, 0, 0, 0};

        double[] re_inter = new double[4];
        double[] im_inter = new double[4];

        double[] expected_re = {6, 15, -36, 15};
        double[] expected_im = {0, -15, 0, 15};

        int cols = 4;

        fft_one_dim(re, im, re_inter, im_inter, 0, 1, cols, cols);

        assertArrayEquals(expected_re, re, 0.0001);
        assertArrayEquals(expected_im, im, 0.0001);

    }

    @Test
    public void test_fft_one_dim_2() {

        double[] re = {0, 18, -15, 3, 5, 10, 5, 9};
        double[] im = new double[8];

        double[] re_inter = new double[8];
        double[] im_inter = new double[8];

        double[] expected_re = {35, 4.89949, 15, -14.89949, -45, -14.89949, 15, 4.89949};
        double[] expected_im = {0, 18.58579, -16, -21.41421, 0, 21.41421, 16, -18.58579};

        int cols = 8;

        fft_one_dim(re, im, re_inter, im_inter, 0, 1, cols, cols);

        assertArrayEquals(expected_re, re, 0.0001);
        assertArrayEquals(expected_im, im, 0.0001);
    }

    @Test
    public void test_fft_one_dim_matrixBlock() {

        MatrixBlock re = new MatrixBlock(1, 4,  new double[]{0, 18, -15, 3});
        MatrixBlock im = new MatrixBlock(1, 4,  new double[]{0, 0, 0, 0});

        double[] expected_re = {6, 15, -36, 15};
        double[] expected_im = {0, -15, 0, 15};

        MatrixBlock[] res = fft(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

    @Test
    public void test_ifft_one_dim_matrixBlock_2() {

        double[] in_re = new double[]{1, -2, 3, -4};
        double[] in_im = new double[]{0, 0, 0, 0};

        MatrixBlock re = new MatrixBlock(1, 4, in_re);
        MatrixBlock im = new MatrixBlock(1, 4, in_im);

        MatrixBlock[] inter = fft(re, im);
        MatrixBlock[] res = ifft(inter[0], inter[1]);

        assertArrayEquals(in_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(in_im, res[1].getDenseBlockValues(), 0.0001);

    }

    @Test
    public void test_fft_two_dim_matrixBlock() {

        MatrixBlock re = new MatrixBlock(2, 2,  new double[]{0, 18, -15, 3});
        MatrixBlock im = new MatrixBlock(2, 2,  new double[]{0, 0, 0, 0});

        double[] expected_re = {6,-36, 30, 0};
        double[] expected_im = {0, 0, 0, 0};

        MatrixBlock[] res = fft(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

    @Test
    public void test_ifft_two_dim_matrixBlock() {

        MatrixBlock re = new MatrixBlock(2, 2,  new double[]{6,-36, 30, 0});
        MatrixBlock im = new MatrixBlock(2, 2,  new double[]{0, 0, 0, 0});

        double[] expected_re = {0, 18, -15, 3};
        double[] expected_im = {0, 0, 0, 0};

        MatrixBlock[] res = ifft(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

    @Test
    public void test_fft_two_dim_matrixBlock_row_1() {

        MatrixBlock re = new MatrixBlock(1, 2,  new double[]{0, 18});
        MatrixBlock im = new MatrixBlock(1, 2,  new double[]{0, 0});

        double[] expected_re = {18, -18};
        double[] expected_im = {0, 0};

        MatrixBlock[] res = fft(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

    @Test
    public void test_fft_two_dim_matrixBlock_row_2() {

        MatrixBlock re = new MatrixBlock(1, 2,  new double[]{ -15, 3});
        MatrixBlock im = new MatrixBlock(1, 2,  new double[]{0, 0});

        double[] expected_re = {-12, -18};
        double[] expected_im = {0, 0};

        MatrixBlock[] res = fft(re, im);

        assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
        assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

    }

}
