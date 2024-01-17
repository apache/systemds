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

import static org.junit.Assert.assertArrayEquals;
import static org.apache.sysds.runtime.matrix.data.LibMatrixSTFT.stft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixSTFT.one_dim_stft;

public class LibMatrixSTFTTest {

    @Test
    public void simple_test() {

        double[] signal = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int windowSize = 4;
        int overlap = 2;
        double[][] stftResult = one_dim_stft(signal, windowSize, overlap);

        // 1st row real part, 2nd row imaginary part
        double[][] expected = {{6, -2, -2, -2, 14, -2, -2, -2, 22, -2, -2, -2, 30, -2, -2, -2, 38, -2, -2, -2, 46, -2, -2, -2, 54, -2, -2, -2},{0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2}};

        for(double[] row : stftResult){
            for (double elem : row){
                System.out.print(elem + " ");
            }
            System.out.println();
        }

        assertArrayEquals(expected[0], stftResult[0], 0.0001);
        assertArrayEquals(expected[1], stftResult[1], 0.0001);

    }

    @Test
    public void matrix_block_one_dim_test(){

        double[] in = {0, 18, -15, 3};

        double[] expected_re = {6,15,-36,15};
        double[] expected_im = {0,-15,0,15};

        MatrixBlock[] res = stft(in, 4, 0);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        for(double elem : res_re){
            System.out.print(elem+" ");
        }
        System.out.println();
        for(double elem : res_im){
            System.out.print(elem+" ");
        }

        assertArrayEquals(expected_re, res_re, 0.0001);
        assertArrayEquals(expected_im, res_im, 0.0001);

    }

    @Test
    public void matrix_block_one_dim_test2(){

        double[] in = {10, 5, -3, 8, 15, -6, 2, 0};

        double[] expected_re = {20.0, 13.0, -6.0, 13.0, 14.0, -18.0, 10.0, -18.0, 11.0, 13.0, 23.0, 13.0};
        double[] expected_im = {0.0, 3.0, 0.0, -3.0, 0.0, -14.0, 0.0, 14.0, 0.0, 6.0, 0.0, -6.0 };

        MatrixBlock[] res = stft(in, 4, 2);
        double[] res_re = res[0].getDenseBlockValues();
        double[] res_im = res[1].getDenseBlockValues();

        for(double elem : res_re){
            System.out.print(elem+" ");
        }
        System.out.println();
        for(double elem : res_im){
            System.out.print(elem+" ");
        }

        assertArrayEquals(expected_re, res_re, 0.0001);
        assertArrayEquals(expected_im, res_im, 0.0001);

    }

}
