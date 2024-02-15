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

public class STFTTest {

	@Test
	public void simple_test() {

		MatrixBlock re = new MatrixBlock(1, 16,  new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

		MatrixBlock[] res = stft(re, 4, 2);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6, -2, -2, -2, 14, -2, -2, -2, 22, -2, -2, -2, 30, -2, -2, -2, 38, -2, -2, -2, 46, -2, -2, -2, 54, -2, -2, -2};
		double[] expected_im = {0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void simple_test_two() {

		MatrixBlock re = new MatrixBlock(1, 15,  new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});

		MatrixBlock[] res = stft(re, 4, 2);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6, -2, -2, -2, 14, -2, -2, -2, 22, -2, -2, -2, 30, -2, -2, -2, 38, -2, -2, -2, 46, -2, -2, -2, 39, -2, 13, -2};
		double[] expected_im = {0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, -13, 0, 13};

		/*
		for (int i = 0; i < res_re.length; i++) {
			System.out.println(res_re[i] + " " + res_im[i]);
		}

		 */
		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void matrix_block_one_dim_test(){

		MatrixBlock re = new MatrixBlock(1, 4,  new double[]{0, 18, -15, 3});

		MatrixBlock[] res = stft(re, 4, 0);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6, 15, -36, 15};
		double[] expected_im = {0, -15, 0, 15};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void matrix_block_one_dim_test2(){

		MatrixBlock re = new MatrixBlock(1, 8,  new double[]{10, 5, -3, 8, 15, -6, 2, 0});

		MatrixBlock[] res = stft(re, 4, 2);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {20.0, 13.0, -6.0, 13.0, 14.0, -18.0, 10.0, -18.0, 11.0, 13.0, 23.0, 13.0};
		double[] expected_im = {0.0, 3.0, 0.0, -3.0, 0.0, -14.0, 0.0, 14.0, 0.0, 6.0, 0.0, -6.0};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void matrix_block_one_dim_test3(){

		MatrixBlock re = new MatrixBlock(1, 8,  new double[]{10, 5, -3, 8, 15, -6, 2, 0});
		MatrixBlock im = new MatrixBlock(1, 8,  new double[]{0, 0, 0, 0, 0, 0, 0, 0});

		MatrixBlock[] res = stft(re, im, 4, 2);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {20.0, 13.0, -6.0, 13.0, 14.0, -18.0, 10.0, -18.0, 11.0, 13.0, 23.0, 13.0};
		double[] expected_im = {0.0, 3.0, 0.0, -3.0, 0.0, -14.0, 0.0, 14.0, 0.0, 6.0, 0.0, -6.0};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void test_two_x_eight() {

		MatrixBlock re = new MatrixBlock(2, 8,  new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

		MatrixBlock[] res = stft(re, 4, 1);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6, -2, -2, -2, 18, -2, -2, -2, 13, 6, -1, 6, 38, -2, -2, -2, 50, -2, -2, -2, 29, 14, -1, 14};
		double[] expected_im = {0, 2, 0, -2, 0, 2, 0, -2, 0, -7, 0, 7, 0, 2, 0, -2, 0, 2, 0, -2, 0, -15, 0, 15};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void test_four_x_four() {

		MatrixBlock re = new MatrixBlock(4, 4,  new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

		MatrixBlock[] res = stft(re, 4, 0);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6, -2, -2, -2, 22, -2, -2, -2, 38, -2, -2, -2, 54, -2, -2, -2};
		double[] expected_im = {0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

	@Test
	public void test_four_x_four_two() {

		MatrixBlock re = new MatrixBlock(4, 5,  new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});

		MatrixBlock[] res = stft(re, 4, 1);

		double[] res_re = res[0].getDenseBlockValues();
		double[] res_im = res[1].getDenseBlockValues();

		double[] expected_re = {6.0, -2.0, -2.0, -2.0, 7.0, 3.0, -1.0, 3.0, 26.0, -2.0, -2.0, -2.0, 17.0, 8.0, -1.0, 8.0, 46.0, -2.0, -2.0, -2.0, 27.0, 13.0, -1.0, 13.0, 66.0, -2.0, -2.0, -2.0, 37.0, 18.0, -1.0, 18.0};
		double[] expected_im = {0.0, 2.0, 0.0, -2.0, 0.0, -4.0, 0.0, 4.0, 0.0, 2.0, 0.0, -2.0, 0.0, -9.0, 0.0, 9.0, 0.0, 2.0, 0.0, -2.0, 0.0, -14.0, 0.0, 14.0, 0.0, 2.0, 0.0, -2.0, 0.0, -19.0, 0.0, 19.0};

		assertArrayEquals(expected_re, res_re, 0.0001);
		assertArrayEquals(expected_im, res_im, 0.0001);

	}

}
