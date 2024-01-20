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

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

public class FourierTest {

	@Test
	public void test_fft_one_dim() {

		MatrixBlock re = new MatrixBlock(1, 4,  new double[]{0, 18, -15, 3});
		MatrixBlock im = new MatrixBlock(1, 4,  new double[4]);

		double[] expected_re = {6, 15, -36, 15};
		double[] expected_im = {0, -15, 0, 15};

		fft(re, im);

		assertArrayEquals(expected_re, re.getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, im.getDenseBlockValues(), 0.0001);

	}

	@Test
	public void test_fft_one_dim_2() {

		MatrixBlock re = new MatrixBlock(1, 8,  new double[]{0, 18, -15, 3, 5, 10, 5, 9});
		MatrixBlock im = new MatrixBlock(1, 8,  new double[8]);

		double[] expected_re = {35, 4.89949, 15, -14.89949, -45, -14.89949, 15, 4.89949};
		double[] expected_im = {0, 18.58579, -16, -21.41421, 0, 21.41421, 16, -18.58579};

		fft(re, im);

		assertArrayEquals(expected_re, re.getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, im.getDenseBlockValues(), 0.0001);
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

	@Test
	public void failed_test_ifft_with_real_numpy_data() {

		// removed 0's at the end
		MatrixBlock re = new MatrixBlock(1, 16,  new double[]{
				0.5398705320215192, 0.1793355360736929, 0.9065254044489506, 0.45004385530909075,
				0.11128090341119468, 0.11742862805303522, 0.7574827475752481, 0.2778193170985158,
				0.9251562928110273, 0.9414429667551927, 0.45131569795507087, 0.9522067687409731,
				0.22491032260636257, 0.6579426733967295, 0.7021558730366062, 0.7861117825617701,
		});

		MatrixBlock im = new MatrixBlock(1, 16,  new double[16]);

		double[] expected_re = {
				0.5613143313659362, -0.020146500453061336, 0.07086545895481336, -0.05003801442765281,
				-0.06351635451036074, -0.03346768844048936, 0.07023899089706032, 0.007330763123826495,
				0.016022890367311193, 0.007330763123826495, 0.07023899089706032, -0.03346768844048936,
				-0.06351635451036074, -0.05003801442765281, 0.07086545895481336, -0.020146500453061336
		};

		double[] expected_im = {
				0.0, -0.07513090216687965, 0.023854392878864396, -0.018752997939582024,
				-0.035626994964481226, -0.07808216015046443, 0.036579082654843526, -0.10605270957897009,
				0.0, 0.10605270957897009, -0.036579082654843526, 0.07808216015046443,
				0.035626994964481226, 0.018752997939582024, -0.023854392878864396, 0.07513090216687965
		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void failed_test_ifft_2d_with_generated_data() {

		// TODO:

		MatrixBlock re = new MatrixBlock(2, 2,  new double[]{
				0.6749989259154331, 0.6278845555308362, 0.995916990652601, 0.511472971081564
		});

		MatrixBlock im = new MatrixBlock(2, 2,  new double[]{
				0.8330832079105173, 0.09857986129294982, 0.6808883894146879, 0.28353782431047303
		});

		double[] expected_re = {
				2.81027344, 0.53155839, -0.20450648, -0.43732965
		};

		double[] expected_im = {
				1.89608928, 1.13185391, -0.03276314, 0.33715278
		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

	@Test
	public void failed_test_ifft_with_complex_numpy_data() {

		MatrixBlock re = new MatrixBlock(1, 16,  new double[]{
				0.17768499045697306, 0.3405035491673728, 0.9272906450602005, 0.28247504052271843,
				0.42775517613102865, 0.8338783039818357, 0.9813749624557385, 0.47224489612381737,
				0.7936831995784907, 0.5584182145651306, 0.5296113722056018, 0.6687593295928902,
				0.9630598447622862, 0.7130539473424196, 0.860081483892192, 0.8985058305053549
		});

		MatrixBlock im = new MatrixBlock(1, 16,  new double[16]);

		// adjusted the expected output
		double[] expected_re = {
				0.6517738, -0.0263837 , -0.03631354, -0.01644966,
				-0.05851095, -0.0849794, -0.01611732, -0.02618679,
				0.05579391, -0.02618679, -0.01611732, -0.0849794,
				-0.05851095, -0.01644966, -0.03631354, -0.0263837
		};

		double[] expected_im = {
				0, -0.04125649, -0.07121312, 0.02554502,
				0.00774181, -0.08723921, -0.02314382, -0.02021455,
				0, 0.02021455, 0.02314382, 0.08723921,
				-0.00774181, -0.02554502, 0.07121312, 0.04125649

		};

		MatrixBlock[] res = ifft(re, im);

		assertArrayEquals(expected_re, res[0].getDenseBlockValues(), 0.0001);
		assertArrayEquals(expected_im, res[1].getDenseBlockValues(), 0.0001);

	}

}
