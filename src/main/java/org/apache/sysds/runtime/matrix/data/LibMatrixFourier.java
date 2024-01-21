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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.math3.util.FastMath;

public class LibMatrixFourier {

	/**
	 * Function to perform FFT on two given matrices.
	 * The first one represents the real values and the second one the imaginary values.
	 * The output also contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re matrix object representing the real values
	 * @param im matrix object representing the imaginary values
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft(MatrixBlock re, MatrixBlock im){

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(rows) || !isPowerOfTwo(cols)) throw new RuntimeException("false dimensions");

		fft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols);

		return new MatrixBlock[]{re, im};
	}

	/**
	 * Function to perform IFFT on two given matrices.
	 * The first one represents the real values and the second one the imaginary values.
	 * The output also contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re matrix object representing the real values
	 * @param im matrix object representing the imaginary values
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft(MatrixBlock re, MatrixBlock im){

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(rows) || !isPowerOfTwo(cols)) throw new RuntimeException("false dimensions");

		ifft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols);

		return new MatrixBlock[]{re, im};
	}

	/**
	 * Function to perform FFT on two given double arrays.
	 * The first one represents the real values and the second one the imaginary values.
	 * Both arrays get updated and contain the result.
	 *
	 * @param re array representing the real values
	 * @param im array representing the imaginary values
	 * @param rows number of rows
	 * @param cols number of rows
	 */
	public static void fft(double[] re, double[] im, int rows, int cols) {

		double[] re_inter = new double[rows*cols];
		double[] im_inter = new double[rows*cols];

		for(int i = 0; i < rows; i++){
			fft_one_dim(re, im, re_inter, im_inter, i*cols, (i+1)*cols, cols, 1);
		}

		for(int j = 0; j < cols; j++){
			fft_one_dim(re, im, re_inter, im_inter, j, j+rows*cols, rows, cols);
		}

	}

	/**
	 * Function to perform IFFT on two given double arrays.
	 * The first one represents the real values and the second one the imaginary values.
	 * Both arrays get updated and contain the result.
	 *
	 * @param re array representing the real values
	 * @param im array representing the imaginary values
	 * @param rows number of rows
	 * @param cols number of rows
	 */
	public static void ifft(double[] re, double[] im, int rows, int cols) {

		double[] re_inter = new double[rows*cols];
		double[] im_inter = new double[rows*cols];

		for(int j = 0; j < cols; j++){
			ifft_one_dim(re, im, re_inter, im_inter, j, j+rows*cols, rows, cols);

		}

		for(int i = 0; i < rows; i++){
			ifft_one_dim(re, im, re_inter, im_inter, i*cols, (i+1)*cols, cols, 1);
		}

	}

	public static void fft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int stop, int num, int minStep) {

		// start inclusive, stop exclusive
		if(num == 1) return;

		// even indices
		for(int step = minStep*(num/2), subNum = 2; subNum <= num; step/=2, subNum*=2){

			double angle = -2*FastMath.PI/subNum;

			// use ceil for the main (sub)array
			for(int sub = 0; sub < FastMath.ceil(num/(2*(double)subNum)); sub++){

				for(int isOdd = 0; isOdd < 2; isOdd++) {

					// no odd values for main (sub)array
					if (isOdd == 1 && subNum == num) return;

					int startSub = start + sub*minStep + isOdd*(step/2);

					// first iterates over even indices, then over odd indices
					for (int j = startSub, cnt = 0; cnt < subNum / 2; j += 2*step, cnt++) {

						double omega_pow_re = FastMath.cos(cnt * angle);
						double omega_pow_im = FastMath.sin(cnt * angle);

						// calculate m using the result of odd index
						double m_re = omega_pow_re * re[j + step] - omega_pow_im * im[j + step];
						double m_im = omega_pow_re * im[j + step] + omega_pow_im * re[j + step];

						int index = startSub + cnt * step;
						re_inter[index] = re[j] + m_re;
						re_inter[index + (stop-start)/2] = re[j] - m_re;

						im_inter[index] = im[j] + m_im;
						im_inter[index + (stop-start)/2] = im[j] - m_im;
					}

					for (int j = startSub; j < startSub + (stop-start); j += step) {
						re[j] = re_inter[j];
						im[j] = im_inter[j];
						re_inter[j] = 0;
						im_inter[j] = 0;
					}

				}

			}

		}

	}

	private static void ifft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int stop, int num, int minStep) {

		// conjugate input
		for (int i = start; i < start+num*minStep; i+=minStep){
			im[i] = -im[i];
		}

		// apply fft
		//fft_one_dim_recursive(re, im, re_inter, im_inter, start, step, num, subArraySize);
		fft_one_dim(re, im, re_inter, im_inter, start, stop, num, minStep);

		// conjugate and scale result
		for (int i = start; i < start+num*minStep; i+=minStep){
			re[i] = re[i]/num;
			im[i] = -im[i]/num;
		}

	}

	/**
	 * Function for checking whether the given integer is a power of two.
	 *
	 * @param n integer to check
	 * @return true if n is a power of two, false otherwise
	 */
	public static boolean isPowerOfTwo(int n){
		return (n != 0) && ((n & (n - 1)) == 0);
	}

	/**
	 * Function to perform FFT on a given matrices.
	 * The given matrix only represents real values.
	 * The output contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re matrix object representing the real values
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft(MatrixBlock re){
		return fft(re, new MatrixBlock(re.getNumRows(),re.getNumColumns(), new double[re.getNumRows()*re.getNumColumns()]));
	}

	/**
	 * Function to perform IFFT on a given matrices.
	 * The given matrix only represents real values.
	 * The output contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re matrix object representing the real values
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft(MatrixBlock re){
		return ifft(re, new MatrixBlock(re.getNumRows(),re.getNumColumns(), new double[re.getNumRows()*re.getNumColumns()]));
	}

}
