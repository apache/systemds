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
			fft_one_dim(re, im, re_inter, im_inter, i*cols, 1, cols, cols);
		}

		for(int j = 0; j < cols; j++){
			fft_one_dim(re, im, re_inter, im_inter, j, cols, rows, rows*cols);
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
			ifft_one_dim(re, im, re_inter, im_inter, j, cols, rows, rows*cols);
		}

		for(int i = 0; i < rows; i++){
			ifft_one_dim(re, im, re_inter, im_inter, i*cols, 1, cols, cols);
		}

	}

	private static void fft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int step, int num, int subArraySize) {

		if(num == 1) return;
		double angle = -2*FastMath.PI/num;

		// fft for even indices
		fft_one_dim(re, im, re_inter, im_inter, start, step*2, num/2, subArraySize);
		// fft for odd indices
		fft_one_dim(re, im, re_inter, im_inter,start+step, step*2, num/2, subArraySize);

		// iterates over even indices
		for(int j = start, cnt = 0; cnt < num/2; j+=(2*step), cnt++){

			double omega_pow_re = FastMath.cos(cnt*angle);
			double omega_pow_im = FastMath.sin(cnt*angle);

			// calculate m using the result of odd index
			double m_re = omega_pow_re * re[j+step] - omega_pow_im * im[j+step];
			double m_im = omega_pow_re * im[j+step] + omega_pow_im * re[j+step];

			int index = start+cnt*step;
			re_inter[index] = re[j] + m_re;
			re_inter[index+subArraySize/2] = re[j] - m_re;

			im_inter[index] = im[j] + m_im;
			im_inter[index+subArraySize/2] = im[j] - m_im;
		}

		for(int j = start; j < start+subArraySize; j+=step){
			re[j] = re_inter[j];
			im[j] = im_inter[j];
			re_inter[j] = 0;
			im_inter[j] = 0;
		}

	}

	private static void ifft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int step, int num, int subArraySize) {

		// conjugate input
		for (int i = start; i < start+num*step; i+=step){
			im[i] = -im[i];
		}

		// apply fft
		fft_one_dim(re, im, re_inter, im_inter, start, step, num, subArraySize);

		// conjugate and scale result
		for (int i = start; i < start+num*step; i+=step){
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
