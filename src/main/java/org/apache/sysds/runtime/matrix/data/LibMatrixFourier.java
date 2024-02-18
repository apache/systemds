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

import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public class LibMatrixFourier {

	/**
	 * Function to perform FFT for two given matrices. The first one represents the real values and the second one the
	 * imaginary values. The output also contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param im      matrix object representing the imaginary values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft(MatrixBlock re, MatrixBlock im, int threads) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(rows) || !isPowerOfTwo(cols))
			throw new RuntimeException("false dimensions");

		fft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols, threads, true);

		return new MatrixBlock[] {re, im};
	}

	/**
	 * Function to perform IFFT for two given matrices. The first one represents the real values and the second one the
	 * imaginary values. The output also contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param im      matrix object representing the imaginary values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft(MatrixBlock re, MatrixBlock im, int threads) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(rows) || !isPowerOfTwo(cols))
			throw new RuntimeException("false dimensions");

		ifft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols, threads, true);

		return new MatrixBlock[] {re, im};
	}

	/**
	 * Function to perform FFT for each row of two given matrices. The first one represents the real values and the
	 * second one the imaginary values. The output also contains one matrix for the real and one for the imaginary
	 * values.
	 *
	 * @param re      matrix object representing the real values
	 * @param im      matrix object representing the imaginary values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft_linearized(MatrixBlock re, MatrixBlock im, int threads) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(cols))
			throw new RuntimeException("false dimensions");

		fft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols, threads, false);

		return new MatrixBlock[] {re, im};
	}

	/**
	 * Function to perform IFFT for each row of two given matrices. The first one represents the real values and the
	 * second one the imaginary values. The output also contains one matrix for the real and one for the imaginary
	 * values.
	 *
	 * @param re      matrix object representing the real values
	 * @param im      matrix object representing the imaginary values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft_linearized(MatrixBlock re, MatrixBlock im, int threads) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();
		if(!isPowerOfTwo(cols))
			throw new RuntimeException("false dimensions");

		ifft(re.getDenseBlockValues(), im.getDenseBlockValues(), rows, cols, threads, false);

		return new MatrixBlock[] {re, im};
	}

	/**
	 * Function to perform FFT for two given double arrays. The first one represents the real values and the second one
	 * the imaginary values. Both arrays get updated and contain the result.
	 *
	 * @param re          array representing the real values
	 * @param im          array representing the imaginary values
	 * @param rows        number of rows
	 * @param cols        number of rows
	 * @param threads     number of threads
	 * @param inclColCalc if true, fft is also calculated for each column, otherwise only for each row
	 */
	public static void fft(double[] re, double[] im, int rows, int cols, int threads, boolean inclColCalc) {

		double[] re_inter = new double[rows * cols];
		double[] im_inter = new double[rows * cols];

		ExecutorService pool = CommonThreadPool.get(threads);

		for(int i = 0; i < rows; i++) {
			final int finalI = i;
			pool.submit(() -> fft_one_dim(re, im, re_inter, im_inter, finalI * cols, (finalI + 1) * cols, cols, 1));
		}
		awaitParallelExecution(pool);

		if(inclColCalc && rows > 1) {
			for(int j = 0; j < cols; j++) {
				final int finalJ = j;
				pool.submit(() -> fft_one_dim(re, im, re_inter, im_inter, finalJ, finalJ + rows * cols, rows, cols));
			}
			awaitParallelExecution(pool);
		}

	}

	/**
	 * Function to perform IFFT for two given double arrays. The first one represents the real values and the second one
	 * the imaginary values. Both arrays get updated and contain the result.
	 *
	 * @param re          array representing the real values
	 * @param im          array representing the imaginary values
	 * @param rows        number of rows
	 * @param cols        number of rows
	 * @param threads     number of threads
	 * @param inclColCalc if true, fft is also calculated for each column, otherwise only for each row
	 */
	public static void ifft(double[] re, double[] im, int rows, int cols, int threads, boolean inclColCalc) {

		double[] re_inter = new double[rows * cols];
		double[] im_inter = new double[rows * cols];

		ExecutorService pool = CommonThreadPool.get(threads);

		if(inclColCalc && rows > 1) {
			for(int j = 0; j < cols; j++) {
				final int finalJ = j;
				pool.submit(() -> ifft_one_dim(re, im, re_inter, im_inter, finalJ, finalJ + rows * cols, rows, cols));
			}
			awaitParallelExecution(pool);
		}

		for(int i = 0; i < rows; i++) {
			final int finalI = i;
			pool.submit(() -> ifft_one_dim(re, im, re_inter, im_inter, finalI * cols, (finalI + 1) * cols, cols, 1));
		}
		awaitParallelExecution(pool);

	}

	/**
	 * Function to perform one-dimensional FFT for two given double arrays. The first one represents the real values and
	 * the second one the imaginary values. Both arrays get updated and contain the result.
	 *
	 * @param re       array representing the real values
	 * @param im       array representing the imaginary values
	 * @param re_inter array for the real values of intermediate results
	 * @param im_inter array for the imaginary values of intermediate results
	 * @param start    start index (incl.)
	 * @param stop     stop index (excl.)
	 * @param num      number of values used for fft
	 * @param minStep  step size between values used for fft
	 */
	public static void fft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int stop,
		int num, int minStep) {

		// start inclusive, stop exclusive
		if(num == 1)
			return;

		// even indices
		for(int step = minStep * (num / 2), subNum = 2; subNum <= num; step /= 2, subNum *= 2) {

			double angle = -2 * FastMath.PI / subNum;

			// use ceil for the main (sub)array
			for(int sub = 0; sub < FastMath.ceil(num / (2 * (double) subNum)); sub++) {

				for(int isOdd = 0; isOdd < 2; isOdd++) {

					// no odd values for main (sub)array
					if(isOdd == 1 && subNum == num)
						return;

					int startSub = start + sub * minStep + isOdd * (step / 2);

					// first iterates over even indices, then over odd indices
					for(int j = startSub, cnt = 0; cnt < subNum / 2; j += 2 * step, cnt++) {

						double omega_pow_re = FastMath.cos(cnt * angle);
						double omega_pow_im = FastMath.sin(cnt * angle);

						// calculate m using the result of odd index
						double m_re = omega_pow_re * re[j + step] - omega_pow_im * im[j + step];
						double m_im = omega_pow_re * im[j + step] + omega_pow_im * re[j + step];

						int index = startSub + cnt * step;
						re_inter[index] = re[j] + m_re;
						re_inter[index + (stop - start) / 2] = re[j] - m_re;

						im_inter[index] = im[j] + m_im;
						im_inter[index + (stop - start) / 2] = im[j] - m_im;
					}

					for(int j = startSub; j < startSub + (stop - start); j += step) {
						re[j] = re_inter[j];
						im[j] = im_inter[j];
						re_inter[j] = 0;
						im_inter[j] = 0;
					}

				}

			}

		}

	}

	private static void ifft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start,
		int stop, int num, int minStep) {

		// conjugate input
		for(int i = start; i < start + num * minStep; i += minStep) {
			im[i] = -im[i];
		}

		// apply fft
		// fft_one_dim_recursive(re, im, re_inter, im_inter, start, step, num, subArraySize);
		fft_one_dim(re, im, re_inter, im_inter, start, stop, num, minStep);

		// conjugate and scale result
		for(int i = start; i < start + num * minStep; i += minStep) {
			re[i] = re[i] / num;
			im[i] = -im[i] / num;
		}

	}

	/**
	 * Function for checking whether the given integer is a power of two.
	 *
	 * @param n integer to check
	 * @return true if n is a power of two, false otherwise
	 */
	public static boolean isPowerOfTwo(int n) {
		return (n != 0) && ((n & (n - 1)) == 0);
	}

	/**
	 * Function to perform FFT for a given matrix. The given matrix only represents real values. The output contains one
	 * matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft(MatrixBlock re, int threads) {
		return fft(re,
			new MatrixBlock(re.getNumRows(), re.getNumColumns(), new double[re.getNumRows() * re.getNumColumns()]),
			threads);
	}

	/**
	 * Function to perform IFFT for a given matrix. The given matrix only represents real values. The output contains
	 * one matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft(MatrixBlock re, int threads) {
		return ifft(re,
			new MatrixBlock(re.getNumRows(), re.getNumColumns(), new double[re.getNumRows() * re.getNumColumns()]),
			threads);
	}

	/**
	 * Function to perform FFT for each row of a given matrix. The given matrix only represents real values. The output
	 * contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] fft_linearized(MatrixBlock re, int threads) {
		return fft_linearized(re,
			new MatrixBlock(re.getNumRows(), re.getNumColumns(), new double[re.getNumRows() * re.getNumColumns()]),
			threads);
	}

	/**
	 * Function to perform IFFT for each row of a given matrix. The given matrix only represents real values. The output
	 * contains one matrix for the real and one for the imaginary values.
	 *
	 * @param re      matrix object representing the real values
	 * @param threads number of threads
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] ifft_linearized(MatrixBlock re, int threads) {
		return ifft_linearized(re,
			new MatrixBlock(re.getNumRows(), re.getNumColumns(), new double[re.getNumRows() * re.getNumColumns()]),
			threads);
	}

	private static void awaitParallelExecution(ExecutorService pool) {

		try {
			int timeout = 100;
			if(!pool.awaitTermination(timeout, TimeUnit.SECONDS)) {
				pool.shutdownNow();
			}
		}
		catch(InterruptedException e) {
			e.printStackTrace();
		}
	}

}
