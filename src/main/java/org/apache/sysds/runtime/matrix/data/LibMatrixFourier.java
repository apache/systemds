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

import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibMatrixFourier {

	protected static final Log LOG = LogFactory.getLog(LibMatrixFourier.class.getName());

	static SoftReference<HashMap<Double, Double>> sinCacheRef = new SoftReference<>(new HashMap<>());
	static SoftReference<HashMap<Double, Double>> cosCacheRef = new SoftReference<>(new HashMap<>());

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

		MatrixBlock re_out = new MatrixBlock();
		re_out.copy(re, false);
		MatrixBlock im_out = new MatrixBlock();
		im_out.copy(im, false);

		fft(re_out.getDenseBlockValues(), im_out.getDenseBlockValues(), rows, cols, threads, true);

		re_out.recomputeNonZeros(threads);
		im_out.recomputeNonZeros(threads);

		return new MatrixBlock[] {re_out, im_out};
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

		MatrixBlock re_out = new MatrixBlock();
		re_out.copy(re, false);
		MatrixBlock im_out = new MatrixBlock();
		im_out.copy(im, false);

		ifft(re_out.getDenseBlockValues(), im_out.getDenseBlockValues(), rows, cols, threads, true);

		re_out.recomputeNonZeros(threads);
		im_out.recomputeNonZeros(threads);

		return new MatrixBlock[] {re_out, im_out};
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

		MatrixBlock re_out = new MatrixBlock();
		re_out.copy(re, false);
		MatrixBlock im_out = new MatrixBlock();
		im_out.copy(im, false);

		fft(re_out.getDenseBlockValues(), im_out.getDenseBlockValues(), rows, cols, threads, false);

		re_out.recomputeNonZeros(threads);
		im_out.recomputeNonZeros(threads);
		return new MatrixBlock[] {re_out, im_out};
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

		MatrixBlock re_out = new MatrixBlock();
		re_out.copy(re, false);
		MatrixBlock im_out = new MatrixBlock();
		im_out.copy(im, false);

		ifft(re_out.getDenseBlockValues(), im_out.getDenseBlockValues(), rows, cols, threads, false);

		re_out.recomputeNonZeros(threads);
		im_out.recomputeNonZeros(threads);

		return new MatrixBlock[] {re_out, im_out};
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

		final ExecutorService pool = CommonThreadPool.get(threads);
		final List<Future<?>> tasks = new ArrayList<>();

		try {
			final int rBlz = Math.max(rows / threads, 32);
			final int cBlz = Math.max(cols / threads, 32);

			for(int i = 0; i < rows; i += rBlz) {
				final int start = i;
				final int end = Math.min(i + rBlz, rows);
				tasks.add(pool.submit(() -> {
					for(int j = start; j < end; j++) {
						fft_one_dim(re, im, re_inter, im_inter, j * cols, (j + 1) * cols, cols, 1);
					}
				}));
			}

			for(Future<?> f : tasks)
				f.get();

			tasks.clear();
			if(inclColCalc && rows > 1) {
				for(int j = 0; j < cols; j += cBlz) {
					final int start = j;
					final int end = Math.min(j + cBlz, cols);
					tasks.add(pool.submit(() -> {
						for(int i = start; i < end; i++) {
							fft_one_dim(re, im, re_inter, im_inter, i, i + rows * cols, rows, cols);
						}
					}));
				}

				for(Future<?> f : tasks)
					f.get();
			}
		}
		catch(InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
		finally {
			pool.shutdown();
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
		final List<Future<?>> tasks = new ArrayList<>();

		try {
			final int rBlz = Math.max(rows / threads, 32);
			final int cBlz = Math.max(cols / threads, 32);

			if(inclColCalc && rows > 1) {
				for(int j = 0; j < cols; j += cBlz) {
					final int start = j;
					final int end = Math.min(j + cBlz, cols);
					tasks.add(pool.submit(() -> {
						for(int i = start; i < end; i++) {
							ifft_one_dim(re, im, re_inter, im_inter, i, i + rows * cols, rows, cols);
						}
					}));
				}

				for(Future<?> f : tasks)
					f.get();
			}

			tasks.clear();
			for(int i = 0; i < rows; i += rBlz) {
				final int start = i;
				final int end = Math.min(i + rBlz, rows);
				tasks.add(pool.submit(() -> {
					for(int j = start; j < end; j++) {
						ifft_one_dim(re, im, re_inter, im_inter, j * cols, (j + 1) * cols, cols, 1);
					}
				}));
			}

			for(Future<?> f : tasks)
				f.get();
		}
		catch(InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
		finally {
			pool.shutdown();
		}

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
			fft_one_dim_iter_sub(re, im, re_inter, im_inter, start, stop, num, minStep, subNum, step, angle);
		}
	}

	/**
	 * Function to call one-dimensional FFT for all subArrays of two given double arrays. The first one represents the
	 * real values and the second one the imaginary values. Both arrays get updated and contain the result.
	 *
	 * @param re       array representing the real values
	 * @param im       array representing the imaginary values
	 * @param re_inter array for the real values of intermediate results
	 * @param im_inter array for the imaginary values of intermediate results
	 * @param start    start index (incl.)
	 * @param stop     stop index (excl.)
	 * @param num      number of values used for fft
	 * @param minStep  step size between values used for fft
	 * @param subNum   number of elements in subarray
	 * @param step     step size between elements in subarray
	 * @param angle    angle
	 */
	private static void fft_one_dim_iter_sub(double[] re, double[] im, double[] re_inter, double[] im_inter, int start,
		int stop, int num, int minStep, int subNum, int step, double angle) {

		// use ceil for the main (sub)array
		for(int sub = 0; sub < FastMath.ceil(num / (2 * (double) subNum)); sub++) {

			int startSub = start + sub * minStep;
			fft_one_dim_sub(re, im, re_inter, im_inter, start, stop, startSub, subNum, step, angle);

			// no odd values for main (sub)array
			if(subNum == num)
				return;

			startSub = start + sub * minStep + step / 2;
			fft_one_dim_sub(re, im, re_inter, im_inter, start, stop, startSub, subNum, step, angle);
		}
	}

	/**
	 * Function to perform one-dimensional FFT for subArrays of two given double arrays. The first one represents the
	 * real values and the second one the imaginary values. Both arrays get updated and contain the result.
	 *
	 * @param re       array representing the real values
	 * @param im       array representing the imaginary values
	 * @param re_inter array for the real values of intermediate results
	 * @param im_inter array for the imaginary values of intermediate results
	 * @param start    start index (incl.)
	 * @param stop     stop index (excl.)
	 * @param startSub start index of subarray (incl.)
	 * @param subNum   number of elements in subarray
	 * @param step     step size between elements in subarray
	 * @param angle    angle
	 */
	private static void fft_one_dim_sub(double[] re, double[] im, double[] re_inter, double[] im_inter, int start,
		int stop, int startSub, int subNum, int step, double angle) {

		HashMap<Double, Double> sinCache = sinCacheRef.get();
		HashMap<Double, Double> cosCache = cosCacheRef.get();

		if(sinCache == null)
			sinCache = new HashMap<>();
		if(cosCache == null)
			cosCache = new HashMap<>();

		for(int j = startSub, cnt = 0; cnt < subNum / 2; j += 2 * step, cnt++) {

			double omega_pow_re = cos(cnt * angle, cosCache);
			double omega_pow_im = sin(cnt * angle, sinCache);

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

	/**
	 * Function to perform one-dimensional IFFT for two given double arrays. The first one represents the real values and
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
	private static void ifft_one_dim(double[] re, double[] im, double[] re_inter, double[] im_inter, int start, int stop,
		int num, int minStep) {

		// conjugate input
		for(int i = start; i < start + num * minStep; i += minStep) {
			im[i] = -im[i];
		}

		// apply fft
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
	 * Function to perform IFFT for a given matrix. The given matrix only represents real values. The output contains one
	 * matrix for the real and one for the imaginary values.
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

	private static double sin(double angle, HashMap<Double, Double> cache) {
		final double v = cache.getOrDefault(angle, -100.0);
		if(Util.eq(v, -100.0)) { // value not in cache.
			final double res = FastMath.sin(angle);
			if(cache.size() < 1000)
				cache.put(angle, res);
			return res;
		}
		else
			return v;

	}

	private static double cos(double angle, HashMap<Double, Double> cache) {
		double v = cache.getOrDefault(angle, -100.0);
		if(Util.eq(v, -100.0)) { // value not in cache.
			double res = FastMath.cos(angle);
			if(cache.size() < 1000)
				cache.put(angle, res);
			return res;
		}
		else
			return v;

	}

}
