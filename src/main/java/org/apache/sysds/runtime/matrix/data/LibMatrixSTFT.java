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

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft_one_dim;

import org.apache.sysds.runtime.util.CommonThreadPool;
import java.util.ArrayList;
import java.util.concurrent.Future;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ExecutionException;
import java.util.List;

public class LibMatrixSTFT {

	/**
	 * Function to perform STFT on two given matrices with windowSize and overlap.
	 * The first one represents the real values and the second one the imaginary values.
	 * The output also contains one matrix for the real and one for the imaginary values.
	 * The results of the fourier transformations are appended to each other in the output.
	 *
	 * @param re matrix object representing the real values
	 * @param im matrix object representing the imaginary values
	 * @param windowSize size of window
	 * @param overlap size of overlap
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] stft(MatrixBlock re, MatrixBlock im, int windowSize, int overlap, int threads) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();

		int stepSize = windowSize - overlap;
		if (stepSize == 0) {
			throw new IllegalArgumentException("windowSize - overlap is zero");
		}

		int numberOfFramesPerRow = (cols - overlap + stepSize - 1) / stepSize;
		int rowLength= numberOfFramesPerRow * windowSize;
		int out_len = rowLength * rows;

		double[] stftOutput_re = new double[out_len];
		double[] stftOutput_im = new double[out_len];

		double[] re_inter = new double[out_len];
		double[] im_inter = new double[out_len];

		//ExecutorService pool = CommonThreadPool.get(1);

		final ExecutorService pool = CommonThreadPool.get(threads);

		final List<Future<?>> tasks = new ArrayList<>();

		try {
			for (int h = 0; h < rows; h++){
				final int finalH = h;
				tasks.add( pool.submit(() -> {
					for (int i = 0; i < numberOfFramesPerRow; i++) {
						for (int j = 0; j < windowSize; j++) {
							if ((i * stepSize + j) < cols) {
								stftOutput_re[finalH * rowLength + i * windowSize + j] = re.getDenseBlockValues()[finalH * cols + i * stepSize + j];
								stftOutput_im[finalH * rowLength + i * windowSize + j] = im.getDenseBlockValues()[finalH * cols + i * stepSize + j];
							}
						}
						fft_one_dim(stftOutput_re, stftOutput_im, re_inter, im_inter, finalH * rowLength + i * windowSize, finalH * rowLength + (i + 1) * windowSize, windowSize, 1);
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


		/*
		for (int i = 0; i < stftOutput_re.length; i++) {
			System.out.println(stftOutput_re[i] + stftOutput_im[i]);
		}

		 */
		int i = 0;
		while (i < 1000000000) {
			i = i + 1;
		}
		i = 0;
		while (i < 1000000000) {
			i = i + 1;
		}
		i = 0;
		while (i < 1000000000) {
			i = i + 1;
		}



		return new MatrixBlock[]{new MatrixBlock(rows, rowLength, stftOutput_re), new MatrixBlock(rows, rowLength, stftOutput_im)};
	}

	/**
	 * Function to perform STFT on a given matrices with windowSize and overlap.
	 * The matrix represents the real values.
	 * The output contains one matrix for the real and one for the imaginary values.
	 * The results of the fourier transformations are appended to each other in the output.
	 *
	 * @param re matrix object representing the real values
	 * @param windowSize size of window
	 * @param overlap size of overlap
	 * @return array of two matrix blocks
	 */
	public static MatrixBlock[] stft(MatrixBlock re, int windowSize, int overlap, int threads){
		return stft(re, new MatrixBlock(re.getNumRows(), re.getNumColumns(),  new double[re.getNumRows() * re.getNumColumns()]), windowSize, overlap, threads);
	}
}
