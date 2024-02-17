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

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft_one_dim;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.isPowerOfTwo;

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
	public static MatrixBlock[] stft(MatrixBlock re, MatrixBlock im, int windowSize, int overlap) {

		int rows = re.getNumRows();
		int cols = re.getNumColumns();

		int stepSize = windowSize - overlap;
		if (stepSize == 0) {
			throw new IllegalArgumentException("windowSize - overlap is zero");
		}


		//double frames = (double) (cols - overlap + stepSize - 1) / stepSize; // frames per row
		//int numberOfFramesPerRow = (int) FastMath.ceil(frames);
		int numberOfFramesPerRow = (cols - overlap + stepSize - 1) / stepSize;
		int rowLength= numberOfFramesPerRow * windowSize;
		int out_len = rowLength * rows;

		double[] stftOutput_re = new double[out_len];
		double[] stftOutput_im = new double[out_len];

		double[] re_inter = new double[out_len];
		double[] im_inter = new double[out_len];

		for (int h = 0; h < rows; h++){
			for (int i = 0; i < numberOfFramesPerRow; i++) {
				for (int j = 0; j < windowSize; j++) {
					if ((i * stepSize + j) < cols) {
						stftOutput_re[h * rowLength + i * windowSize + j] = re.getDenseBlockValues()[h * cols + i * stepSize + j];
						stftOutput_im[h * rowLength + i * windowSize + j] = im.getDenseBlockValues()[h * cols + i * stepSize + j];
					}
				}
				fft_one_dim(stftOutput_re, stftOutput_im, re_inter, im_inter, h * rowLength + i * windowSize, h * rowLength + (i+1) * windowSize, windowSize, 1);
			}
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
	public static MatrixBlock[] stft(MatrixBlock re, int windowSize, int overlap){
		//return stft(re.getDenseBlockValues(), new double[re.getDenseBlockValues().length], windowSize, overlap);
		return stft(re, new MatrixBlock(re.getNumRows(), re.getNumColumns(),  new double[re.getNumRows() * re.getNumColumns()]), windowSize, overlap);
	}

}
