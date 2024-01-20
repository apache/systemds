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
        return stft(re.getDenseBlockValues(), windowSize, overlap);
    }

    /**
     * Function to perform STFT on two given matrices with windowSize and overlap.
     * The first one represents the real values and the second one the imaginary values.
     * The output also contains one matrix for the real and one for the imaginary values.
     * The results of the fourier transformations are appended to each other in the output.
     *
     * @param signal array representing the signal
     * @param windowSize size of window
     * @param overlap size of overlap
     * @return array of two matrix blocks
     */
    public static MatrixBlock[] stft(double[] signal, int windowSize, int overlap) {

        if (windowSize < 1 || (windowSize & (windowSize - 1)) != 0) {
            throw new IllegalArgumentException("windowSize is not a power of two");
        }
        int stepSize = windowSize - overlap;
        int totalFrames = (signal.length - overlap + stepSize - 1) / stepSize;
        int out_len = totalFrames * windowSize;

        // Initialize the STFT output array: [time frame][frequency bin][real & imaginary parts]
        double[] stftOutput_re = new double[out_len];
        double[] stftOutput_im = new double[out_len];

        for (int i = 0; i < totalFrames; i++) {
            for (int j = 0; j < windowSize; j++) {
                stftOutput_re[i * windowSize + j] = ((i * stepSize + j) < signal.length) ? signal[i * stepSize + j] : 0;
            }
        }

        for (int frame = 0; frame < totalFrames; frame++) {
            // Perform FFT on windowed signal
            double[] re_inter = new double[out_len];
            double[] im_inter = new double[out_len];
            fft_one_dim(stftOutput_re, stftOutput_im, re_inter, im_inter, frame * windowSize, 1, windowSize, windowSize);
        }
        return new MatrixBlock[]{new MatrixBlock(1, out_len, stftOutput_re), new MatrixBlock(1, out_len, stftOutput_im)};
    }

    /**
     * Function to perform STFT on two given matrices with windowSize and overlap.
     * The first one represents the real values and the second one the imaginary values.
     * The output also contains one matrix for the real and one for the imaginary values.
     * The results of the fourier transformations are appended to each other in the output.
     *
     * @param in matrix object representing the real values
     * @param windowSize size of window
     * @param overlap size of overlap
     * @return array of two matrix blocks
     */
    public static MatrixBlock[] stft(MatrixBlock[] in, int windowSize, int overlap){
        return stft(in[0].getDenseBlockValues(), windowSize, overlap);
    }

}
