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

package org.apache.sysds.runtime.compress.estim;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorFactory {
	protected static final Log LOG = LogFactory.getLog(CompressedSizeEstimatorFactory.class.getName());

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, CompressionSettings cs, int k) {

		final int nRows = cs.transposed ? data.getNumColumns() : data.getNumRows();
		final int nCols = cs.transposed ? data.getNumRows() : data.getNumColumns();
		final int nnzRows = (int) Math.ceil(data.getNonZeros() / nCols);

		final double sampleRatio = cs.samplingRatio;
		final int sampleSize = getSampleSize(sampleRatio, nRows, nCols, cs.minimumSampleSize, cs.maxSampleSize);

		if(nCols > 1000) {
			return tryToMakeSampleEstimator(data, cs, sampleRatio, sampleSize / 10, nRows, nnzRows, k);
		}
		else {
			if(shouldUseExactEstimator(cs, nRows, sampleSize, nnzRows)) {
				if(sampleSize > nnzRows && nRows > 10000 && nCols > 10 && !cs.transposed) {
					if(! cs.isInSparkInstruction)
						LOG.info("Transposing for exact estimator");
					data = LibMatrixReorg.transpose(data,
						new MatrixBlock(data.getNumColumns(), data.getNumRows(), data.isInSparseFormat()), k);
					cs.transposed = true;
				}
				if(! cs.isInSparkInstruction)
					LOG.info("Using Exact estimator");
				return new CompressedSizeEstimatorExact(data, cs);
			}
			else {
				if(! cs.isInSparkInstruction)
					LOG.info("Trying sample size: " + sampleSize);
				return tryToMakeSampleEstimator(data, cs, sampleRatio, sampleSize, nRows, nnzRows, k);
			}
		}

	}

	private static CompressedSizeEstimator tryToMakeSampleEstimator(MatrixBlock data, CompressionSettings cs,
		double sampleRatio, int sampleSize, int nRows, int nnzRows, int k) {
		CompressedSizeEstimatorSample estS = new CompressedSizeEstimatorSample(data, cs, sampleSize, k);
		int double_number = 1;
		while(estS.getSample() == null) {
			if(! cs.isInSparkInstruction)
				LOG.warn("Doubling sample size " + double_number++);
			sampleSize = sampleSize * 2;
			if(shouldUseExactEstimator(cs, nRows, sampleSize, nnzRows))
				return new CompressedSizeEstimatorExact(data, cs);
			else
				estS.sampleData(sampleSize);
		}
		return estS;
	}

	private static boolean shouldUseExactEstimator(CompressionSettings cs, int nRows, int sampleSize, int nnzRows) {
		return cs.samplingRatio >= 1.0 || nRows < cs.minimumSampleSize || sampleSize >= nnzRows;
	}

	/**
	 * This function returns the sample size to use.
	 * 
	 * The sampling is bound by the maximum sampling and the minimum sampling other than that a linear relation is used
	 * with the sample ratio.
	 * 
	 * Also influencing the sample size is the number of columns. If the number of columns is large the sample size is
	 * scaled down, this gives worse estimations of distinct items, but it makes sure that the compression time is more
	 * consistent.
	 * 
	 * @param sampleRatio       The sample ratio
	 * @param nRows             The number of rows
	 * @param nCols             The number of columns
	 * @param minimumSampleSize the minimum sample size
	 * @param maxSampleSize     the maximum sample size
	 * @return The sample size to use.
	 */
	private static int getSampleSize(double sampleRatio, int nRows, int nCols, int minSampleSize, int maxSampleSize) {
		int sampleSize = (int) Math.ceil(nRows * sampleRatio / Math.max(1, (double)nCols / 150));
		if(sampleSize < 20000)
			sampleSize *= 2;
		return Math.min(Math.max(sampleSize, minSampleSize), maxSampleSize);
	}
}
