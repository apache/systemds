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
		final double sparsity = data.getSparsity();
		final double sampleRatio = cs.samplingRatio;
		final int minSample = cs.minimumSampleSize;
		final int maxSample = Math.min(cs.maxSampleSize, nRows);
		final int sampleSize = getSampleSize(sampleRatio, nRows, nCols, sparsity, minSample, maxSample);

		return getSizeEstimator(data, cs, sampleSize, k);
	}

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, CompressionSettings cs, int sampleSize,
		int k) {
		final int nRows = cs.transposed ? data.getNumColumns() : data.getNumRows();
		final int nCols = cs.transposed ? data.getNumRows() : data.getNumColumns();

		if(sampleSize >= (double) nRows * 0.8) {
			if(nRows > 10000 && nCols > 10 && data.isInSparseFormat() && !cs.transposed) {
				if(!cs.isInSparkInstruction)
					LOG.debug("Transposing for exact estimator");
				data = LibMatrixReorg.transpose(data,
					new MatrixBlock(data.getNumColumns(), data.getNumRows(), data.isInSparseFormat()), k);
				cs.transposed = true;
			}
			if(!cs.isInSparkInstruction)
				LOG.debug("Using Exact estimator");
			return new CompressedSizeEstimatorExact(data, cs);
		}

		if(nCols > 1000 && data.getSparsity() < 0.00001)
			return CompressedSizeEstimatorUltraSparse.create(data, cs, k);
		else {
			if(!cs.isInSparkInstruction)
				LOG.debug("Trying sample size: " + sampleSize);
			return new CompressedSizeEstimatorSample(data, cs, sampleSize, k);
		}
	}

	/**
	 * This function returns the sample size to use.
	 * 
	 * The sampling is bound by the maximum sampling and the minimum sampling.
	 * 
	 * The sampling is calculated based on the a power of the number of rows and a sampling fraction
	 * 
	 * @param sampleRatio       The sample ratio
	 * @param nRows             The number of rows
	 * @param nCols             The number of columns
	 * @param sparsity          The sparsity of the input
	 * @param minimumSampleSize The minimum sample size
	 * @param maxSampleSize     The maximum sample size
	 * @return The sample size to use.
	 */
	private static int getSampleSize(double sampleRatio, int nRows, int nCols, double sparsity, int minSampleSize,
		int maxSampleSize) {

		// Start sample size at the min sample size.
		int sampleSize = minSampleSize;

		// Scale the sample size disproportionally with the number of rows in the input.
		// Since the the number of rows needed to classify the contained values in a population doesn't scale linearly.
		sampleSize += (int) Math.ceil(Math.pow(nRows, 0.65));

		// Scale sample size based on overall sparsity so that if the input is very sparse, increase the sample size.
		sampleSize = (int) (sampleSize * (1.0 / Math.min(sparsity + 0.2, 1.0)));

		// adhere to maximum sample size.
		sampleSize = (int) Math.max(minSampleSize, Math.min(sampleSize, maxSampleSize));

		return sampleSize;
	}
}
