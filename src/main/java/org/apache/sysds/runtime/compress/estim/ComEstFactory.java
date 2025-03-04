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
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.lib.CLALibSlice;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface ComEstFactory {
	static final Log LOG = LogFactory.getLog(ComEstFactory.class.getName());

	/**
	 * Create an estimator for the input data with the given settings and parallelization degree.
	 * 
	 * @param data The matrix to extract compression information from.
	 * @param cs   The settings for the compression
	 * @param k    The parallelization degree
	 * @return A new CompressionSizeEstimator used to extract information of column groups
	 */
	public static AComEst createEstimator(MatrixBlock data, CompressionSettings cs, int k) {
		final int nRows = cs.transposed ? data.getNumColumns() : data.getNumRows();
		final int nCols = cs.transposed ? data.getNumRows() : data.getNumColumns();
		final double sparsity = data.getSparsity();
		final int sampleSize = getSampleSize(cs, nRows, nCols, sparsity);
		
		if(data instanceof CompressedMatrixBlock)
			return createCompressedEstimator((CompressedMatrixBlock) data, cs, sampleSize, k);

		if(data.isEmpty())
			return createExactEstimator(data, cs);
		return createEstimator(data, cs, sampleSize, k, nRows);
	}

	/**
	 * Create an estimator for the input data with the given settings and parallelization degree.
	 * 
	 * @param data       The matrix to extract compression information from.
	 * @param cs         The settings for the compression
	 * @param sampleSize The number of rows to extract from the input data to extract information from.
	 * @param k          The parallelization degree
	 * @return A new CompressionSizeEstimator used to extract information of column groups
	 */
	public static AComEst createEstimator(MatrixBlock data, CompressionSettings cs, int sampleSize, int k) {
		final int nRows = cs.transposed ? data.getNumColumns() : data.getNumRows();
		return createEstimator(data, cs, sampleSize, k, nRows);
	}

	private static AComEst createEstimator(MatrixBlock data, CompressionSettings cs, int sampleSize, int k, int nRows) {
		if(sampleSize >= nRows * 0.8) // if sample size is larger than 80% use entire input as sample.
			return createExactEstimator(data, cs);
		else
			return createSampleEstimator(data, cs, sampleSize, k);
	}

	private static ComEstExact createExactEstimator(MatrixBlock data, CompressionSettings cs) {
		LOG.debug("Using full sample");
		return new ComEstExact(data, cs);
	}

	private static AComEst createCompressedEstimator(CompressedMatrixBlock data, CompressionSettings cs, int sampleSize,
		int k) {
		if(sampleSize < data.getNumRows()) {
			LOG.debug("Trying to sample");
			final MatrixBlock slice = CLALibSlice.sliceRowsCompressed(data, 0, sampleSize);
			if(slice instanceof CompressedMatrixBlock) {
				LOG.debug("Using Sampled Compressed Estimator " + sampleSize);
				return new ComEstCompressedSample((CompressedMatrixBlock) slice, cs, data, k);
			}
		}
		LOG.debug("Using Full Compressed Estimator");
		return new ComEstCompressed(data, cs);
	}

	private static ComEstSample createSampleEstimator(MatrixBlock data, CompressionSettings cs, int sampleSize, int k) {
		LOG.debug("Using sample size: " + sampleSize);
		return new ComEstSample(data, cs, sampleSize, k);
	}

	/**
	 * Get sampleSize based on compression settings.
	 * 
	 * @param cs       The compression settings
	 * @param nRows    Number of rows in input
	 * @param nCols    Number of columns in input
	 * @param sparsity The sparsity of the input
	 * @return a sample size
	 */
	private static int getSampleSize(CompressionSettings cs, int nRows, int nCols, double sparsity) {
		final int maxSize = Math.min(cs.maxSampleSize, nRows);
		return getSampleSize(cs.samplePower, nRows, nCols, sparsity, cs.minimumSampleSize, maxSize);
	}

	/**
	 * This function returns the sample size to use.
	 * 
	 * The sampling is bound by the maximum sampling and the minimum sampling.
	 * 
	 * The sampling is calculated based on the a power of the number of rows and a sampling fraction
	 * 
	 * @param samplePower   The sample power
	 * @param nRows         The number of rows
	 * @param nCols         The number of columns
	 * @param sparsity      The sparsity of the input
	 * @param minSampleSize The minimum sample size
	 * @param maxSampleSize The maximum sample size
	 * @return The sample size to use.
	 */
	public static int getSampleSize(double samplePower, int nRows, int nCols, double sparsity, int minSampleSize,
		int maxSampleSize) {

		// Start sample size at the min sample size as the basis sample.
		int sampleSize = minSampleSize;

		// ensure samplePower is in valid range
		samplePower = Math.max(0, Math.min(1, samplePower));

		// Scale the sample size with the number of rows in the input.
		// Sub linearly since the the number of rows needed to classify the contained values in a population doesn't scale
		// linearly.
		sampleSize += (int) Math.ceil(Math.pow(nRows, samplePower));

		// Scale sample size based on overall sparsity so that if the input is very sparse, increase the sample size.
		sampleSize = (int) (sampleSize * (1.0 / Math.min(sparsity + 0.2, 1.0)));

		// adhere to maximum sample size.
		sampleSize = Math.max(minSampleSize, Math.min(sampleSize, maxSampleSize));

		// cap at number of rows.
		sampleSize = Math.min(nRows, sampleSize);

		return sampleSize;
	}
}
