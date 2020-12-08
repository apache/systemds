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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorFactory {
	protected static final Log LOG = LogFactory.getLog(CompressedSizeEstimatorFactory.class.getName());

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, CompressionSettings compSettings) {
		long elements = compSettings.transposeInput ? data.getNumColumns() : data.getNumRows();
		elements = data.getNonZeros() / (compSettings.transposeInput ? data.getNumRows() : data.getNumColumns());
		CompressedSizeEstimator est;
		if(compSettings.samplingRatio >= 1.0 || elements < 1000) {
			est = new CompressedSizeEstimatorExact(data, compSettings);
			LOG.debug("Using Exact estimator for compression");
		}
		else {
			int sampleSize = Math.max((int) Math.ceil(elements * compSettings.samplingRatio), 10000);
			int[] sampleRows = CompressedSizeEstimatorSample
				.getSortedUniformSample(compSettings.transposeInput ? data.getNumColumns(): data.getNumRows(), sampleSize, compSettings.seed);
			est = new CompressedSizeEstimatorSample(data, compSettings, sampleRows);
			if(LOG.isDebugEnabled()) {

				LOG.debug("Using Sampled estimator for compression with sample size: " + sampleSize);
			}
		}
		return est;
	}
}
