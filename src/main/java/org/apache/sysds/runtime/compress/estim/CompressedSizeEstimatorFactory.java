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

	private static final int minimumSampleSize = 2000;

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, CompressionSettings compSettings) {
		long elements = compSettings.transposed ? data.getNumColumns() : data.getNumRows();
		elements = data.getNonZeros() / (compSettings.transposed ? data.getNumRows() : data.getNumColumns());
		CompressedSizeEstimator est;

		// Calculate the sample size.
		// If the sample size is very small, set it to the minimum size
		int sampleSize = Math.max((int) Math.ceil(elements * compSettings.samplingRatio), minimumSampleSize);
		if(compSettings.samplingRatio >= 1.0 || elements < minimumSampleSize || sampleSize > elements) {
			est = new CompressedSizeEstimatorExact(data, compSettings, compSettings.transposed);
		}
		else {
			int[] sampleRows = CompressedSizeEstimatorSample.getSortedUniformSample(
				compSettings.transposed ? data.getNumColumns() : data.getNumRows(),
				sampleSize,
				compSettings.seed);
				est = new CompressedSizeEstimatorSample(data, compSettings, sampleRows, compSettings.transposed);
		}

		LOG.debug(est);
		return est;
	}
}
