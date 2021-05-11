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

	public static final int minimumSampleSize = 2000;

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, CompressionSettings cs) {

		final long nRows = cs.transposed ? data.getNumColumns() : data.getNumRows();
		
		// Calculate the sample size.
		// If the sample size is very small, set it to the minimum size
		final int sampleSize = Math.max((int) Math.ceil(nRows * cs.samplingRatio), minimumSampleSize);

		CompressedSizeEstimator est;
		if(cs.samplingRatio >= 1.0 || nRows < minimumSampleSize || sampleSize > nRows)
			est = new CompressedSizeEstimatorExact(data, cs);
		else
			est = new CompressedSizeEstimatorSample(data, cs, sampleSize);

		LOG.debug("Estimating using: " + est);
		return est;
	}
}
