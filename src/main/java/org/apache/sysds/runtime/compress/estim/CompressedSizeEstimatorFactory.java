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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CompressedSizeEstimatorFactory {
	public static final boolean EXTRACT_SAMPLE_ONCE = true;

	public static CompressedSizeEstimator getSizeEstimator(MatrixBlock data, int numRows, long seed, double sampling_ratio) {
		return (sampling_ratio == 1.0) ? new CompressedSizeEstimatorExact(data) : new CompressedSizeEstimatorSample(
			data, (int) Math.ceil(numRows * sampling_ratio), seed);
	}
}
