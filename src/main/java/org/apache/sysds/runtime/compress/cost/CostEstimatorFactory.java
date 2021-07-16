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

package org.apache.sysds.runtime.compress.cost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;

public final class CostEstimatorFactory {

	protected static final Log LOG = LogFactory.getLog(CostEstimatorFactory.class.getName());

	public enum CostType {
		MEMORY, LEFT_MATRIX_MULT, DECOMPRESSION, TSMM, W_TREE, HYBRID_W_TREE, DISTINCT, AUTO;
	}

	public static ICostEstimate create(CompressionSettings cs, WTreeRoot root, int nRows, int nCols) {
		switch(cs.costComputationType) {
			case DISTINCT:
				return new DistinctCostEstimator(nRows, cs);
			case W_TREE:
			case AUTO:
				if(root != null) {
					CostEstimatorBuilder b = new CostEstimatorBuilder(root);
					if(LOG.isDebugEnabled())
						LOG.debug(b);
					return b.create(nRows, nCols);
				}
				else
					return new DistinctCostEstimator(nRows, cs);
			case MEMORY:
			default:
				return new MemoryCostEstimator();
		}
	}

}
