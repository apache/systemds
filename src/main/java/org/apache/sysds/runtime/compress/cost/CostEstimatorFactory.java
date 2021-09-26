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

public final class CostEstimatorFactory {

	protected static final Log LOG = LogFactory.getLog(CostEstimatorFactory.class.getName());

	public enum CostType {
		MEMORY, W_TREE, HYBRID_W_TREE, DISTINCT, AUTO;
	}

	public static ICostEstimate create(CompressionSettings cs, CostEstimatorBuilder costVector, int nRows, int nCols,
		double sparsity) {
		switch(cs.costComputationType) {
			case DISTINCT:
				return new DistinctCostEstimator(nRows, cs, sparsity);
			case HYBRID_W_TREE:
				if(costVector != null)
					return costVector.create(nRows, nCols, sparsity, true);
				else
					return genDefaultCostCase(nRows, nCols, sparsity, true);
			case MEMORY:
				return new MemoryCostEstimator(nRows, nCols, sparsity);
			case W_TREE:
			case AUTO:
			default:
				if(costVector != null)
					return costVector.create(nRows, nCols, sparsity, cs.isInSparkInstruction);
				else
					return genDefaultCostCase(nRows, nCols, sparsity, cs.isInSparkInstruction);
		}
	}

	public static ICostEstimate genDefaultCostCase(int nRows, int nCols, double sparsity,
		boolean isInSparkInstruction) {
		if(isInSparkInstruction)
			return new HybridCostEstimator(nRows, nCols, sparsity, 1, 1, 0, 1, 1, 1, 10, true);
		else
			return new ComputationCostEstimator(nRows, nCols, sparsity, 1, 1, 0, 1, 1, 1, 10, true);
	}
}
