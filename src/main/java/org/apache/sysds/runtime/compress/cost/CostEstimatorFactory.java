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

import org.apache.sysds.runtime.compress.CompressionSettings;

/**
 * Factory class for the construction of cost estimators for compression
 */
public interface CostEstimatorFactory {

	public enum CostType {
		MEMORY, W_TREE, HYBRID_W_TREE, DISTINCT, AUTO;
	}

	/**
	 * Create a cost estimator to enable comparison of different suggested compression formats
	 * 
	 * @param cs          The compression settings to use.
	 * @param costBuilder A cost builder to build the specific specialized formats of cost estimators
	 * @param nRows       The number of rows in a given input to compress
	 * @param nCols       The number of columns in a given input to compress
	 * @param sparsity    The sparsity of the input to compress
	 * @return A cost estimator
	 */
	public static ACostEstimate create(CompressionSettings cs, CostEstimatorBuilder costBuilder, int nRows, int nCols,
		double sparsity) {
		switch(cs.costComputationType) {
			case DISTINCT:
				return new DistinctCostEstimator(nRows, cs, sparsity);
			case HYBRID_W_TREE:
				if(costBuilder != null)
					return costBuilder.createHybrid();
			case W_TREE:
			case AUTO:
				if(costBuilder != null)
					return costBuilder.create(cs.isInSparkInstruction);
			case MEMORY:
			default:
				return new MemoryCostEstimator();
		}
	}
}
