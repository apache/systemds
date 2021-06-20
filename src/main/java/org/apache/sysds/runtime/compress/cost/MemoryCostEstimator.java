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

import java.util.Collection;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public class MemoryCostEstimator implements ICostEstimate {

	public MemoryCostEstimator() {

	}

	@Override
	public double getUncompressedCost(int nRows, int nCols, int sparsity) {
		throw new DMLCompressionException("Use original matrix size instead of estimate");
	}

	@Override
	public double getCostOfColumnGroup(CompressedSizeInfoColGroup g) {
		return g.getMinSize();
	}

	@Override
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gss) {
		throw new DMLCompressionException("Memory based compression is not related to comparing all columns");
	}

	@Override
	public double getCostOfCollectionOfGroups(Collection<CompressedSizeInfoColGroup> gss,
		CompressedSizeInfoColGroup g) {
		throw new DMLCompressionException("Memory based compression is not related to comparing all columns");
	}

	@Override
	public double getCostOfTwoGroups(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		throw new DMLCompressionException("Memory based compression is not related to comparing all columns");
	}

	@Override
	public boolean isCompareAll() {
		return false;
	}

	@Override
	public boolean shouldAnalyze(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}

	@Override
	public boolean shouldTryJoin(CompressedSizeInfoColGroup g1, CompressedSizeInfoColGroup g2) {
		return true;
	}

}
