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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSizes;

/**
 * Information collected about a specific ColGroup's compression size.
 */
public class CompressedSizeInfoColGroup {

	private final int[] _columns;
	private final int _numVals;
	private final int _numOffs;
	private final double _cardinalityRatio;
	private final long _minSize;
	private final CompressionType _bestCompressionType;
	private final Map<CompressionType, Long> _sizes;

	public CompressedSizeInfoColGroup(EstimationFactors fact, Set<CompressionType> validCompressionTypes) {
		_numVals = fact.numVals;
		_numOffs = fact.numOffs;
		_cardinalityRatio = fact.numVals > 0 ? fact.numRows / fact.numVals : Double.POSITIVE_INFINITY ;
		_columns = fact.cols;
		_sizes = calculateCompressionSizes(fact, validCompressionTypes);
		Map.Entry<CompressionType, Long> bestEntry = null;
		for(Map.Entry<CompressionType, Long> ent : _sizes.entrySet()) {
			if(bestEntry == null || ent.getValue() < bestEntry.getValue())
				bestEntry = ent;
		}

		_bestCompressionType = bestEntry.getKey();
		_minSize = bestEntry.getValue();
	}

	public long getCompressionSize(CompressionType ct) {
		return _sizes.get(ct);
	}

	public CompressionType getBestCompressionType() {
		return _bestCompressionType;
	}

	public Map<CompressionType, Long> getAllCompressionSizes() {
		return _sizes;
	}

	public long getMinSize() {
		return _minSize;
	}

	/**
	 * Note cardinality is the same as number of distinct values.
	 * 
	 * @return cardinality or number of distinct values.
	 */
	public int getNumVals() {
		return _numVals;
	}

	/**
	 * Number of offsets, or number of non zero values.
	 * 
	 * @return Number of non zeros or number of values.
	 */
	public int getNumOffs() {
		return _numOffs;
	}

	public int[] getColumns(){
		return _columns;
	}

	public double getCardinalityRatio(){
		return _cardinalityRatio;
	}

	private static Map<CompressionType, Long> calculateCompressionSizes(EstimationFactors fact,
		Set<CompressionType> validCompressionTypes) {
		Map<CompressionType, Long> res = new HashMap<>();
		for(CompressionType ct : validCompressionTypes) {
			res.put(ct, getCompressionSize(ct, fact));
		}
		return res;
	}

	private static Long getCompressionSize(CompressionType ct, EstimationFactors fact) {

		switch(ct) {
			case DDC:
				return ColGroupSizes.estimateInMemorySizeDDC(fact.numCols, fact.numVals + (fact.containsZero ? 1: 0), fact.numRows, fact.lossy);
			case RLE:
				return ColGroupSizes
					.estimateInMemorySizeRLE(fact.numCols, fact.numVals * fact.numCols, fact.numRuns, fact.numRows, fact.lossy);
			case OLE:
				return ColGroupSizes
					.estimateInMemorySizeOLE(fact.numCols, fact.numVals * fact.numCols, fact.numOffs + fact.numVals, fact.numRows, fact.lossy);
			case UNCOMPRESSED:
				return ColGroupSizes.estimateInMemorySizeUncompressed(fact.numRows,
					fact.numCols,
					((double) fact.numOffs / (fact.numRows * fact.numCols)));
			case SDC:
				if(fact.numOffs == 1)
					return ColGroupSizes.estimateInMemorySizeSDCSingle(fact.numCols,
						fact.numVals,
						fact.numRows,
						fact.largestOff,
						fact.lossy);
				return ColGroupSizes
					.estimateInMemorySizeSDC(fact.numCols, fact.numVals, fact.numRows, fact.largestOff, fact.lossy);
			case CONST:
				if(fact.numOffs == 0)
					return ColGroupSizes.estimateInMemorySizeEMPTY(fact.numCols);
				else if(fact.numOffs == fact.numRuns && fact.numVals == 1)
					return ColGroupSizes.estimateInMemorySizeCONST(fact.numCols, fact.numVals, fact.lossy);
				else
					return Long.MAX_VALUE;
			default:
				throw new NotImplementedException("The col compression Type is not yet supported");
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_bestCompressionType);
		sb.append("  ");
		sb.append(Arrays.toString(_columns));
		sb.append("\n" + _sizes);
		return sb.toString();
	}
}
