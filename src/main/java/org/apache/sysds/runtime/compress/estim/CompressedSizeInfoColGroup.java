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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSizes;

/**
 * Information collected about a specific ColGroup's compression size.
 */
public class CompressedSizeInfoColGroup {

	private final int _numVals;
	private final int _numOffs;
	private final long _minSize;
	private final CompressionType _bestCompressionType;
	private final Map<CompressionType, Long> _sizes;

	public CompressedSizeInfoColGroup(EstimationFactors fact, Set<CompressionType> validCompressionTypes) {
		_numVals = fact.numVals;
		_numOffs = fact.numOffs;
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
	public int getEstCard() {
		return _numVals;
	}

	/**
	 * Number of offsets, or number of non zero values.
	 * 
	 * @return Number of non zeros or number of values.
	 */
	public int getEstNnz() {
		return _numOffs;
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
		long size = 0;
		switch(ct) {
			case DDC:
				if(fact.numVals < 256) {
					size = ColGroupSizes.estimateInMemorySizeDDC1(fact.numCols, fact.numVals, fact.numRows, fact.lossy);
				}
				else {
					size = ColGroupSizes.estimateInMemorySizeDDC2(fact.numCols, fact.numVals, fact.numRows, fact.lossy);
				}
				break;
			case RLE:
				size = ColGroupSizes
					.estimateInMemorySizeRLE(fact.numCols, fact.numVals, fact.numRuns, fact.numRows, fact.lossy);
				break;
			case OLE:
				size = ColGroupSizes
					.estimateInMemorySizeOLE(fact.numCols, fact.numVals, fact.numOffs, fact.numRows, fact.lossy);
				break;
			case UNCOMPRESSED:
				size = ColGroupSizes.estimateInMemorySizeUncompressed(fact.numRows,
					fact.numCols,
					((double) fact.numVals / (fact.numRows * fact.numCols)));
				break;
			default:
				throw new NotImplementedException("The col compression Type is not yet supported");
		}
		return size;
	}
}
