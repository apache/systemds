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

import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSizes;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;

/**
 * Information collected about a specific ColGroup's compression size.
 */
public class CompressedSizeInfoColGroup {

	protected static final Log LOG = LogFactory.getLog(CompressedSizeInfoColGroup.class.getName());

	private final IColIndex _cols;
	private final EstimationFactors _facts;
	private final double _minSize;
	private final CompressionType _bestCompressionType;
	private final EnumMap<CompressionType, Double> _sizes;

	/**
	 * Map containing a mapping to unique values, but not necessarily the actual values contained in this column group
	 */
	private IEncode _map;

	public CompressedSizeInfoColGroup(IColIndex cols, int nVal, int nRow, CompressionType bestCompressionType) {
		_cols = cols;
		_facts = new EstimationFactors(nVal, nRow);
		_minSize = -1;
		_bestCompressionType = bestCompressionType;
		_sizes = new EnumMap<>(CompressionType.class);
		_sizes.put(bestCompressionType, _minSize);
	}

	public CompressedSizeInfoColGroup(IColIndex cols, EstimationFactors facts, CompressionType bestCompressionType) {
		_cols = cols;
		_facts = facts;
		_minSize = -1;
		_bestCompressionType = bestCompressionType;
		_sizes = new EnumMap<>(CompressionType.class);
		_sizes.put(bestCompressionType, _minSize);
	}

	public CompressedSizeInfoColGroup(IColIndex cols, EstimationFactors facts, long minSize,
		CompressionType bestCompressionType) {
		_cols = cols;
		_facts = facts;
		_minSize = minSize;
		_bestCompressionType = bestCompressionType;
		_sizes = new EnumMap<>(CompressionType.class);
		_sizes.put(bestCompressionType, _minSize);
	}

	public CompressedSizeInfoColGroup(IColIndex columns, EstimationFactors facts, long minSize,
		CompressionType bestCompression, IEncode map) {
		_cols = columns;
		_facts = facts;
		_minSize = minSize;
		_bestCompressionType = bestCompression;
		_sizes = new EnumMap<>(CompressionType.class);
		_sizes.put(bestCompression, _minSize);
		_map = map;
	}

	public CompressedSizeInfoColGroup(IColIndex columns, EstimationFactors facts,
		Set<CompressionType> validCompressionTypes, IEncode map) {
		_cols = columns;
		_facts = facts;
		_sizes = calculateCompressionSizes(_cols, facts, validCompressionTypes);

		CompressionType tmpBestCompressionType = CompressionType.UNCOMPRESSED;
		double tmpBestCompressionSize = _sizes.getOrDefault(tmpBestCompressionType, Double.MAX_VALUE);
		for(Map.Entry<CompressionType, Double> ent : _sizes.entrySet()) {
			if(ent.getValue() < tmpBestCompressionSize) {
				tmpBestCompressionType = ent.getKey();
				tmpBestCompressionSize = ent.getValue();
			}
		}

		_bestCompressionType = tmpBestCompressionType;
		_minSize = tmpBestCompressionSize;
		_map = map;
	}

	/**
	 * Create empty or const.
	 * 
	 * @param columns columns
	 * @param nRows   number of rows
	 * @param ct      The type intended either Empty or Const
	 */
	public CompressedSizeInfoColGroup(IColIndex columns, int nRows, CompressionType ct) {
		_cols = columns;
		_facts = new EstimationFactors(0, nRows);
		_sizes = new EnumMap<>(CompressionType.class);
		switch(ct) {
			case EMPTY:
				_sizes.put(ct, (double) ColGroupSizes.estimateInMemorySizeEMPTY(columns.size(), columns.isContiguous()));
				break;
			case CONST:
				_sizes.put(ct,
					(double) ColGroupSizes.estimateInMemorySizeCONST(columns.size(), columns.isContiguous(), 1.0, false));
				break;
			case UNCOMPRESSED:
				_sizes.put(ct, (double) ColGroupSizes.estimateInMemorySizeUncompressed(nRows, columns.isContiguous(),
					columns.size(), 1.0));
				break;
			default:
				throw new DMLCompressionException("Invalid instantiation of const Cost");
		}
		_bestCompressionType = ct;
		_minSize = _sizes.get(ct);
		_map = null;
	}

	public double getCompressionSize(CompressionType ct) {
		if(_sizes != null) {
			Double s = _sizes.get(ct);
			if(s == null)
				throw new DMLCompressionException("Asked for valid " + ct + " but got null. contains:" + _sizes);
			return s;
		}
		else
			throw new DMLCompressionException("There was no encodings analyzed");
	}

	public CompressionType getBestCompressionType(CompressionSettings cs) {
		return _bestCompressionType;
	}

	public CompressionType getBestCompressionType() {
		return _bestCompressionType;
	}

	public Map<CompressionType, Double> getAllCompressionSizes() {
		return _sizes;
	}

	public double getMinSize() {
		return _minSize;
	}

	/**
	 * Note cardinality is the same as number of distinct values.
	 * 
	 * @return cardinality or number of distinct values.
	 */
	public int getNumVals() {
		return (_facts != null) ? _facts.numVals : -1;
	}

	/**
	 * Number of offsets, or number of non zero values.
	 * 
	 * @return Number of non zeros or number of values.
	 */
	public int getNumOffs() {
		return _facts.numOffs;
	}

	public IColIndex getColumns() {
		return _cols;
	}

	public int getNumRows() {
		return _facts.numRows;
	}

	public double getMostCommonFraction() {
		return (double) _facts.largestOff / _facts.numRows;
	}

	public int getLargestOffInstances() {
		return _facts.largestOff;
	}

	public double getTupleSparsity() {
		return _facts.tupleSparsity;
	}

	public EstimationFactors getFacts() {
		return _facts;
	}

	public IEncode getMap() {
		return _map;
	}

	public void setMap(IEncode map) {
		_map = map;
	}

	public boolean containsZeros() {
		return _facts.numOffs < _facts.numRows;
	}

	private static EnumMap<CompressionType, Double> calculateCompressionSizes(IColIndex cols, EstimationFactors fact,
		Set<CompressionType> validCompressionTypes) {
		if(validCompressionTypes.size() > 10 )
			throw new DMLCompressionException("Invalid big number of compression types");
		EnumMap<CompressionType, Double> res = new EnumMap<>(CompressionType.class);
		for(CompressionType ct : validCompressionTypes) {
			double compSize = getCompressionSize(cols, ct, fact);
			if(compSize > 0)
				res.put(ct, compSize);
		}
		return res;
	}

	public boolean isEmpty() {
		return _bestCompressionType == CompressionType.EMPTY || _sizes.containsKey(CompressionType.EMPTY);
	}

	public boolean isConst() {
		return _bestCompressionType == CompressionType.CONST || _sizes.containsKey(CompressionType.CONST);
	}

	public boolean isIncompressable() {
		return _bestCompressionType == CompressionType.UNCOMPRESSED;
	}

	private static double getCompressionSize(IColIndex cols, CompressionType ct, EstimationFactors fact) {
		int nv;
		final int numCols = cols.size();
		final boolean contiguousColumns = cols.isContiguous();
		switch(ct) {
			case LinearFunctional:
				return ColGroupSizes.estimateInMemorySizeLinearFunctional(numCols, contiguousColumns);
			case DeltaDDC:
				throw new NotImplementedException();
			case DDC:
				nv = fact.numVals + (fact.numOffs < fact.numRows ? 1 : 0);
				return ColGroupSizes.estimateInMemorySizeDDC(numCols, contiguousColumns, nv, fact.numRows,
					fact.tupleSparsity, fact.lossy);
			case RLE:
				return ColGroupSizes.estimateInMemorySizeRLE(numCols, contiguousColumns, fact.numVals, fact.numRuns,
					fact.numRows, fact.tupleSparsity, fact.lossy);
			case OLE:
				nv = fact.numVals + (fact.zeroIsMostFrequent ? 1 : 0);
				return ColGroupSizes.estimateInMemorySizeOLE(numCols, contiguousColumns, nv, fact.numOffs + fact.numVals,
					fact.numRows, fact.tupleSparsity, fact.lossy);
			case UNCOMPRESSED:
				return ColGroupSizes.estimateInMemorySizeUncompressed(fact.numRows, contiguousColumns, numCols,
					fact.overAllSparsity);
			case SDC:
				return ColGroupSizes.estimateInMemorySizeSDC(numCols, contiguousColumns, fact.numVals, fact.numRows,
					fact.largestOff, fact.tupleSparsity, fact.zeroIsMostFrequent, fact.lossy);
			case CONST:
				if(fact.numOffs == fact.numRows && fact.numVals == 1)
					return ColGroupSizes.estimateInMemorySizeCONST(numCols, contiguousColumns, fact.tupleSparsity,
						fact.lossy);
				else
					return -1;
			case EMPTY:
				if(fact.numOffs == 0)
					return ColGroupSizes.estimateInMemorySizeEMPTY(numCols, contiguousColumns);
				else
					return -1;
			default:
				throw new NotImplementedException("The col compression Type is not yet supported");
		}
	}

	public void clearMap() {
		_map = null;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" cols: " + _cols);
		sb.append(String.format(" common: %4.3f", getMostCommonFraction()));
		sb.append(" Sizes: " + _sizes);
		sb.append(" facts: " + _facts);
		sb.append(" mapIsNull: " + (_map == null));
		if(_map != null) {
			String s = _map.toString();
			if(s.length() > 1000) {
				sb.append(s, 0, 1000);
			}
			else {
				sb.append(s);
			}
		}
		return sb.toString();
	}

}
