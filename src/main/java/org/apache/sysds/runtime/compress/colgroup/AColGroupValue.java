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

package org.apache.sysds.runtime.compress.colgroup;

import java.lang.ref.SoftReference;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.operators.CMOperator;

public abstract class AColGroupValue extends ADictBasedColGroup {
	private static final long serialVersionUID = -6835757655517301955L;

	/** The count of each distinct value contained in the dictionary */
	private SoftReference<int[]> counts = null;

	/**
	 * A abstract class for column groups that contain IDictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected AColGroupValue(IColIndex colIndices, IDictionary dict, int[] cachedCounts) {
		super(colIndices, dict);
		if(cachedCounts != null)
			counts = new SoftReference<>(cachedCounts);
	}

	@Override
	public int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.size());
	}

	/**
	 * Returns the counts of values inside the dictionary. If already calculated it will return the previous counts. This
	 * produce an overhead in cases where the count is calculated, but the overhead will be limited to number of distinct
	 * tuples in the dictionary.
	 * 
	 * The returned counts always contains the number of zero tuples as well if there are some contained, even if they
	 * are not materialized.
	 *
	 * @return The count of each value in the MatrixBlock.
	 */
	public final int[] getCounts() {
		int[] ret = getCachedCounts();
		if(ret == null) {
			ret = getCounts(new int[getNumValues()]);
			counts = new SoftReference<>(ret);
		}
		return ret;
	}

	/**
	 * Get the cached counts.
	 * 
	 * If they are not materialized or the garbage collector have removed them, then null is returned.
	 * 
	 * @return The counts or null.
	 */
	protected final int[] getCachedCounts() {

		return counts != null ? counts.get() : null;
	}

	protected abstract int[] getCounts(int[] out);

	@Override
	protected void computeSum(double[] c, int nRows) {
		c[0] += _dict.sum(getCounts(), _colIndexes.size());
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		_dict.colSum(c, getCounts(), _colIndexes);
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSq(getCounts(), _colIndexes.size());
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict.colSumSq(c, getCounts(), _colIndexes);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		_dict.product(c, getCounts(), _colIndexes.size());
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProduct(c, getCounts(), _colIndexes);
	}

	@Override
	protected double[] preAggSumRows() {
		return _dict.sumAllRowsToDouble(_colIndexes.size());
	}

	@Override
	protected double[] preAggSumSqRows() {
		return _dict.sumAllRowsToDoubleSq(_colIndexes.size());
	}

	@Override
	protected double[] preAggProductRows() {
		return _dict.productAllRowsToDouble(_colIndexes.size());
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return _dict.aggregateRows(builtin, _colIndexes.size());
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		final IColIndex retIndexes = ColIndexFactory.create(1);
		if(_colIndexes.size() == 1)
			return copyAndSet(retIndexes, _dict);

		final IDictionary retDict = _dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.size());
		if(retDict == null)
			return new ColGroupEmpty(retIndexes);
		else
			return copyAndSet(retIndexes, retDict);

	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		final IDictionary retDict = _dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.size());
		if(retDict == null)
			return new ColGroupEmpty(outputCols);

		if(retDict.getNumberOfValues(outputCols.size()) != getNumValues())
			throw new DMLCompressionException("Invalid Slice Multi Columns");

		return copyAndSet(outputCols, retDict);
	}

	@Override
	protected void tsmm(double[] result, int numColumns, int nRows) {
		final int[] counts = getCounts();
		tsmm(result, numColumns, counts, _dict, _colIndexes);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		int[] counts = getCounts();
		return _dict.getNumberNonZeros(counts, _colIndexes.size());
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += 8; // Counts reference
		return size;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		IDictionary replaced = _dict.replace(pattern, replace, _colIndexes.size());
		return copyAndSet(replaced);
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		return _dict.centralMoment(op.fn, getCounts(), nRows);
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		IDictionary d = _dict.rexpandCols(max, ignore, cast, _colIndexes.size());
		if(d == null) {
			if(max <= 0)
				return null;
			return ColGroupEmpty.create(max);
		}
		else {
			IColIndex outCols = ColIndexFactory.create(d.getNumberOfColumns(_dict.getNumberOfValues(1)));
			return copyAndSet(outCols, d);
		}
	}

	@Override
	public void clear() {
		counts = null;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%s", "Values: ", _dict.getClass().getSimpleName()));
		sb.append(_dict.getString(_colIndexes.size()));
		return sb.toString();
	}

}
