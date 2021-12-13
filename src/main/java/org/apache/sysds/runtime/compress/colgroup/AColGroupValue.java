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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Base class for column groups encoded with value dictionary. This include column groups such as DDC OLE and RLE.
 * 
 */
public abstract class AColGroupValue extends AColGroupCompressed implements Cloneable {
	private static final long serialVersionUID = -6835757655517301955L;

	/** The number of rows in the column group */
	final protected int _numRows;

	/**
	 * ColGroup Implementation Contains zero tuple. Note this is not if it contains a zero value. If false then the
	 * stored values are filling the ColGroup making it a dense representation, that can be leveraged in operations.
	 */
	protected boolean _zeros = false;

	/** Distinct value tuples associated with individual bitmaps. */
	protected transient ADictionary _dict;

	/** The count of each distinct value contained in the dictionary */
	private transient SoftReference<int[]> counts;

	protected AColGroupValue(int numRows) {
		super();
		_numRows = numRows;
	}

	/**
	 * A Abstract class for column groups that contain ADictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param numRows      The number of rows contained in this group
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected AColGroupValue(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices);
		_numRows = numRows;
		_dict = dict;
		if(cachedCounts == null)
			counts = null;
		else
			counts = new SoftReference<>(cachedCounts);
	}

	@Override
	public final void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			if(mb.isEmpty()) // Early abort if the dictionary is empty.
				return;
			else if(mb.isInSparseFormat())
				decompressToDenseBlockSparseDictionary(db, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else
			decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, _dict.getValues());
	}

	@Override
	public final void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			if(mb.isEmpty()) // Early abort if the dictionary is empty.
				return;
			else if(mb.isInSparseFormat())
				decompressToSparseBlockSparseDictionary(sb, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToSparseBlockDenseDictionary(sb, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else
			decompressToSparseBlockDenseDictionary(sb, rl, ru, offR, offC, _dict.getValues());
	}

	/**
	 * Decompress to DenseBlock using a sparse dictionary to lookup into.
	 * 
	 * @param db   The dense db block to decompress into
	 * @param rl   The row to start decompression from
	 * @param ru   The row to end decompression at
	 * @param offR The row offset to insert into
	 * @param offC The column offset to insert into
	 * @param sb   The sparse dictionary block to take value tuples from
	 */
	protected abstract void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb);

	/**
	 * Decompress to DenseBlock using a dense dictionary to lookup into.
	 * 
	 * @param db     The dense db block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offR   The row offset to insert into
	 * @param offC   The column offset to insert into
	 * @param values The dense dictionary values, linearized row major.
	 */
	protected abstract void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values);

	/**
	 * Decompress to SparseBlock using a sparse dictionary to lookup into.
	 * 
	 * @param ret  The dense ret block to decompress into
	 * @param rl   The row to start decompression from
	 * @param ru   The row to end decompression at
	 * @param offR The row offset to insert into
	 * @param offC The column offset to insert into
	 * @param sb   The sparse dictionary block to take value tuples from
	 */
	protected abstract void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb);

	/**
	 * Decompress to SparseBlock using a dense dictionary to lookup into.
	 * 
	 * @param ret    The dense ret block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offR   The row offset to insert into
	 * @param offC   The column offset to insert into
	 * @param values The dense dictionary values, linearized row major.
	 */
	protected abstract void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values);

	@Override
	public int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.length);
	}

	public final ADictionary getDictionary() {
		return _dict;
	}

	public final MatrixBlock getValuesAsBlock() {
		_dict = _dict.getMBDict(_colIndexes.length);
		MatrixBlock ret = ((MatrixBlockDictionary) _dict).getMatrixBlock();
		if(_zeros) {
			MatrixBlock tmp = new MatrixBlock();
			ret.append(new MatrixBlock(1, _colIndexes.length, 0), tmp, false);
			return tmp;
		}
		return ret;
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
			ret = getCounts(new int[getNumValues() + (_zeros ? 1 : 0)]);
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
	public final int[] getCachedCounts() {
		return counts != null ? counts.get() : null;
	}

	private int[] rightMMGetColsDense(double[] b, int cl, int cu, int cut) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();
		final int retCols = (cu - cl);
		for(int k = 0; k < _colIndexes.length; k++) {
			int rowIdxOffset = _colIndexes[k] * cut;
			for(int h = cl; h < cu; h++) {
				double v = b[rowIdxOffset + h];
				if(v != 0.0) {
					aggregateColumnsSet.add(h);
				}
			}
			if(aggregateColumnsSet.size() == retCols)
				break;
		}

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return aggregateColumns;
	}

	private int[] rightMMGetColsSparse(SparseBlock b, int retCols) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();

		for(int h = 0; h < _colIndexes.length; h++) {
			int colIdx = _colIndexes[h];
			if(!b.isEmpty(colIdx)) {
				int[] sIndexes = b.indexes(colIdx);
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
					aggregateColumnsSet.add(sIndexes[i]);
				}
			}
			if(aggregateColumnsSet.size() == retCols)
				break;
		}

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return aggregateColumns;
	}

	private double[] rightMMPreAggSparse(int numVals, SparseBlock b, int[] aggregateColumns, int cl, int cu, int cut) {
		final double[] ret = new double[numVals * aggregateColumns.length];
		for(int h = 0; h < _colIndexes.length; h++) {
			int colIdx = _colIndexes[h];
			if(!b.isEmpty(colIdx)) {
				double[] sValues = b.values(colIdx);
				int[] sIndexes = b.indexes(colIdx);
				int retIdx = 0;
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
					while(aggregateColumns[retIdx] < sIndexes[i])
						retIdx++;
					if(sIndexes[i] == aggregateColumns[retIdx])
						for(int j = 0, offOrg = h;
							j < numVals * aggregateColumns.length;
							j += aggregateColumns.length, offOrg += _colIndexes.length) {
							ret[j + retIdx] += _dict.getValue(offOrg) * sValues[i];
						}
				}
			}
		}
		return ret;
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		if(_zeros)
			c = builtin.execute(c, 0);
		return _dict.aggregate(c, builtin);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		if(_zeros)
			for(int x = 0; x < _colIndexes.length; x++)
				c[_colIndexes[x]] = builtin.execute(c[_colIndexes[x]], 0);

		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_zeros = in.readBoolean();
		_dict = DictionaryFactory.read(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		out.writeBoolean(_zeros);
		_dict.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 1; // zeros boolean
		ret += _dict.getExactSizeOnDisk();

		return ret;
	}

	public abstract int[] getCounts(int[] out);

	@Override
	protected void computeSum(double[] c, int nRows) {
		c[0] += _dict.sum(getCounts(), _colIndexes.length);
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		_dict.colSum(c, getCounts(), _colIndexes);
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		c[0] += _dict.sumSq(getCounts(), _colIndexes.length);
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		_dict.colSumSq(c, getCounts(), _colIndexes);
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		c[0] *= _dict.product(getCounts(), _colIndexes.length);
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggSumRows(){
		return _dict.sumAllRowsToDouble(_colIndexes.length);
	}

	@Override
	protected double[] preAggSumSqRows(){
		return _dict.sumAllRowsToDoubleSq(_colIndexes.length);
	}

	@Override
	protected double[] preAggProductRows(){
		throw new NotImplementedException();
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin){
		return _dict.aggregateRows(builtin, _colIndexes.length);
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProduct(c, getCounts(), _colIndexes);
	}

	protected Object clone() {
		try {
			return super.clone();
		}
		catch(CloneNotSupportedException e) {
			throw new DMLCompressionException("Error while cloning: " + getClass().getSimpleName(), e);
		}
	}

	public AColGroup copyAndSet(double[] newDictionary) {
		return copyAndSet(new Dictionary(newDictionary));
	}

	public AColGroup copyAndSet(ADictionary newDictionary) {
		AColGroupValue clone = (AColGroupValue) this.clone();
		clone._dict = newDictionary;
		return clone;
	}

	public AColGroup copyAndSet(int[] colIndexes, double[] newDictionary) {
		return copyAndSet(colIndexes, new Dictionary(newDictionary));
	}

	public AColGroup copyAndSet(int[] colIndexes, ADictionary newDictionary) {
		AColGroupValue clone = (AColGroupValue) this.clone();
		clone._dict = newDictionary;
		clone.setColIndices(colIndexes);
		return clone;
	}

	@Override
	public AColGroupValue copy() {
		return (AColGroupValue) this.clone();
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		final AColGroupValue ret = (AColGroupValue) copy();
		ret._colIndexes = new int[] {0};
		if(_colIndexes.length == 1)
			ret._dict = ret._dict.clone();
		else
			ret._dict = ret._dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.length);

		return ret;
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		final AColGroupValue ret = (AColGroupValue) copy();
		ret._dict = ret._dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.length);
		ret._colIndexes = outputCols;
		return ret;
	}

	@Override
	protected void tsmm(double[] result, int numColumns, int nRows) {
		final int[] counts = getCounts();
		tsmm(result, numColumns, counts, _dict, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		if(pattern == 0 && _zeros)
			return true;
		return _dict.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		int[] counts = getCounts();
		return _dict.getNumberNonZeros(counts, _colIndexes.length);
	}

	public synchronized void forceMatrixBlockDictionary() {
		if(!(_dict instanceof MatrixBlockDictionary))
			_dict = _dict.getMBDict(_colIndexes.length);
	}

	@Override
	public final AColGroup rightMultByMatrix(MatrixBlock right) {

		if(right.isEmpty())
			return null;
		final int cl = 0;
		final int cr = right.getNumColumns();
		final int numVals = getNumValues();
		if(right.isInSparseFormat()) {
			final SparseBlock sb = right.getSparseBlock();
			final int[] agCols = rightMMGetColsSparse(sb, cr);
			if(agCols.length == 0)
				return null;
			return copyAndSet(agCols, rightMMPreAggSparse(numVals, sb, agCols, cl, cr, cr));
		}
		else {
			final double[] rightV = right.getDenseBlockValues();
			final int[] agCols = rightMMGetColsDense(rightV, cl, cr, cr);
			if(agCols.length == 0)
				return null;
			ADictionary d = _dict.preaggValuesFromDense(numVals, _colIndexes, agCols, rightV, cr);
			if(d == null)
				return null;
			return copyAndSet(agCols, d);
		}
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += 8; // Dictionary Reference.
		size += 8; // Counts reference
		size += 4; // Int nRows
		size += 1; // _zeros boolean reference
		size += 1; // _lossy boolean reference
		size += 2; // padding
		size += _dict.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		return copyAndSet(replaced);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%s", "Values: " , _dict.getClass().getSimpleName()));
		sb.append(_dict.getString(_colIndexes.length));
		return sb.toString();
	}
}
