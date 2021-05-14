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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.pre.ArrPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Base class for column groups encoded with value dictionary. This include column groups such as DDC OLE and RLE.
 * 
 */
public abstract class ColGroupValue extends ColGroupCompressed implements Cloneable {
	private static final long serialVersionUID = 3786247536054353658L;

	/** thread-local pairs of reusable temporary vectors for positions and values */
	private static ThreadLocal<Pair<int[], double[]>> memPool = new ThreadLocal<Pair<int[], double[]>>() {
		@Override
		protected Pair<int[], double[]> initialValue() {
			return null;
		}
	};

	/**
	 * ColGroup Implementation Contains zero tuple. Note this is not if it contains a zero value. If false then the
	 * stored values are filling the ColGroup making it a dense representation, that can be leveraged in operations.
	 */
	protected boolean _zeros = false;

	/** Distinct value tuples associated with individual bitmaps. */
	protected ADictionary _dict;

	/** The count of each distinct value contained in the dictionary */
	private int[] counts;

	protected ColGroupValue(int numRows) {
		super(numRows);
	}

	protected ColGroupValue(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices, numRows);
		_dict = dict;
		counts = cachedCounts;
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		decompressToBlock(target, rl, ru, rl);
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT) {
		decompressToBlock(target, rl, ru, offT, getValues());
	}

	@Override
	public final int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.length);
	}

	@Override
	public final double[] getValues() {
		return _dict != null ? _dict.getValues() : null;
	}

	public final ADictionary getDictionary() {
		return _dict;
	}

	@Override
	public final void addMinMax(double[] ret) {
		_dict.addMaxAndMin(ret, _colIndexes);
	}

	protected void setDictionary(ADictionary dict) {
		_dict = dict;
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		final double[] values = getValues();
		int vlen = values.length;
		int rlen = _zeros ? vlen + 1 : vlen;
		MatrixBlock ret = new MatrixBlock(rlen, 1, false);
		for(int i = 0; i < vlen; i++)
			ret.quickSetValue(i, 0, values[i]);
		return ret;
	}

	/**
	 * Returns the counts of values inside the dictionary. If already calculated it will return the previous counts.
	 * This produce an overhead in cases where the count is calculated, but the overhead will be limited to number of
	 * distinct tuples in the dictionary.
	 * 
	 * The returned counts always contains the number of zeros as well if there are some contained, even if they are not
	 * materialized.
	 *
	 * @return the count of each value in the MatrixBlock.
	 */
	public final int[] getCounts() {

		if(counts == null && _dict != null) {
			counts = getCounts(new int[getNumValues() + (_zeros ? 1 : 0)]);
			return counts;
		}
		else
			return counts;

	}

	public final int[] getCachedCounts() {
		return counts;
	}

	/**
	 * Returns the counts of values inside the MatrixBlock returned in getValuesAsBlock Throws an exception if the
	 * getIfCountsType is false.
	 * 
	 * The returned counts always contains the number of zeros as well if there are some contained, even if they are not
	 * materialized.
	 *
	 * @param rl the lower index of the interval of rows queried
	 * @param ru the the upper boundary of the interval of rows queried
	 * @return the count of each value in the MatrixBlock.
	 */
	public final int[] getCounts(int rl, int ru) {
		int[] tmp;
		if(_zeros) {
			tmp = allocIVector(getNumValues() + 1, true);
		}
		else {
			tmp = allocIVector(getNumValues(), true);
		}
		return getCounts(rl, ru, tmp);
	}

	public boolean getIfCountsType() {
		return true;
	}

	@Override
	protected int containsAllZeroTuple() {
		return _dict.hasZeroTuple(_colIndexes.length);
	}

	protected final double sumValues(int valIx, double[] b, double[] dictVals) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		double val = 0;
		for(int i = 0; i < numCols; i++)
			val += dictVals[valOff + i] * b[_colIndexes[i]];
		return val;
	}

	protected final double sumValues(int valIx, double[] b, double[] dictVals, int off) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		double val = 0;
		for(int i = 0; i < numCols; i++)
			val += dictVals[valOff + i] * b[_colIndexes[i] + off];
		return val;
	}

	protected double[] preaggValues(int numVals, double[] b, double[] dictVals) {
		return preaggValues(numVals, b, false, dictVals, 0);
	}

	protected double[] preaggValues(int numVals, double[] b, double[] dictVals, int off) {
		return preaggValues(numVals, b, false, dictVals, off);
	}

	protected double[] preaggValues(int numVals, double[] b, boolean allocNew, double[] dictVals, int off) {
		// + 1 to enable containing a zero value. which we have added at the length of
		// the arrays index.
		double[] ret = allocNew ? new double[numVals + 1] : allocDVector(numVals + 1, true);

		if(_colIndexes.length == 1) {
			for(int k = 0; k < numVals; k++)
				ret[k] = dictVals[k] * b[_colIndexes[0] + off];
		}
		else {
			for(int k = 0; k < numVals; k++)
				ret[k] = sumValues(k, b, dictVals, off);
		}

		return ret;
	}

	private int[] getAggregateColumnsSetDense(double[] b, int cl, int cu, int cut) {
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

	private Pair<int[], double[]> preaggValuesFromDense(final int numVals, final double[] b, final int cl, final int cu,
		final int cut) {

		final int[] aggregateColumns = getAggregateColumnsSetDense(b, cl, cu, cut);
		final double[] ret = new double[numVals * aggregateColumns.length];

		for(int k = 0, off = 0;
			k < numVals * _colIndexes.length;
			k += _colIndexes.length, off += aggregateColumns.length) {
			for(int h = 0; h < _colIndexes.length; h++) {
				int idb = _colIndexes[h] * cut;
				double v = _dict.getValue(k + h);
				if(v != 0)
					for(int i = 0; i < aggregateColumns.length; i++)
						ret[off + i] += v * b[idb + aggregateColumns[i]];
			}
		}

		return new ImmutablePair<>(aggregateColumns, ret);
	}

	private int[] getAggregateColumnsSetSparse(SparseBlock b) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();

		for(int h = 0; h < _colIndexes.length; h++) {
			int colIdx = _colIndexes[h];
			if(!b.isEmpty(colIdx)) {
				int[] sIndexes = b.indexes(colIdx);
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
					aggregateColumnsSet.add(sIndexes[i]);
				}
			}
		}

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return aggregateColumns;
	}

	private Pair<int[], double[]> preaggValuesFromSparse(int numVals, SparseBlock b, int cl, int cu, int cut) {

		int[] aggregateColumns = getAggregateColumnsSetSparse(b);

		double[] ret = new double[numVals * aggregateColumns.length];

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
		return new ImmutablePair<>(aggregateColumns, ret);
	}

	public Pair<int[], double[]> preaggForRightMultiplyValues(int numVals, MatrixBlock b, int cl, int cu, int cut) {
		return b.isInSparseFormat() ? preaggValuesFromSparse(numVals, b.getSparseBlock(), cl, cu,
			cut) : preaggValuesFromDense(numVals, b.getDenseBlockValues(), cl, cu, cut);
	}

	// protected static double[] sparsePreaggValues(int numVals, double v, boolean allocNew, ADictionary dict) {
	// double[] ret = allocNew ? new double[numVals + 1] : allocDVector(numVals + 1, true);

	// for(int k = 0; k < numVals; k++)
	// ret[k] = dictVals[k] * v;
	// return ret;
	// }

	protected double computeMxx(double c, Builtin builtin) {
		if(_zeros)
			c = builtin.execute(c, 0);
		if(_dict != null)
			return _dict.aggregate(c, builtin);
		else
			return c;
	}

	protected void computeColMxx(double[] c, Builtin builtin) {
		if(_zeros) {
			for(int x = 0; x < _colIndexes.length; x++)
				c[_colIndexes[x]] = builtin.execute(c[_colIndexes[x]], 0);
		}
		if(_dict != null)
			_dict.aggregateCols(c, builtin, _colIndexes);
	}

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value metadata stored in the dictionary.
	 * 
	 * @param op scalar operation to perform
	 * @return transformed copy of value metadata for this column group
	 */
	protected ADictionary applyScalarOp(ScalarOperator op) {
		return _dict.clone().apply(op);
	}

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value metadata stored in the dictionary. This
	 * specific method is used in cases where an new entry is to be added in the dictionary.
	 * 
	 * Method should only be called if the newVal is not 0! Also the newVal should already have the operator applied.
	 * 
	 * @param op      The Operator to apply to the underlying data.
	 * @param newVal  The new Value to append to the underlying data.
	 * @param numCols The number of columns in the ColGroup, to specify how many copies of the newVal should be
	 *                appended.
	 * @return The new Dictionary containing the values.
	 */
	protected ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		return _dict.applyScalarOp(op, newVal, numCols);
	}

	/**
	 * Apply the binary row-wise operator to the dictionary, and copy it appropriately if needed.
	 * 
	 * @param fn         The function to apply.
	 * @param v          The vector to apply on each tuple of the dictionary.
	 * @param sparseSafe Specify if the operation is sparseSafe. if false then allocate a new tuple.
	 * @param left       Specify which side the operation is executed on.
	 * @return The new Dictionary with values.
	 */
	public ADictionary applyBinaryRowOp(ValueFunction fn, double[] v, boolean sparseSafe, boolean left) {
		return sparseSafe ? _dict.clone().applyBinaryRowOp(fn, v, sparseSafe, _colIndexes, left) : _dict
			.applyBinaryRowOp(fn, v, sparseSafe, _colIndexes, left);
	}

	protected void setandExecute(double[] c, boolean square, double val, int rix) {
		if(square)
			c[rix] += val * val;
		else
			c[rix] += val;
	}

	public static void setupThreadLocalMemory(int len) {
		if(memPool.get() == null || memPool.get().getLeft().length < len) {
			Pair<int[], double[]> p = new ImmutablePair<>(new int[len], new double[len]);
			memPool.set(p);
		}
	}

	public static void cleanupThreadLocalMemory() {
		memPool.remove();
	}

	protected static double[] allocDVector(int len, boolean reset) {
		Pair<int[], double[]> p = memPool.get();

		// sanity check for missing setup
		if(p == null) {
			return new double[len];
		}

		if(p.getValue().length < len) {
			setupThreadLocalMemory(len);
			return p.getValue();
		}

		// get and reset if necessary
		double[] tmp = p.getValue();
		if(reset)
			Arrays.fill(tmp, 0, len, 0);
		return tmp;
	}

	protected static int[] allocIVector(int len, boolean reset) {
		Pair<int[], double[]> p = memPool.get();

		// sanity check for missing setup
		if(p == null)
			return new int[len + 1];

		if(p.getKey().length < len) {
			setupThreadLocalMemory(len);
			return p.getKey();
		}

		int[] tmp = p.getKey();
		if(reset)
			Arrays.fill(tmp, 0, len, 0);
		return tmp;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" Is Lossy: " + _dict.isLossy() + " num Rows: " + getNumRows() + " contain zero row:" + _zeros);
		sb.append(super.toString());
		if(_dict != null) {
			sb.append(String.format("\n%15s%5d ", "Values:", _dict.getValues().length));
			sb.append(_dict.getString(_colIndexes.length));
		}
		return sb.toString();
	}

	@Override
	public boolean isLossy() {
		return _dict.isLossy();
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
		out.writeBoolean(_dict.isLossy());
		_dict.write(out);

	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 1; // zeros boolean
		ret += 1; // lossy boolean
		// distinct values (groups of values)
		ret += 1; // Dict exists boolean
		if(_dict != null)
			ret += _dict.getExactSizeOnDisk();

		return ret;
	}

	public abstract int[] getCounts(int[] out);

	public abstract int[] getCounts(int rl, int ru, int[] out);

	protected void computeSum(double[] c, boolean square) {
		if(_dict != null)
			if(square)
				c[0] += _dict.sumsq(getCounts(), _colIndexes.length);
			else
				c[0] += _dict.sum(getCounts(), _colIndexes.length);
	}

	protected abstract void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean);

	protected void computeColSums(double[] c, boolean square) {
		_dict.colSum(c, getCounts(), _colIndexes, square);
	}

	protected abstract void computeRowMxx(double[] c, Builtin builtin, int rl, int ru);

	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}

	public AColGroup copyAndSet(double[] newDictionary) {
		try {
			ColGroupValue clone = (ColGroupValue) this.clone();
			clone.setDictionary(new Dictionary(newDictionary));
			return clone;
		}
		catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}

	public AColGroup copyAndSet(ADictionary newDictionary) {
		try {
			ColGroupValue clone = (ColGroupValue) this.clone();
			clone.setDictionary(newDictionary);
			return clone;
		}
		catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}

	public AColGroup copyAndSet(int[] colIndexes, double[] newDictionary) {
		try {
			ColGroupValue clone = (ColGroupValue) this.clone();
			clone.setDictionary(new Dictionary(newDictionary));
			clone.setColIndices(colIndexes);
			return clone;
		}
		catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}

	public AColGroup copyAndSet(int[] colIndexes, ADictionary newDictionary) {
		try {
			ColGroupValue clone = (ColGroupValue) this.clone();
			clone.setDictionary(newDictionary);
			clone.setColIndices(colIndexes);
			return clone;
		}
		catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public ColGroupValue copy() {
		try {
			ColGroupValue clone = (ColGroupValue) this.clone();
			return clone;
		}
		catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		ColGroupValue ret = (ColGroupValue) copy();
		ret._colIndexes = new int[] {0};
		if(ret._dict != null)
			if(_colIndexes.length == 1)
				ret._dict = ret._dict.clone();
			else
				ret._dict = ret._dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.length);

		return ret;
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {

		ColGroupValue ret = (ColGroupValue) copy();
		ret._dict = ret._dict != null ? ret._dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.length) : null;
		ret._colIndexes = outputCols;

		return ret;
	}

	/**
	 * Post scale for left Multiplication
	 * 
	 * @param dictValues The dictionary values materialized as double array.
	 * @param vals       The values aggregated from the left side row vector.
	 * @param c          The output matrix
	 * @param numVals    The number of values contained in the dictionary.
	 */
	protected void postScaling(double[] dictValues, double[] vals, MatrixBlock c, int numVals) {
		postScaling(dictValues, vals, c, numVals, 0);
	}

	/**
	 * Post scale for left Multiplication
	 * 
	 * @param dictValues The dictionary values materialized as double array.
	 * @param vals       The values aggregated from the left side row vector.
	 * @param c          The output matrix
	 * @param numVals    The number of values contained in the dictionary.
	 * @param row        The row index in the output c to assign the result to.
	 * @param totalCols  The total number of columns in c.
	 */
	protected void postScaling(double[] dictValues, double[] vals, MatrixBlock c, int numVals, int row) {
		final int ncol = getNumCols();
		int valOff = 0;
		final double[] cv = c.getDenseBlockValues();
		final int totalCols = c.getNumColumns();

		for(int k = 0; k < numVals; k++) {
			double aval = vals[k];
			for(int j = 0; j < ncol; j++) {
				int colIx = _colIndexes[j] + row * totalCols;
				cv[colIx] += aval * dictValues[valOff++];
			}
		}
	}

	/**
	 * Pre aggregate a vector
	 * 
	 * @param a The vector to aggregate
	 * @return The pre-aggregated values.
	 */
	public double[] preAggregate(double[] a) {
		return preAggregate(a, 0);
	}

	public abstract MatrixBlock preAggregate(MatrixBlock m, int rl, int ru);

	/**
	 * Pre aggregates for left multiplication
	 * 
	 * @param a   The input dense vector or matrix to aggregate
	 * @param row The row index to aggregate
	 * @return The pre-aggregated values.
	 */
	public abstract double[] preAggregate(double[] a, int row);

	/**
	 * Pre aggregate for left multiplication
	 * 
	 * @param sb The vector to aggregate
	 * @return The pre-aggregated values.
	 */
	public double[] preAggregate(SparseBlock sb) {
		return preAggregateSparseWithCheck(sb, 0);
	}

	private double[] preAggregateSparseWithCheck(SparseBlock sb, int row) {
		if(sb != null && !sb.isEmpty(row))
			return preAggregateSparse(sb, row);
		else
			return null;
	}

	/**
	 * Pre aggregate for left multiplication of sparse vector or matrix.
	 * 
	 * @param sb  The input sparse vector or matrix to aggregate
	 * @param row The row index to aggregate
	 * @return The pre-aggregated values.
	 */
	public abstract double[] preAggregateSparse(SparseBlock sb, int row);

	public abstract int getIndexStructureHash();

	public IPreAggregate preAggregate(ColGroupValue lhs) {
		IPreAggregate r = preCallAggregate(lhs);
		return r;
	}

	public IPreAggregate preCallAggregate(ColGroupValue lhs) {
		// LOG.error(lhs.getClass().getSimpleName() + " in " + this.getClass().getSimpleName() + " "
		// + Arrays.toString(lhs.getColIndices()) + " " + Arrays.toString(this.getColIndices()));

		if(lhs instanceof ColGroupDDC)
			return preAggregateDDC((ColGroupDDC) lhs);
		else if(lhs instanceof ColGroupSDC)
			return preAggregateSDC((ColGroupSDC) lhs);
		else if(lhs instanceof ColGroupSDCSingle)
			return preAggregateSDCSingle((ColGroupSDCSingle) lhs);
		else if(lhs instanceof ColGroupSDCZeros)
			return preAggregateSDCZeros((ColGroupSDCZeros) lhs);
		else if(lhs instanceof ColGroupSDCSingleZeros)
			return preAggregateSDCSingleZeros((ColGroupSDCSingleZeros) lhs);
		else if(lhs instanceof ColGroupOLE)
			return preAggregateOLE((ColGroupOLE) lhs);
		else if(lhs instanceof ColGroupRLE)
			return preAggregateRLE((ColGroupRLE) lhs);
		else if(lhs instanceof ColGroupConst)
			return preAggregateCONST((ColGroupConst) lhs);

		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	public IPreAggregate preAggregateCONST(ColGroupConst lhs) {
		// LOG.error(Arrays.toString(getCounts()));
		return new ArrPreAggregate(getCounts());
	}

	public abstract IPreAggregate preAggregateDDC(ColGroupDDC lhs);

	public abstract IPreAggregate preAggregateSDC(ColGroupSDC lhs);

	public abstract IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs);

	public abstract IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs);

	public abstract IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs);

	public abstract IPreAggregate preAggregateOLE(ColGroupOLE lhs);

	public abstract IPreAggregate preAggregateRLE(ColGroupRLE lhs);

	/**
	 * Pre aggregate into a dictionary. It is assumed that "that" have more distinct values than, "this".
	 * 
	 * @param that      the other column group whose indexes are used for aggregation.
	 * @param preModify specifies if the matrix in this
	 * @return A aggregate dictionary
	 */
	public Dictionary preAggregateThatIndexStructure(ColGroupValue that, boolean preModify) {
		int outputLength = that._colIndexes.length * this.getNumValues();
		Dictionary ret = new Dictionary(new double[outputLength]);

		// if(preModify)
		// LOG.error(preModify + " " + that.getClass().getSimpleName() + " in " + this.getClass().getSimpleName());

		if(that instanceof ColGroupDDC)
			return preAggregateThatDDCStructure((ColGroupDDC) that, ret);
		else if(that instanceof ColGroupSDC)
			return preAggregateThatSDCStructure((ColGroupSDC) that, ret, preModify);
		else if(that instanceof ColGroupSDCSingle)
			return preAggregateThatSDCSingleStructure((ColGroupSDCSingle) that, ret, preModify);
		else if(that instanceof ColGroupSDCSingleZeros)
			return preAggregateThatSDCSingleZerosStructure((ColGroupSDCSingleZeros) that, ret);
		else if(that instanceof ColGroupSDCZeros)
			return preAggregateThatSDCZerosStructure((ColGroupSDCZeros) that, ret);
		else if(that instanceof ColGroupConst)
			return preAggregateThatConstStructure((ColGroupConst) that, ret);

		throw new NotImplementedException("Not supported pre aggregate using index structure of :"
			+ that.getClass().getSimpleName() + " in " + this.getClass().getSimpleName());
	}

	public abstract Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret);

	public abstract Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified);

	public abstract Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret);

	public abstract Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret);

	public abstract Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret,
		boolean preModified);

	public Dictionary preAggregateThatConstStructure(ColGroupConst that, Dictionary ret) {
		computeColSums(ret.getValues(), false);
		return ret;
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		if(lhs instanceof ColGroupEmpty)
			return;
		else if(lhs instanceof ColGroupValue)
			leftMultByColGroupValue((ColGroupValue) lhs, result);
		else if(lhs instanceof ColGroupUncompressed) {
			LOG.warn("Inefficient transpose of uncompressed to fit to "
				+ "template need t(UnCompressedColGroup) %*% AColGroup support");
			MatrixBlock ucCG = ((ColGroupUncompressed) lhs).getData();
			MatrixBlock tmp = new MatrixBlock(ucCG.getNumColumns(), ucCG.getNumRows(), ucCG.isInSparseFormat());
			LibMatrixReorg.transpose(ucCG, tmp, InfrastructureAnalyzer.getLocalParallelism());
			leftMultByMatrix(tmp, result);

		}
		else
			throw new DMLCompressionException(
				"Not supported left multiplication with A ColGroup of type: " + lhs.getClass().getSimpleName());
	}

	private void leftMultByColGroupValue(ColGroupValue lhs, MatrixBlock result) {
		final int nvL = lhs.getNumValues();
		final int nvR = this.getNumValues();
		final double[] lhValues = lhs.getValues();
		final double[] rhValues = this.getValues();
		final int lCol = lhs._colIndexes.length;
		final int rCol = this._colIndexes.length;
		final double[] resV = result.getDenseBlockValues();
		final int numCols = result.getNumColumns();

		final double threshold = 0.2;

		if(sameIndexStructure(lhs)) {
			int[] agI = getCounts();
			for(int i = 0; i < agI.length; i++) {
				if(i < nvL)
					for(int l = 0; l < lCol; l++) {
						final int leftOff = lhs._colIndexes[l] * numCols;
						final double lhV = lhValues[i * lCol + l] * agI[i];
						if(lhV != 0)
							for(int r = 0; r < rCol; r++) {
								final double rhV = rhValues[i * rCol + r];
								final double va = lhV * rhV;
								resV[leftOff + this._colIndexes[r]] += va;
							}
					}
			}
		}
		else if(lhs instanceof ColGroupConst || this instanceof ColGroupConst) {
			double[] r = this instanceof ColGroupConst ? rhValues : this._dict.colSum(getCounts(), rCol);
			double[] l = lhs instanceof ColGroupConst ? lhValues : lhs._dict.colSum(lhs.getCounts(), lCol);
			vectorVectorMultiply(l, lhs._colIndexes, r, this._colIndexes, resV, numCols);
		}
		else {
			int[] countsRight = getCounts();
			int mostFrequentRight = Math.max(countsRight[0], countsRight[countsRight.length - 1]);
			double percentageRight = (double) mostFrequentRight / this._numRows;
			double skipRight = percentageRight * rCol;
			int[] countsLeft = lhs.getCounts();
			int mostFrequentLeft = Math.max(countsLeft[0], countsLeft[countsLeft.length - 1]);
			double percentageLeft = (double) mostFrequentLeft / this._numRows;
			double skipLeft = percentageLeft * lCol;

			if(skipRight > threshold && percentageRight > percentageLeft && !(this instanceof ColGroupDDC)) {
				double[] mct = this._dict.getMostCommonTuple(this.getCounts(), rCol);
				double[] lhsSum = lhs._dict.colSum(lhs.getCounts(), lCol);
				if(mct != null)
					vectorVectorMultiply(lhsSum, lhs._colIndexes, mct, this._colIndexes, resV, numCols);

				ColGroupValue thisM = (mct != null) ? (ColGroupValue) this
					.copyAndSet(this._dict.subtractTuple(mct)) : this;
				Dictionary preAgg = lhs.preAggregateThatIndexStructure(thisM, true);
				matrixMultDictionariesAndOutputToColIndexes(lhValues, preAgg.getValues(), lhs._colIndexes,
					this._colIndexes, resV, numCols);
			}
			else if(skipLeft > threshold && !(lhs instanceof ColGroupDDC)) {
				double[] mct = lhs._dict.getMostCommonTuple(lhs.getCounts(), lCol);
				double[] thisColSum = this._dict.colSum(getCounts(), rCol);
				if(mct != null)
					vectorVectorMultiply(mct, lhs._colIndexes, thisColSum, this._colIndexes, resV, numCols);

				ColGroupValue lhsM = (mct != null) ? (ColGroupValue) lhs.copyAndSet(lhs._dict.subtractTuple(mct)) : lhs;
				Dictionary preAgg = this.preAggregateThatIndexStructure(lhsM, true);
				matrixMultDictionariesAndOutputToColIndexes(preAgg.getValues(), rhValues, lhs._colIndexes,
					this._colIndexes, resV, numCols);
			}
			else if(nvR * rCol < nvL * lCol) {
				Dictionary preAgg = lhs.preAggregateThatIndexStructure(this, false);
				matrixMultDictionariesAndOutputToColIndexes(lhValues, preAgg.getValues(), lhs._colIndexes,
					this._colIndexes, resV, numCols);
			}
			else {
				Dictionary preAgg = this.preAggregateThatIndexStructure(lhs, false);
				matrixMultDictionariesAndOutputToColIndexes(preAgg.getValues(), rhValues, lhs._colIndexes,
					this._colIndexes, resV, numCols);
			}
		}
	}

	@Override
	public void tsmm(double[] result, int numColumns) {
		int[] counts = getCounts();
		double[] values = getValues();
		int[] columns = getColIndices();
		if(values == null)
			return;
		for(int i = 0; i < columns.length; i++) {
			final int y = columns[i] * numColumns;
			for(int j = i; j < columns.length; j++) {
				final int x = columns[j];
				for(int h = 0; h < values.length / columns.length; h++) {
					double a = values[h * columns.length + i];
					double b = values[h * columns.length + j];
					result[x + y] += a * b * counts[h];
				}
			}
		}
	}

	@Override
	public boolean containsValue(double pattern) {
		return _dict.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros() {
		int[] counts = getCounts();
		return _dict.getNumberNonZeros(counts, _colIndexes.length);
	}

	private static void vectorVectorMultiply(final double[] left, final int[] leftRows, final double[] right,
		final int[] rightColumns, final double[] result, final int outCols) {
		if(left.length != leftRows.length) {
			// LOG.error(Arrays.toString(left));
			// LOG.error(Arrays.toString(right));
			// LOG.error(Arrays.toString(leftRows));
			// LOG.error(Arrays.toString(rightColumns));
			throw new DMLCompressionException(
				"Error left length " + left.length + " not equal columns length" + leftRows.length);
		}
		if(right.length != rightColumns.length)
			throw new DMLCompressionException(
				"Error right not equal length " + right.length + "  " + rightColumns.length);
		for(int row = 0; row < leftRows.length; row++) {
			final int outputRowOffset = leftRows[row] * outCols;
			final double vLeft = left[row];
			for(int col = 0; col < rightColumns.length; col++)
				result[outputRowOffset + rightColumns[col]] += vLeft * right[col];
		}
	}

	private static boolean logMM = true;

	/**
	 * Matrix Multiply the two matrices, note that the left side is transposed,
	 * 
	 * making the multiplication a: t(left) %*% right
	 * 
	 * @param left      The left side matrix, transposed linearized row major
	 * @param right     The right hand side linearized row major
	 * @param rowsLeft  The number of rows and the row indexes on the left hand side
	 * @param colsRight The number of columns and the column indexes on the right hand side
	 * @param result    The result matrix to put the results into, linearized row major
	 * @param outCols   The output columns count, to know how much to offset into with results.
	 */
	private static void matrixMultDictionariesAndOutputToColIndexes(double[] left, double[] right, int[] rowsLeft,
		int[] colsRight, double[] result, int outCols) {

		try {
			final int rows = left.length / rowsLeft.length;
			if(rows != right.length / colsRight.length)
				throw new DMLCompressionException(
					"Not equal number of rows: " + rows + " " + right.length / colsRight.length);
			for(int k = 0; k < rows; k++) {
				final int offL = k * rowsLeft.length;
				final int offR = k * colsRight.length;
				// final int offL = k * colsRight.length;
				// final int offR = k * rowsLeft.length;
				// if(offR < right.length && offL < left.length)
				for(int i = 0; i < rowsLeft.length; i++) {
					final int offOut = rowsLeft[i] * outCols;
					final double vl = left[offL + i];
					if(vl != 0)
						for(int j = 0; j < colsRight.length; j++) {
							final double vr = right[offR + j];
							result[offOut + colsRight[j]] += vl * vr;
						}
				}
			}
		}
		catch(Exception e) {

			if(logMM) {
				StringBuilder sb = new StringBuilder();
				sb.append("\nLeft (transposed):\n");
				for(int i = 0; i < rowsLeft.length; i++) {
					for(int j = i * rowsLeft.length; j < (i + 1) * rowsLeft.length; j++)
						sb.append(left[j] + ", ");
					sb.append("\n");
				}
				LOG.error(sb);

				sb = new StringBuilder();
				sb.append("\nRight:\n");
				for(int i = 0; i < colsRight.length; i++) {
					for(int j = i * colsRight.length; j < (i + 1) * colsRight.length; j++)
						sb.append(right[j] + ", ");
					sb.append("\n");
				}
				LOG.error(sb);
				logMM = false;
			}
			throw new DMLCompressionException("MM of pre aggregated colGroups failed", e);
		}
	}

	@Override
	public boolean isDense() {
		return !_zeros;
	}

	/**
	 * Multiply with a matrix on the left.
	 * 
	 * @param matrix matrix to left multiply
	 * @param result matrix block result
	 * @param values The materialized values contained in the ColGroupValue
	 * @param rl     The row to start the matrix multiplication from
	 * @param ru     The row to stop the matrix multiplication at.
	 */
	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, double[] values, int rl, int ru) {
		final int numVals = getNumValues();
		if(!(_dict instanceof MatrixBlockDictionary))
			_dict = _dict.getAsMatrixBlockDictionary(_colIndexes.length);

		MatrixBlock dictM = ((MatrixBlockDictionary) _dict).getMatrixBlock();
		dictM.examSparsity();
		MatrixBlock tmpRes = new MatrixBlock(1, _colIndexes.length, false);
		for(int i = rl; i < ru; i++) {
			double[] vals = matrix.isInSparseFormat() ? preAggregateSparseWithCheck(matrix.getSparseBlock(),
				i) : preAggregate(matrix.getDenseBlockValues(), i);
			if(vals != null) {
				DenseBlock preAggV = new DenseBlockFP64(new int[] {1, numVals}, vals);
				MatrixBlock preAgg = new MatrixBlock(1, numVals, preAggV);
				preAgg.setNonZeros(numVals);
				// LOG.error("PreAgg Sparsity " + preAgg.getSparsity() + " nnz " + preAgg.getNonZeros());
				LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
				addToResult(tmpRes, result, i);
				tmpRes.reset();
			}
		}
	}

	private void addToResult(MatrixBlock tmp, MatrixBlock result, int row) {
		if(tmp.isEmpty())
			return;
		else if(tmp.isInSparseFormat()) {
			throw new NotImplementedException();
		}
		else {
			final double[] tmpV = tmp.getDenseBlockValues();
			final double[] retV = result.getDenseBlockValues();
			final int nColRet = result.getNumColumns();
			// final int nColTmp = tmp.getNumColumns();
			final int offR = row * nColRet;
			// for(int row = rl, offT = 0, offR = rl * nColRet; row < ru; row++, offT += nColTmp, offR += nColRet) {
			for(int col = 0; col < _colIndexes.length; col++) {
				final int colOffset = _colIndexes[col];
				retV[offR + colOffset] += tmpV[col];
			}
			// }
		}

	}

	@Override
	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		leftMultByMatrix(matrix, result, getValues(), rl, ru);
	}

	public AColGroup rightMultByMatrix(MatrixBlock right) {
		Pair<int[], double[]> pre = preaggForRightMultiplyValues(getNumValues(), right, 0, right.getNumColumns(),
			right.getNumColumns());
		if(pre.getLeft().length > 0)
			return copyAndSet(pre.getLeft(), pre.getRight());
		return null;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += 8; // Dictionary Reference.
		size += 8; // Counts reference
		size += 1; // _zeros boolean reference
		size += 1; // _lossy boolean reference
		size += 2; // padding
		size += _dict.getInMemorySize();
		return size;
	}
}
