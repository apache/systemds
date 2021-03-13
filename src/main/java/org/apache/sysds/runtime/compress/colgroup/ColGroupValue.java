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
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.pre.ArrPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.MapPreAggregate;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Base class for column groups encoded with value dictionary. This include column groups such as DDC OLE and RLE.
 * 
 */
public abstract class ColGroupValue extends AColGroup implements Cloneable {
	private static final long serialVersionUID = 3786247536054353658L;

	/** thread-local pairs of reusable temporary vectors for positions and values */
	private static ThreadLocal<Pair<int[], double[]>> memPool = new ThreadLocal<Pair<int[], double[]>>() {
		@Override
		protected Pair<int[], double[]> initialValue() {
			return null;
		}
	};

	/** Distinct value tuples associated with individual bitmaps. */
	protected ADictionary _dict;
	protected int[] counts;

	protected ColGroupValue() {
		super();
	}

	/**
	 * Main constructor for the ColGroupValues. Used to contain the dictionaries used for the different types of
	 * ColGroup.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 * @param cs         The Compression settings used for compression
	 */
	protected ColGroupValue(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows);

		_zeros = ubm.getNumOffsets() < (long) numRows;
		// sort values by frequency, if requested
		if(cs.sortValuesByLength && numRows > CompressionSettings.BITMAP_BLOCK_SZ)
			ubm.sortValuesByFrequency();

		switch(ubm.getType()) {
			case Full:
				_dict = new Dictionary(((Bitmap) ubm).getValues());
				break;
			case Lossy:
				_dict = new QDictionary((BitmapLossy) ubm).makeDoubleDictionary();
				// _lossy = true;
				break;
		}
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

	/**
	 * Obtain number of distinct sets of values associated with the bitmaps in this column group.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps in this column group
	 */
	public int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.length);
	}

	@Override
	public double[] getValues() {
		return _dict != null ? _dict.getValues() : null;
	}

	public ADictionary getDictionary() {
		return _dict;
	}

	public void addMinMax(double[] ret) {
		_dict.addMaxAndMin(ret, _colIndexes);
	}

	protected void setDictionary(ADictionary dict) {
		_dict = dict;
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		final double[] values = getValues();
		if(values != null) {

			int vlen = values.length;

			int rlen = _zeros ? vlen + 1 : vlen;
			MatrixBlock ret = new MatrixBlock(rlen, 1, false);
			for(int i = 0; i < vlen; i++)
				ret.quickSetValue(i, 0, values[i]);
			return ret;
		}
		return new MatrixBlock(0, 0, false);
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

	protected int containsAllZeroValue() {
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

	private Pair<int[], double[]> preaggValuesFromDense(final int numVals, final double[] b, double[] dictVals,
		final int cl, final int cu, final int cut) {

		int[] aggregateColumns = getAggregateColumnsSetDense(b, cl, cu, cut);
		double[] ret = new double[numVals * aggregateColumns.length];

		for(int k = 0, off = 0;
			k < numVals * _colIndexes.length;
			k += _colIndexes.length, off += aggregateColumns.length) {
			for(int h = 0; h < _colIndexes.length; h++) {
				int idb = _colIndexes[h] * cut;
				double v = dictVals[k + h];
				for(int i = 0; i < aggregateColumns.length; i++) {
					ret[off + i] += v * b[idb + aggregateColumns[i]];
				}

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

	private Pair<int[], double[]> preaggValuesFromSparse(int numVals, SparseBlock b, double[] dictVals, int cl, int cu,
		int cut) {

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
							ret[j + retIdx] += dictVals[offOrg] * sValues[i];
						}
				}
			}
		}
		return new ImmutablePair<>(aggregateColumns, ret);
	}

	public Pair<int[], double[]> preaggValues(int numVals, MatrixBlock b, double[] dictVals, int cl, int cu, int cut) {
		return b.isInSparseFormat() ? preaggValuesFromSparse(numVals,
			b.getSparseBlock(),
			dictVals,
			cl,
			cu,
			cut) : preaggValuesFromDense(numVals, b.getDenseBlockValues(), dictVals, cl, cu, cut);
	}

	protected static double[] sparsePreaggValues(int numVals, double v, boolean allocNew, double[] dictVals) {
		double[] ret = allocNew ? new double[numVals + 1] : allocDVector(numVals + 1, true);

		for(int k = 0; k < numVals; k++)
			ret[k] = dictVals[k] * v;
		return ret;
	}

	public double getMin() {
		return computeMxx(Double.POSITIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MIN));
	}

	public double getMax() {
		return computeMxx(Double.NEGATIVE_INFINITY, Builtin.getBuiltinFnObject(BuiltinCode.MAX));
	}

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

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] c) {
		unaryAggregateOperations(op, c, 0, _numRows);
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int rl, int ru) {
		// sum and sumsq (reduceall/reducerow over tuples and counts)
		if(op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq ||
			op.aggOp.increOp.fn instanceof Mean) {
			KahanFunction kplus = (op.aggOp.increOp.fn instanceof KahanPlus ||
				op.aggOp.increOp.fn instanceof Mean) ? KahanPlus
					.getKahanPlusFnObject() : KahanPlusSq.getKahanPlusSqFnObject();
			boolean mean = op.aggOp.increOp.fn instanceof Mean;
			if(op.indexFn instanceof ReduceAll)
				computeSum(c, kplus instanceof KahanPlusSq);
			else if(op.indexFn instanceof ReduceCol)
				computeRowSums(c, kplus instanceof KahanPlusSq, rl, ru, mean);
			else if(op.indexFn instanceof ReduceRow)
				computeColSums(c, kplus instanceof KahanPlusSq);
		}
		// min and max (reduceall/reducerow over tuples only)
		else if(op.aggOp.increOp.fn instanceof Builtin &&
			(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX ||
				((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN)) {
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;

			if(op.indexFn instanceof ReduceAll)
				c[0] = computeMxx(c[0], builtin);
			else if(op.indexFn instanceof ReduceCol)
				computeRowMxx(c, builtin, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColMxx(c, builtin);
		}
		else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
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
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Columns:", _colIndexes.length));
		sb.append(Arrays.toString(_colIndexes));
		if(_dict != null) {
			sb.append(String.format("\n%15s%5d ", "Values:", _dict.getValues().length));
			_dict.getString(sb, _colIndexes.length);
		}
		return sb.toString();
	}

	@Override
	public boolean isLossy() {
		return _lossy;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_numRows = in.readInt();
		int numCols = in.readInt();
		_zeros = in.readBoolean();
		_lossy = in.readBoolean();
		// read col indices
		_colIndexes = new int[numCols];
		for(int i = 0; i < numCols; i++)
			_colIndexes[i] = in.readInt();

		if(in.readBoolean())
			_dict = ADictionary.read(in, _lossy);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(_numRows);
		out.writeInt(getNumCols());
		out.writeBoolean(_zeros);
		out.writeBoolean(_lossy);

		// write col indices
		for(int i = 0; i < _colIndexes.length; i++)
			out.writeInt(_colIndexes[i]);

		if(_dict != null) {
			out.writeBoolean(true);
			_dict.write(out);
		}
		else
			out.writeBoolean(false);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = 0; // header
		ret += 4; // num rows int
		ret += 4; // num cols int
		ret += 1; // zeros boolean
		ret += 1; // lossy boolean
		// col indices
		ret += 4 * _colIndexes.length;
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
	public void leftMultByRowVector(double[] a, double[] c) {
		final double[] values = getValues();
		final int numVals = getNumValues();
		leftMultByRowVector(a, c, numVals, values);
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int offT) {
		final double[] values = getValues();
		final int numVals = getNumValues();
		leftMultByRowVector(a, c, numVals, values, offT);
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
		final double[] vals = preAggregate(a);
		postScaling(values, vals, c, numVals);
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values, int offT) {
		final double[] vals = preAggregate(a);
		postScaling(values, vals, c, numVals, 0, 0, offT);
	}

	@Override
	public AColGroup sliceColumns(int cl, int cu) {
		if(cu - cl == 1)
			return sliceSingleColumn(cl);
		else
			return sliceMultiColumns(cl, cu);
	}

	private AColGroup sliceSingleColumn(int col) {
		ColGroupValue ret = (ColGroupValue) copy();

		int idx = Arrays.binarySearch(_colIndexes, col);
		// Binary search returns negative value if the column is not found.
		// Therefore return null, if the column is not inside this colGroup.
		if(idx >= 0) {
			ret._colIndexes = new int[1];
			ret._colIndexes[0] = _colIndexes[idx] - col;
			if(ret._dict != null)
				if(_colIndexes.length == 1)
					ret._dict = ret._dict.clone();
				else
					ret._dict = ret._dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.length);

			return ret;
		}
		else
			return null;
	}

	private AColGroup sliceMultiColumns(int cl, int cu) {
		ColGroupValue ret = (ColGroupValue) copy();
		int idStart = 0;
		int idEnd = 0;
		for(int i = 0; i < _colIndexes.length; i++) {
			if(_colIndexes[i] < cl)
				idStart++;
			if(_colIndexes[i] < cu)
				idEnd++;
		}
		int numberOfOutputColumns = idEnd - idStart;
		if(numberOfOutputColumns > 0) {
			ret._dict = ret._dict != null ? ret._dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.length) : null;

			ret._colIndexes = new int[numberOfOutputColumns];
			// Incrementing idStart here so make sure that the dictionary is extracted
			// before
			for(int i = 0; i < numberOfOutputColumns; i++) {
				ret._colIndexes[i] = _colIndexes[idStart++] - cl;
			}
			return ret;
		}
		else
			return null;
	}

	/**
	 * Post scale for left Multiplication
	 * 
	 * @param dictValues The dictionary values materialized as double array.
	 * @param vals       The values aggregated from the left side row vector.
	 * @param c          The output matrix
	 * @param numVals    The number of values contained in the dictionary.
	 */
	protected void postScaling(double[] dictValues, double[] vals, double[] c, int numVals) {
		postScaling(dictValues, vals, c, numVals, 0, 0);
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
	protected void postScaling(double[] dictValues, double[] vals, double[] c, int numVals, int row, int totalCols) {
		final int ncol = getNumCols();
		int valOff = 0;

		for(int k = 0; k < numVals; k++) {
			double aval = vals[k];
			for(int j = 0; j < ncol; j++) {
				int colIx = _colIndexes[j] + row * totalCols;
				c[colIx] += aval * dictValues[valOff++];
			}
		}
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
	 * @param offT       The offset into C to assign the values from usefull in case the c output is a smaller temporary
	 *                   array.
	 */
	protected void postScaling(double[] dictValues, double[] vals, double[] c, int numVals, int row, int totalCols,
		int offT) {
		final int ncol = getNumCols();
		int valOff = 0;

		for(int k = 0; k < numVals; k++) {
			double aval = vals[k];
			for(int j = 0; j < ncol; j++) {
				int colIx = _colIndexes[j] + row * totalCols;
				c[offT + colIx] += aval * dictValues[valOff++];
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
		return preAggregateSparse(sb, 0);
	}

	/**
	 * Pre aggregate for left multiplication of sparse vector or matrix.
	 * 
	 * @param sb  The input sparse vector or matrix to aggregate
	 * @param row The row index to aggregate
	 * @return The pre-aggregated values.
	 */
	public abstract double[] preAggregateSparse(SparseBlock sb, int row);

	public abstract boolean sameIndexStructure(ColGroupValue that);

	public abstract int getIndexStructureHash();

	// try to make a memo table.
	// private static Map<Integer,Map<Integer,Memo>> memorizer = new HashMap<>();

	// private class Memo {
	// ColGroupValue lhs;
	// ColGroupValue rhs;
	// IPreAggregate agg;

	// private Memo(ColGroupValue lhs, ColGroupValue rhs, IPreAggregate agg) {
	// this.lhs = lhs;
	// this.rhs = rhs;
	// this.agg = agg;
	// }
	// }

	public IPreAggregate preAggregate(ColGroupValue lhs) {
		// int lhsH = lhs.getIndexStructureHash();
		// int rhsH = this.getIndexStructureHash();
		// if(memorizer.get(lhsH) == null){
		// memorizer.put(lhsH, new HashMap<>());
		// }
		// Memo m = memorizer.get(lhsH).get(rhsH);
		// if(m != null && lhs.sameIndexStructure(m.lhs) &&
		// this.sameIndexStructure(m.rhs))
		// return m.agg;

		IPreAggregate r = preCallAggregate(lhs);
		// Map<Integer,Memo> l = memorizer.get(lhsH);
		// l.put(rhsH, new Memo(lhs, this, r));
		return r;
	}

	public IPreAggregate preCallAggregate(ColGroupValue lhs) {
		// LOG.error(lhs.getClass().getSimpleName() + " in " +
		// this.getClass().getSimpleName() + " "
		// + Arrays.toString(lhs.getColIndices()) + " " +
		// Arrays.toString(this.getColIndices()));

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
		else if(lhs instanceof ColGroupEmpty)
			return null;

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
	 * Multiply with a matrix on the left.
	 * 
	 * @param lhs     Left hand side ColGroupValue
	 * @param result  Dense result matrix block
	 * @param numRows The number of rows in the left hand side matrix
	 * @param numCols The number of columns in the colGroups parent matrix.
	 */
	public void leftMultByAggregatedColGroup(ColGroupValue lhs, double[] result, final int numRows, final int numCols) {

		final int nvL = lhs.getNumValues();
		final int nvR = this.getNumValues();
		final double[] lhValues = lhs.getValues();
		final double[] rhValues = this.getValues();
		final int lCol = lhs._colIndexes.length;
		final int rCol = this._colIndexes.length;

		if(sameIndexStructure(lhs)) {
			int[] agI = getCounts();
			for(int a = 0, off = 0; a < nvL; a++, off += nvL + 1)
				leftMultDictEntry(agI[a], off, nvL, lCol, rCol, lhs, numCols, lhValues, rhValues, result);
		}
		else {
			// LOG.error(lCol +" "+ nvL);
			// LOG.error(rCol +" "+ nvR);
			IPreAggregate ag = preAggregate(lhs);
			if(ag == null)
				return;
			else if(ag instanceof MapPreAggregate)
				leftMultMapPreAggregate(nvL,
					lCol,
					rCol,
					lhs,
					numCols,
					lhValues,
					rhValues,
					result,
					(MapPreAggregate) ag);
			else
				leftMultArrayPreAggregate(nvL,
					nvR,
					lCol,
					rCol,
					lhs,
					numCols,
					lhValues,
					rhValues,
					result,
					((ArrPreAggregate) ag).getArr());
		}
	}

	private void leftMultMapPreAggregate(final int nvL, final int lCol, final int rCol, final ColGroupValue lhs,
		final int numCols, double[] lhValues, double[] rhValues, double[] c, MapPreAggregate agM) {
		final int[] map = agM.getMap();
		final int aggSize = agM.getSize();
		for(int k = 0; k < aggSize; k += 2)
			leftMultDictEntry(map[k + 1], map[k], nvL, lCol, rCol, lhs, numCols, lhValues, rhValues, c);
		leftMultDictEntry(agM.getMapFreeValue(), 0, nvL, lCol, rCol, lhs, numCols, lhValues, rhValues, c);
	}

	private void leftMultArrayPreAggregate(final int nvL, final int nvR, final int lCol, final int rCol,
		final ColGroupValue lhs, final int numCols, double[] lhValues, double[] rhValues, double[] c, int[] arr) {
		for(int a = 0; a < nvL * nvR; a++)
			leftMultDictEntry(arr[a], a, nvL, lCol, rCol, lhs, numCols, lhValues, rhValues, c);
	}

	private void leftMultDictEntry(final int m, final int a, final int nvL, final int lCol, final int rCol,
		final ColGroupValue lhs, final int numCols, double[] lhValues, double[] rhValues, double[] c) {

		if(m > 0) {
			final int lhsRowOffset = (a % nvL) * lCol;
			final int rhsRowOffset = (a / nvL) * rCol;

			for(int j = 0; j < lCol; j++) {
				final int resultOff = lhs._colIndexes[j] * numCols;
				final double lh = lhValues[lhsRowOffset + j] * m;
				if(lh != 0)
					for(int i = 0; i < rCol; i++) {
						double rh = rhValues[rhsRowOffset + i];
						c[resultOff + _colIndexes[i]] += lh * rh;
					}
			}
		}
	}

	@Override
	public void leftMultBySelfDiagonalColGroup(double[] result, int numColumns) {
		int[] counts = getCounts();
		double[] values = getValues();
		int[] columns = getColIndices();
		if(values == null)
			return;
		for(int i = 0; i < columns.length; i++) {
			final int y = columns[i] * numColumns;
			for(int j = 0; j < columns.length; j++) {
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
	public boolean containsValue(double pattern){
		return _dict.containsValue(pattern);
	}

	@Override
	public long getNumberNonZeros(){
		int[] counts = getCounts();
		return _dict.getNumberNonZeros(counts, _colIndexes.length);
	}
}
