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

import java.util.Arrays;

import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.UncompressedBitmap;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * Base class for column groups encoded with value dictionary.
 * 
 */
public abstract class ColGroupValue extends ColGroup {
	private static final long serialVersionUID = 3786247536054353658L;

	// thread-local pairs of reusable temporary vectors for positions and values
	private static ThreadLocal<Pair<int[], double[]>> memPool = new ThreadLocal<Pair<int[], double[]>>() {
		@Override
		protected Pair<int[], double[]> initialValue() {
			return new Pair<>();
		}
	};

	/** Distinct values associated with individual bitmaps. */
	protected Dictionary _dict;

	public ColGroupValue() {
		super();
	}

	/**
	 * Stores the headers for the individual bitmaps.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 */
	public ColGroupValue(int[] colIndices, int numRows, UncompressedBitmap ubm) {
		super(colIndices, numRows);

		// sort values by frequency, if requested
		if(CompressionSettings.SORT_VALUES_BY_LENGTH && numRows > BitmapEncoder.BITMAP_BLOCK_SZ) {
			ubm.sortValuesByFrequency();
		}

		// extract and store distinct values (bitmaps handled by subclasses)
		_dict = new Dictionary(ubm.getValues());
	}

	/**
	 * Constructor for subclass methods that need to create shallow copies
	 * 
	 * @param colIndices raw column index information
	 * @param numRows    number of rows in the block
	 * @param values     set of distinct values for the block (associated bitmaps are kept in the subclass)
	 */
	protected ColGroupValue(int[] colIndices, int numRows, double[] values) {
		super(colIndices, numRows);
		_dict = new Dictionary(values);
	}

	@Override
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeGroupValue(_colIndexes.length, getNumValues());
	}

	public long getDictionarySize() {
		//NOTE: this estimate needs to be consistent with the estimate above,
		//so for now we use the (incorrect) double array size, not the dictionary size
		return (_dict != null) ? MemoryEstimates.doubleArrayCost(_dict.getValues().length) : 0;
	}

	/**
	 * Obtain number of distinct sets of values associated with the bitmaps in this column group.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps in this column group
	 */
	public int getNumValues() {
		return _dict.getValues().length / _colIndexes.length;
	}

	public double[] getValues() {
		return _dict.getValues();
	}

	public void setValues(double[] values) {
		_dict = new Dictionary(values);
	}

	public double getValue(int k, int col) {
		return _dict.getValues()[k * getNumCols() + col];
	}
	
	public void setDictionary(Dictionary dict) {
		_dict = dict;
	}

	public MatrixBlock getValuesAsBlock() {
		boolean containsZeros = (this instanceof ColGroupOffset) ? ((ColGroupOffset) this)._zeros : false;
		final double[] values = getValues();
		int vlen = values.length;
		int rlen = containsZeros ? vlen + 1 : vlen;
		MatrixBlock ret = new MatrixBlock(rlen, 1, false);
		for(int i = 0; i < vlen; i++)
			ret.quickSetValue(i, 0, values[i]);
		return ret;
	}

	public final int[] getCounts() {
		int[] tmp = new int[getNumValues()];
		return getCounts(tmp);
	}

	public abstract int[] getCounts(int[] out);

	public final int[] getCounts(int rl, int ru) {
		int[] tmp = new int[getNumValues()];
		return getCounts(rl, ru, tmp);
	}

	public abstract int[] getCounts(int rl, int ru, int[] out);

	public int[] getCounts(boolean inclZeros) {
		int[] counts = getCounts();
		if(inclZeros && this instanceof ColGroupOffset) {
			counts = Arrays.copyOf(counts, counts.length + 1);
			int sum = 0;
			for(int i = 0; i < counts.length; i++)
				sum += counts[i];
			counts[counts.length - 1] = getNumRows() - sum;
		}
		return counts;
	}

	public MatrixBlock getCountsAsBlock() {
		return getCountsAsBlock(getCounts());
	}

	public static MatrixBlock getCountsAsBlock(int[] counts) {
		MatrixBlock ret = new MatrixBlock(counts.length, 1, false);
		for(int i = 0; i < counts.length; i++)
			ret.quickSetValue(i, 0, counts[i]);
		return ret;
	}

	protected int containsAllZeroValue() {
		return _dict.hasZeroTuple(getNumCols());
	}

	protected final double[] sumAllValues(KahanFunction kplus, KahanObject kbuff) {
		return sumAllValues(kplus, kbuff, true);
	}

	public final double sumValues(int valIx, KahanFunction kplus, KahanObject kbuff) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		final double[] values = _dict.getValues();
		kbuff.set(0, 0);
		for(int i = 0; i < numCols; i++)
			kplus.execute2(kbuff, values[valOff + i]);
		return kbuff._sum;
	}

	protected final double[] sumAllValues(KahanFunction kplus, KahanObject kbuff, boolean allocNew) {
		// quick path: sum
		if(getNumCols() == 1 && kplus instanceof KahanPlus)
			return _dict.getValues(); // shallow copy of values

		// pre-aggregate value tuple
		final int numVals = getNumValues();
		double[] ret = allocNew ? new double[numVals] : allocDVector(numVals, false);
		for(int k = 0; k < numVals; k++)
			ret[k] = sumValues(k, kplus, kbuff);

		return ret;
	}

	protected final double sumValues(int valIx, double[] b) {
		final int numCols = getNumCols();
		final int valOff = valIx * numCols;
		final double[] values = _dict.getValues();
		double val = 0;
		for(int i = 0; i < numCols; i++)
			val += values[valOff + i] * b[i];
		return val;
	}

	protected final double[] preaggValues(int numVals, double[] b) {
		return preaggValues(numVals, b, false);
	}

	protected final double[] preaggValues(int numVals, double[] b, boolean allocNew) {
		double[] ret = allocNew ? new double[numVals] : allocDVector(numVals, false);
		for(int k = 0; k < numVals; k++)
			ret[k] = sumValues(k, b);

		return ret;
	}

	/**
	 * NOTE: Shared across OLE/RLE/DDC because value-only computation.
	 * 
	 * @param result  output matrix block
	 * @param builtin function object
	 * @param zeros   indicator if column group contains zero values
	 */
	protected void computeMxx(MatrixBlock result, Builtin builtin, boolean zeros) {
		// init and 0-value handling
		double val = (builtin.getBuiltinCode() == BuiltinCode.MAX) ?
			Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		if(zeros)
			val = builtin.execute(val, 0);

		// iterate over all values only
		val = _dict.aggregate(val, builtin);
		
		// compute new partial aggregate
		val = builtin.execute(val, result.quickGetValue(0, 0));
		result.quickSetValue(0, 0, val);
	}

	/**
	 * NOTE: Shared across OLE/RLE/DDC because value-only computation.
	 * 
	 * @param result  output matrix block
	 * @param builtin function object
	 * @param zeros   indicator if column group contains zero values
	 */
	protected void computeColMxx(MatrixBlock result, Builtin builtin, boolean zeros) {
		final int numCols = getNumCols();

		// init and 0-value handling
		double[] vals = new double[numCols];
		Arrays.fill(vals, (builtin.getBuiltinCode() == BuiltinCode.MAX) ?
			Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
		if(zeros) {
			for(int j = 0; j < numCols; j++)
				vals[j] = builtin.execute(vals[j], 0);
		}

		// iterate over all values only
		vals = _dict.aggregateCols(vals, builtin, _colIndexes);
		
		// copy results to output
		for(int j = 0; j < numCols; j++)
			result.quickSetValue(0, _colIndexes[j], vals[j]);
	}

	// additional vector-matrix multiplication to avoid DDC uncompression
	public abstract void leftMultByRowVector(ColGroupDDC vector, MatrixBlock result);

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value metadata stored in the superclass.
	 * 
	 * @param op scalar operation to perform
	 * @return transformed copy of value metadata for this column group
	 */
	protected double[] applyScalarOp(ScalarOperator op) {
		return _dict.clone().apply(op).getValues();
	}

	protected double[] applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		double[] values = _dict.getValues(); //allocate new array just once
		Dictionary tmp = new Dictionary(Arrays.copyOf(values, values.length+numCols));
		double[] ret = tmp.apply(op).getValues();

		// add new value to the end
		Arrays.fill(ret, values.length, values.length+numCols, newVal);
		return ret;
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) {
		unaryAggregateOperations(op, result, 0, getNumRows());
	}

	/**
	 * 
	 * @param op     aggregation operator
	 * @param result output matrix block
	 * @param rl     row lower index, inclusive
	 * @param ru     row upper index, exclusive
	 */
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru) {
		// sum and sumsq (reduceall/reducerow over tuples and counts)
		if(op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq) {
			KahanFunction kplus = (op.aggOp.increOp.fn instanceof KahanPlus) ? KahanPlus
				.getKahanPlusFnObject() : KahanPlusSq.getKahanPlusSqFnObject();

			if(op.indexFn instanceof ReduceAll)
				computeSum(result, kplus);
			else if(op.indexFn instanceof ReduceCol)
				computeRowSums(result, kplus, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColSums(result, kplus);
		}
		// min and max (reduceall/reducerow over tuples only)
		else if(op.aggOp.increOp.fn instanceof Builtin &&
			(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX ||
				((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN)) {
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;

			if(op.indexFn instanceof ReduceAll)
				computeMxx(result, builtin, _zeros);
			else if(op.indexFn instanceof ReduceCol)
				computeRowMxx(result, builtin, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColMxx(result, builtin, _zeros);
		}
		else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
	}

	protected abstract void computeSum(MatrixBlock result, KahanFunction kplus );

	protected abstract void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru);

	protected abstract void computeColSums(MatrixBlock result, KahanFunction kplus);

	protected abstract void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru);

	// dynamic memory management

	public static void setupThreadLocalMemory(int len) {
		Pair<int[], double[]> p = new Pair<>();
		p.setKey(new int[len]);
		p.setValue(new double[len]);
		memPool.set(p);
	}

	public static void cleanupThreadLocalMemory() {
		memPool.remove();
	}

	protected static double[] allocDVector(int len, boolean reset) {
		Pair<int[], double[]> p = memPool.get();

		// sanity check for missing setup
		if(p.getValue() == null)
			return new double[len];

		// get and reset if necessary
		double[] tmp = p.getValue();
		if(reset)
			Arrays.fill(tmp, 0, len, 0);
		return tmp;
	}

	protected static int[] allocIVector(int len, boolean reset) {
		Pair<int[], double[]> p = memPool.get();

		// sanity check for missing setup
		if(p.getKey() == null)
			return new int[len];

		// get and reset if necessary
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
		sb.append(String.format("\n%15s%5d ", "Values:", _dict.getValues().length));
		sb.append(Arrays.toString(_dict.getValues()));
		return sb.toString();
	}
}
