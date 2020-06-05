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

import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.AbstractBitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
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
	protected IDictionary _dict;

	public ColGroupValue() {
		super();
	}

	/**
	 * Stores the headers for the individual bitmaps.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 * @param cs         The Compression settings used for compression
	 */
	public ColGroupValue(int[] colIndices, int numRows, AbstractBitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows);
		_lossy = false;
		_zeros = ubm.containsZero();

		// sort values by frequency, if requested
		if(cs.sortValuesByLength && numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			ubm.sortValuesByFrequency();
		}
		switch(ubm.getType()) {
			case Full:
				_dict = new Dictionary(((Bitmap) ubm).getValues());
				break;
			case Lossy:
				_dict = new QDictionary((BitmapLossy) ubm);
				_lossy = true;
				break;
		}
		// extract and store distinct values (bitmaps handled by subclasses)
		// _dict = new Dictionary(ubm.getValues());
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

	public long getDictionarySize() {
		// NOTE: this estimate needs to be consistent with the estimate above,
		// so for now we use the (incorrect) double array size, not the dictionary size
		return (_dict != null) ? MemoryEstimates.doubleArrayCost(_dict.getValues().length) : 0;
	}

	/**
	 * Obtain number of distinct sets of values associated with the bitmaps in this column group.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps in this column group
	 */
	public int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.length);
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

	public final int[] getCounts() {
		int[] tmp = new int[getNumValues()];
		tmp = getCounts(tmp);
		if(_zeros && this instanceof ColGroupOffset) {
			tmp = Arrays.copyOf(tmp, tmp.length + 1);
			int sum = Arrays.stream(tmp).sum();
			tmp[tmp.length - 1] = getNumRows() - sum;
		}
		return tmp;
	}

	public abstract int[] getCounts(int[] out);

	public final int[] getCounts(int rl, int ru) {
		int[] tmp = new int[getNumValues()];
		return getCounts(rl, ru, tmp);
	}

	public boolean getIfCountsType() {
		return true;
	}

	public abstract int[] getCounts(int rl, int ru, int[] out);

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
		return _dict.hasZeroTuple(_colIndexes.length);
	}

	// protected final double[] sumAllValues(KahanFunction kplus, KahanObject kbuff) {
	// return sumAllValues(kplus, kbuff, true);
	// }

	// protected final double[] sumAllValues(KahanFunction kplus, KahanObject kbuff, boolean allocNew) {
	// // quick path: sum
	// if(getNumCols() > 1 && _dict instanceof QDictionary && kplus instanceof KahanPlus){
	// return sumAllValuesQToDouble();
	// }
	// else if(getNumCols() == 1 && kplus instanceof KahanPlus)
	// return _dict.getValues(); // shallow copy of values

	// // pre-aggregate value tuple
	// final int numVals = getNumValues();
	// double[] ret = allocNew ? new double[numVals] : allocDVector(numVals, false);
	// for(int k = 0; k < numVals; k++)
	// ret[k] = sumValues(k, kplus, kbuff);

	// return ret;
	// }

	// /**
	// * Method for summing all value tuples in the dictionary.
	// *
	// * This method assumes two things
	// *
	// * 1. That you dont call it if the number of columns in this ColGroup is 1. (then use
	// ((QDictionary)_dict)._values)
	// * 2. That it is not used for anything else than KahnPlus.
	// * @return an short array of the sum of each row in the quantized array.
	// */
	// protected final short[] sumAllValuesQ(){
	// final byte[] values = ((QDictionary)_dict)._values;
	// short[] res = new short[getNumValues()];

	// for(int i = 0, off = 0; off< values.length; i++, off += _colIndexes.length){
	// for( int j = 0 ; j < _colIndexes.length; j++){
	// res[i] += values[off + j];
	// }
	// }
	// return res;
	// }

	// protected static final double[] sumAllValuesQToDouble(QDictionary dict, int nrCol){
	// final byte[] values = dict._values;
	// double[] res = new double[dict.getNumberOfValues()];

	// for(int i = 0, off = 0; off< values.length; i++, off += _colIndexes.length){
	// for( int j = 0 ; j < _colIndexes.length; j++){
	// res[i] += values[off + j];
	// }
	// res[i] = res[i] * dict._scale;
	// }
	// return res;
	// }

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
	 */
	protected void computeMxx(MatrixBlock result, Builtin builtin) {
		// init and 0-value handling
		double val = (builtin
			.getBuiltinCode() == BuiltinCode.MAX) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		if(_zeros)
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
	 */
	protected void computeColMxx(MatrixBlock result, Builtin builtin) {
		final int numCols = getNumCols();

		// init and 0-value handling
		double[] vals = new double[numCols];

		// TODO fix edge cases in colMax. Since currently we rely on looking at rows in dict to specify if we start with
		// zeros or not
		if(!_zeros && _dict.getValuesLength() / numCols == getNumRows()) {
			Arrays.fill(vals,
				(builtin.getBuiltinCode() == BuiltinCode.MAX) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY);
		}

		// iterate over all values only
		vals = _dict.aggregateCols(vals, builtin, _colIndexes);
		// copy results to output
		for(int j = 0; j < numCols; j++)
			result.quickSetValue(0, _colIndexes[j], vals[j]);
	}

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
		double[] values = _dict.getValues(); // allocate new array just once
		Dictionary tmp = new Dictionary(Arrays.copyOf(values, values.length + numCols));
		double[] ret = tmp.apply(op).getValues();

		// add new value to the end
		Arrays.fill(ret, values.length, values.length + numCols, newVal);
		return ret;
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) {
		unaryAggregateOperations(op, result, 0, getNumRows());
	}

	@Override
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
				computeMxx(result, builtin);
			else if(op.indexFn instanceof ReduceCol)
				computeRowMxx(result, builtin, rl, ru);
			else if(op.indexFn instanceof ReduceRow)
				computeColMxx(result, builtin);
		}
		else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
	}

	protected abstract void computeSum(MatrixBlock result, KahanFunction kplus);

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

		_dict = IDictionary.read(in, _lossy);

	}

	@Override
	public void write(DataOutput out) throws IOException {
		int numCols = getNumCols();
		out.writeInt(_numRows);
		out.writeInt(numCols);
		out.writeBoolean(_zeros);
		out.writeBoolean(_lossy);

		// write col indices
		for(int i = 0; i < _colIndexes.length; i++)
			out.writeInt(_colIndexes[i]);

		_dict.write(out);

	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = 0; // header
		ret += 4; // num rows int
		ret += 4; // num cols int
		ret += 1; // Zeros boolean
		ret += 1; // lossy boolean
		// col indices
		ret += 4 * _colIndexes.length;
		// distinct values (groups of values)
		ret += _dict.getExactSizeOnDisk();
		return ret;
	}

}
