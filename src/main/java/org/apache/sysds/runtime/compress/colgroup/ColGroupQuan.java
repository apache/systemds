// /*
//  * Licensed to the Apache Software Foundation (ASF) under one
//  * or more contributor license agreements.  See the NOTICE file
//  * distributed with this work for additional information
//  * regarding copyright ownership.  The ASF licenses this file
//  * to you under the Apache License, Version 2.0 (the
//  * "License"); you may not use this file except in compliance
//  * with the License.  You may obtain a copy of the License at
//  *
//  *   http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing,
//  * software distributed under the License is distributed on an
//  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//  * KIND, either express or implied.  See the License for the
//  * specific language governing permissions and limitations
//  * under the License.
//  */

// package org.apache.sysds.runtime.compress.colgroup;

// import java.io.DataInput;
// import java.io.DataOutput;
// import java.io.IOException;
// import java.util.Arrays;
// import java.util.Iterator;

// import org.apache.commons.lang.NotImplementedException;
// import org.apache.sysds.runtime.DMLCompressionException;
// import org.apache.sysds.runtime.DMLScriptException;
// import org.apache.sysds.runtime.compress.utils.AbstractBitmap;
// import org.apache.sysds.runtime.compress.utils.BitmapLossy;
// import org.apache.sysds.runtime.functionobjects.Builtin;
// import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
// import org.apache.sysds.runtime.functionobjects.KahanPlus;
// import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
// import org.apache.sysds.runtime.functionobjects.ReduceAll;
// import org.apache.sysds.runtime.functionobjects.ReduceCol;
// import org.apache.sysds.runtime.functionobjects.ReduceRow;
// import org.apache.sysds.runtime.matrix.data.IJV;
// import org.apache.sysds.runtime.matrix.data.MatrixBlock;
// import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
// import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

// public class ColGroupQuan extends ColGroup {

// 	private static final long serialVersionUID = -9157476271360522008L;

// 	protected QDictionary _values;

// 	protected ColGroupQuan() {
// 		super();
// 	}

// 	protected ColGroupQuan(int[] colIndexes, int numRows, AbstractBitmap ubm) {
// 		// throw new NotImplementedException();
// 		super(colIndexes, numRows);
// 		byte[] lossyValues = ((BitmapLossy)ubm).getValues();
// 		byte[] values = new byte[numRows * colIndexes.length];
// 		for(int i = 0; i < lossyValues.length; i++) {
// 			int[] runs = ubm.getOffsetsList(i).extractValues();
// 			byte curV = lossyValues[i];

// 			for(int j = 0; j < ubm.getOffsetsList(i).size(); j++) {
// 				values[runs[j]] = curV;
// 			}
// 		}

// 		_values = new QDictionary(values, ((BitmapLossy)ubm).getScale());
// 	}

// 	protected ColGroupQuan(int[] colIndexes, int numRows, QDictionary values) {
// 		super(colIndexes, numRows);
// 		_values = values;
// 	}

// 	@Override
// 	public boolean getIfCountsType() {
// 		return false;
// 	}

// 	private ColGroupQuan(int[] colIndexes, QDictionary values) {
// 		super(colIndexes, values.getValuesLength() / colIndexes.length);
// 		this._values = values;
// 	}

// 	@Override
// 	public CompressionType getCompType() {
// 		return CompressionType.QUAN;
// 	}

// 	@Override
// 	protected ColGroupType getColGroupType() {
// 		return ColGroupType.QUAN8S;
// 	}

// 	@Override
// 	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
// 		if(_values == null || _values.getValuesLength()   == 0) {
// 			return;
// 		}
// 		// TODO Fix Loop to not multiply
// 		for(int row = rl; row < ru; row++) {
// 			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
// 				int col = _colIndexes[colIx];
// 				target.quickSetValue(row, col, _values.getValue(row * colIx + row));
// 			}
// 		}
// 	}

// 	@Override
// 	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
// 		if(_values == null || _values.getValuesLength() == 0) {
// 			return;
// 		}
// 		for(int row = 0; row < _numRows; row++) {
// 			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
// 				int col = _colIndexes[colIx];
// 				target.quickSetValue(row, col, _values.getValue(row * colIx + row));
// 			}
// 		}
// 	}

// 	@Override
// 	public void decompressToBlock(MatrixBlock target, int colpos) {
// 		if(_values == null || _values.getValuesLength()  == 0)
// 			return;

// 		double[] c = target.getDenseBlockValues();
// 		int nnz = 0;
// 		// TODO Fix for multi col group
// 		for(int row = 0; row < _numRows; row++) {
// 			double val = _values.getValue(row);
// 			if(val != 0) {
// 				nnz++;
// 			}
// 			c[row] = val;
// 		}
// 		target.setNonZeros(nnz);
// 	}

// 	@Override
// 	public void write(DataOutput out) throws IOException {

// 		out.writeInt(_numRows);
// 		out.writeInt(_colIndexes.length);

// 		for(int i = 0; i < _colIndexes.length; i++)
// 			out.writeInt(_colIndexes[i]);

// 		for(int i = 0; i < _values.getValuesLength() ; i++)
// 			out.writeByte(_values.getValueByte(i));

// 		out.writeDouble(_values.getScale());
// 	}

// 	@Override
// 	public void readFields(DataInput in) throws IOException {
// 		_numRows = in.readInt();
// 		int numCols = in.readInt();

// 		_colIndexes = new int[numCols];
// 		for(int i = 0; i < _colIndexes.length; i++)
// 			_colIndexes[i] = in.readInt();

// 		byte[] values = new byte[_numRows * numCols];
// 		for(int i = 0; i < values.length; i++)
// 			values[i] = in.readByte();

// 		double scale = in.readDouble();

// 		_values = new QDictionary(values, scale);
// 	}

// 	@Override
// 	public long getExactSizeOnDisk() {
// 		long ret = 8; // header
// 		ret += 8; // Object header of QDictionary
// 		ret += 4 * _colIndexes.length;
// 		ret += _values.getValuesLength() ;
// 		ret += 8; // scale value
// 		return ret;
// 	}

// 	@Override
// 	public double get(int r, int c) {
// 		int colIx = Arrays.binarySearch(_colIndexes, c);
// 		return _values.getValue(r * colIx + r);
// 	}

// 	@Override
// 	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru) {

// 		double[] b = ColGroupConverter.getDenseVector(vector);
// 		double[] c = result.getDenseBlockValues();

// 		if(_colIndexes.length == 1) {
// 			double r = b[_colIndexes[0]] * _values.getScale();
// 			for(int row = rl; row < ru; row++) {
// 				c[row] += _values.getValueByte(row) * r;
// 			}
// 		}
// 		else {

// 			// prepare reduced rhs w/ relevant values
// 			double[] sb = new double[_colIndexes.length];
// 			for(int j = 0; j < _colIndexes.length; j++) {
// 				sb[j] = b[_colIndexes[j]];
// 			}

// 			int colIx = 0;
// 			for(int off = 0; off < _values.getValuesLength() ; off += _numRows) {
// 				double r = _values.getScale() * sb[colIx];
// 				for(int row = rl; row < ru; row++) {
// 					c[row] += _values.getValueByte(off + row) * r;
// 				}
// 				colIx++;
// 			}
// 		}
// 	}

// 	@Override
// 	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result) {
// 		double[] a = ColGroupConverter.getDenseVector(vector);
// 		double[] c = result.getDenseBlockValues();

// 		for(int row = 0; row < _numRows; row++) {
// 			double val = _values.getValue(row);
// 			for(int col = 0; col < _colIndexes.length; col++) {
// 				double value = val * a[row * col + row];
// 				c[_colIndexes[col]] += value;
// 			}
// 		}

// 	}

// 	@Override
// 	public void leftMultByRowVector(ColGroupDDC vector, MatrixBlock result) {
// 		throw new NotImplementedException();
// 	}

// 	@Override
// 	public ColGroup scalarOperation(ScalarOperator op) {
// 		QDictionary res = _values.apply(op);
// 		return new ColGroupQuan(_colIndexes, res);
// 	}

// 	@Override
// 	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) {
// 		unaryAggregateOperations(op, result, 0, getNumRows());
// 	}

// 	@Override
// 	public long estimateInMemorySize() {
// 		return ColGroupSizes.estimateInMemorySizeQuan(getNumRows(), getNumCols());
// 	}

// 	@Override
// 	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru) {

// 		if(op.aggOp.increOp.fn instanceof KahanPlus) {

// 			// Not using KahnObject because we already lost some of that precision anyway in
// 			// quantization.
// 			if(op.indexFn instanceof ReduceAll)
// 				computeSum(result);
// 			else if(op.indexFn instanceof ReduceCol)
// 				computeRowSums(result, rl, ru);
// 			else if(op.indexFn instanceof ReduceRow)
// 				computeColSums(result);
// 		}
// 		else if(op.aggOp.increOp.fn instanceof KahanPlusSq) {
// 			if(op.indexFn instanceof ReduceAll)
// 				computeSumSq(result);
// 			else if(op.indexFn instanceof ReduceCol)
// 				computeRowSumsSq(result, rl, ru);
// 			else if(op.indexFn instanceof ReduceRow)
// 				computeColSumsSq(result);
// 		}
// 		else if(op.aggOp.increOp.fn instanceof Builtin &&
// 			(((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX ||
// 				((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN)) {
// 			Builtin builtin = (Builtin) op.aggOp.increOp.fn;
// 			// min and max (reduceall/reducerow over tuples only)

// 			if(op.indexFn instanceof ReduceAll)
// 				computeMxx(result, builtin, _zeros);
// 			else if(op.indexFn instanceof ReduceCol)
// 				computeRowMxx(result, builtin, rl, ru);
// 			else if(op.indexFn instanceof ReduceRow)
// 				computeColMxx(result, builtin, _zeros);
// 		}
// 		else {
// 			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
// 		}
// 	}

// 	protected void computeSum(MatrixBlock result) {
// 		long sum = 0L;
// 		for(int i = 0; i < _values.length(); i++) {
// 			sum += _values.getValueByte(i);
// 		}
// 		result.quickSetValue(0, 0, result.getValue(0, 0) + (double) sum * _values.getScale());
// 	}

// 	protected void computeSumSq(MatrixBlock result) {

// 		double sumsq = 0;
// 		for(int i = 0; i < _values.length(); i++) {
// 			double v = _values.getValue(i);
// 			sumsq += v * v;
// 		}
// 		result.quickSetValue(0, 0, result.getValue(0, 0) + sumsq);
// 	}

// 	protected void computeRowSums(MatrixBlock result, int rl, int ru) {
// 		if(_colIndexes.length < 256) {
// 			short[] rowSums = new short[ru - rl];
// 			for(int row = rl; row < ru; row++) {
// 				for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
// 					rowSums[row - rl] += _values.getValueByte(row * colIx + row);
// 				}
// 			}
// 			for(int row = rl; row < ru; row++) {
// 				result.quickSetValue(row, 0, result.getValue(row, 0) + rowSums[row - rl] * _values.getScale());
// 			}
// 		}
// 		else {
// 			throw new NotImplementedException("Not Implemented number of columns in ColGroupQuan row sum");
// 		}
// 	}

// 	protected void computeRowSumsSq(MatrixBlock result, int rl, int ru) {
// 		// TODO FIX Loop Index calculation!
// 		if(_colIndexes.length < 256) {
// 			float[] rowSumSq = new float[ru - rl];
// 			for(int row = rl; row < ru; row++) {
// 				for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
// 					double v = _values.getValue(row * colIx + row);
// 					rowSumSq[row - rl] += v * v;
// 				}
// 			}

// 			for(int row = rl; row < ru; row++) {
// 				result.quickSetValue(row, 0, result.getValue(row, 0) + rowSumSq[row - rl]);
// 			}

// 		}
// 		else {
// 			throw new NotImplementedException("Not Implemented number of columns in ColGroupQuan row sum");
// 		}
// 	}

// 	protected void computeColSums(MatrixBlock result) {
// 		// TODO AVOID division
// 		if(_numRows < 256) {
// 			short[] colSums = new short[_colIndexes.length];
// 			for(int i = 0; i < _values.length(); i++) {
// 				colSums[i / _numRows] += _values.getValueByte(i);
// 			}

// 			for(int col = 0; col < _colIndexes.length; col++) {
// 				result.quickSetValue(0, _colIndexes[col], colSums[col] * _values.getScale());
// 			}
// 		}
// 		else if(_numRows < 16777216) { // (Int max + 1) / (short max + 1)
// 			int[] colSums = new int[_colIndexes.length];
// 			for(int i = 0; i < _values.length(); i++) {
// 				colSums[i / _numRows] += _values.getValueByte(i);
// 			}

// 			for(int col = 0; col < _colIndexes.length; col++) {
// 				result.quickSetValue(0, _colIndexes[col], colSums[col] * _values.getScale());
// 			}
// 		}
// 		else {
// 			double[] colSums = new double[_colIndexes.length];
// 			for(int i = 0; i < _values.length(); i++) {
// 				colSums[i / _numRows] += _values.getValueByte(i);
// 			}

// 			for(int col = 0; col < _colIndexes.length; col++) {
// 				result.quickSetValue(0, _colIndexes[col], colSums[col] * _values.getScale());
// 			}
// 		}
// 	}

// 	protected void computeColSumsSq(MatrixBlock result) {

// 		// TODO Avoid Division!
// 		double[] sumsq = new double[_colIndexes.length];
// 		for(int i = 0; i < _values.length(); i++) {
// 			double v = _values.getValue(i);
// 			sumsq[i / _numRows] += v * v;
// 		}

// 		for(int col = 0; col < _colIndexes.length; col++) {
// 			result.quickSetValue(0, _colIndexes[col], sumsq[col]);
// 		}

// 	}

// 	protected void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru) {
// 		double[] c = result.getDenseBlockValues();
// 		// TODO: Fix Loop!
// 		for(int row = rl; row < ru; row++) {
// 			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {

// 				double v = _values.getValue(row * colIx + row);
// 				// System.out.println(v);
// 				c[row] = builtin.execute(c[row], v);
// 			}
// 		}

// 	}

// 	protected void computeMxx(MatrixBlock result, Builtin builtin, boolean zeros) {

// 		double res = 0;
// 		for(int i = 0; i < _values.length(); i++) {
// 			res = builtin.execute(res, _values.getValue(i));
// 		}
// 		result.quickSetValue(0, 0, res);
// 	}

// 	protected void computeColMxx(MatrixBlock result, Builtin builtin, boolean zeros) {
// 		double[] colRes = new double[_colIndexes.length];
// 		// TODO FIX INDEX CALCULATION / loop
// 		for(int i = 0; i < _values.length(); i++) {
// 			colRes[i / _numRows] = builtin.execute(colRes[i / _numRows], _values.getValue(i));
// 		}

// 		for(int col = 0; col < _colIndexes.length; col++) {
// 			result.quickSetValue(0, _colIndexes[col], colRes[col]);
// 		}
// 	}

// 	@Override
// 	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
// 		return new QuanValueIterator();
// 	}

// 	private class QuanValueIterator implements Iterator<IJV> {

// 		@Override
// 		public boolean hasNext() {
// 			throw new NotImplementedException("Not Implemented");
// 		}

// 		@Override
// 		public IJV next() {
// 			throw new NotImplementedException("Not Implemented");
// 		}

// 	}

// 	@Override
// 	public ColGroupRowIterator getRowIterator(int rl, int ru) {

// 		return new QuanRowIterator();
// 	}

// 	private class QuanRowIterator extends ColGroupRowIterator {

// 		@Override
// 		public void next(double[] buff, int rowIx, int segIx, boolean last) {
// 			throw new NotImplementedException("Not Implemented");
// 		}

// 	}

// 	@Override
// 	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {

// 		for(int row = rl; row < ru; row++) {
// 			int lnnz = 0;
// 			for(int colIx = 0; colIx < _colIndexes.length; colIx++) {
// 				lnnz += (_values.getValue(row * colIx + row) != 0) ? 1 : 0;
// 			}
// 			rnnz[row - rl] += lnnz;
// 		}
// 	}

// 	@Override
// 	public MatrixBlock getValuesAsBlock() {
// 		MatrixBlock target = new MatrixBlock(_numRows, _colIndexes.length, 0.0);
// 		decompressToBlock(target, _colIndexes);
// 		return target;
// 	}

// 	@Override
// 	public int[] getCounts() {
// 		throw new DMLCompressionException(
// 			"Invalid function call, the counts in Uncompressed Col Group is always 1 for each value");
// 	}

// 	@Override
// 	public int[] getCounts(boolean includeZero) {
// 		throw new DMLCompressionException(
// 			"Invalid function call, the counts in Uncompressed Col Group is always 1 for each value");
// 	}

// 	@Override
// 	public double[] getValues() {
// 		return _values.getValues();
// 	}

// 	@Override
// 	public boolean isLossy() {
// 		return true;
// 	}

// }