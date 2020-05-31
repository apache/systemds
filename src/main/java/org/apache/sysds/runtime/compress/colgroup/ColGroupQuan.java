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
import java.util.DoubleSummaryStatistics;
import java.util.Iterator;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.UncompressedBitmap;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class ColGroupQuan extends ColGroup {

	private static final long serialVersionUID = -9157476271360522008L;

	protected double _scale;
	protected byte[] _values;

	protected ColGroupQuan() {
		super();
	}

	protected ColGroupQuan(int[] colIndexes, int numRows, UncompressedBitmap ubm) {
		super(colIndexes, numRows);
		_values = new byte[ubm.getNumColumns() * numRows];

		double[] valuesFullPrecision = ubm.getValues();
		DoubleSummaryStatistics stat = Arrays.stream(valuesFullPrecision).summaryStatistics();
		double max = Math.abs(Math.max(stat.getMax(), Math.abs(stat.getMin())));
		if(Double.isInfinite(max)){
			throw new DMLCompressionException("Invalid ColGroupQuan, can't quantize Infinite value.");
		} else if (max == 0){
			_scale = 1;
			LOG.error("ColGroup! column with only 0 values good excuse to make new ColGroup");
		} else{
			_scale = max / (double) (Byte.MAX_VALUE);
		}
		for (int i = 0; i < valuesFullPrecision.length; i++) {
			int[] runs = ubm.getOffsetsList(i).extractValues();
			double curV = valuesFullPrecision[i];
			double scaledVal = curV / _scale;
			if(Double.isNaN(scaledVal) || Double.isInfinite(scaledVal)){
				throw new DMLRuntimeException("Something went wrong in scaling values");
			}
			byte scaledValQuan = (byte) (scaledVal);
			for (int j = 0; j < ubm.getOffsetsList(i).size(); j++) {
				_values[runs[j]] = scaledValQuan;
			}
		}
	}

	@Override
	public boolean getIfCountsType(){
		return false;
	}

	private ColGroupQuan(int[] colIndexes, double scale, byte[] values) {
		super(colIndexes, values.length / colIndexes.length);
		this._scale = scale;
		this._values = values;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.QUAN;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.QUAN8S;
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		if (_values == null || _values.length == 0) {
			return;
		}
		for (int row = rl; row < ru; row++) {
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				byte qVal = _values[row * colIx + row];
				double val = qVal * _scale;
				target.quickSetValue(row, col, val);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		if (_values == null || _values.length == 0) {
			return;
		}
		for (int row = 0; row < _numRows; row++) {
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double val = _values[row * colIx + row] * _scale;
				target.quickSetValue(row, col, val);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		if (_values == null || _values.length == 0)
			return;

		/**
		 * target.getDenseBlockValues() because this decompress is used for
		 * TransposeSelfMatrixMult meaning that the result is allocated directly into
		 * the result row or col matrix with the same code !
		 */
		// double[] c = target.getDenseBlockValues();

		// for (int row = 0; row < _numRows; row++) {
		// c[row] = (double)_values[row * colpos + row] * _scale;
		// }
		// target.setNonZeros(_numRows);

		double[] c = target.getDenseBlockValues();
		int nnz = 0;

		for (int row = 0; row < _numRows; row++) {
			double val = _values[row * colpos + row];
			if (val != 0) {
				nnz++;
			}
			c[row] = val * _scale;
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void write(DataOutput out) throws IOException {

		out.writeInt(_numRows);
		out.writeInt(_colIndexes.length);

		for (int i = 0; i < _colIndexes.length; i++)
			out.writeInt(_colIndexes[i]);

		for (int i = 0; i < _values.length; i++)
			out.writeByte(_values[i]);

		out.writeDouble(_scale);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_numRows = in.readInt();
		int numCols = in.readInt();

		_colIndexes = new int[numCols];
		for (int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] = in.readInt();

		_values = new byte[_numRows * numCols];
		for (int i = 0; i < _values.length; i++)
			_values[i] = in.readByte();

		_scale = in.readDouble();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = 8; // header
		ret += 4 * _colIndexes.length;
		ret += _values.length;
		return ret;
	}

	@Override
	public double get(int r, int c) {
		int colIx = Arrays.binarySearch(_colIndexes, c);
		return _values[r * colIx + r] * _scale;
	}

	@Override
	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru) {
		double[] b = ColGroupConverter.getDenseVector(vector);
		double[] c = result.getDenseBlockValues();

		// prepare reduced rhs w/ relevant values
		double[] sb = new double[_colIndexes.length];
		for (int j = 0; j < _colIndexes.length; j++) {
			sb[j] = b[_colIndexes[j]];
		}

		for (int row = rl; row < ru; row++) {
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				c[row] += (_values[row * colIx + row] * _scale) * sb[colIx];
			}
		}
	}

	@Override
	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result) {
		double[] a = ColGroupConverter.getDenseVector(vector);
		double[] c = result.getDenseBlockValues();

		for (int row = 0; row < _numRows; row++) {
			double val = _values[row] * _scale;
			for (int col = 0; col < _colIndexes.length; col++) {
				double value = val * a[row * col + row];
				c[_colIndexes[col]] += value;
			}
		}

	}

	@Override
	public void leftMultByRowVector(ColGroupDDC vector, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		if (op.fn instanceof Multiply) {
			return new ColGroupQuan(_colIndexes, op.executeScalar(_scale), _values);
		}
		double[] temp = new double[_values.length];
		double max = op.executeScalar((double)_values[0] * _scale);
		temp[0] = max;
		for (int i = 1; i < _values.length; i++) {
			temp[i] = op.executeScalar((double)_values[i] * _scale);
			double absTemp = Math.abs(temp[i]);
			if (absTemp > max) {
				max = absTemp;
			}
		}
		byte[] newValues = new byte[_values.length];
		double newScale = max / (double) (Byte.MAX_VALUE);
		for (int i = 0; i < _values.length; i++) {
			newValues[i] = (byte) ((double)temp[i] / newScale);
		}

		return new ColGroupQuan(_colIndexes, newScale, newValues);
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) {
		unaryAggregateOperations(op, result, 0, getNumRows());
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru) {

		if (op.aggOp.increOp.fn instanceof KahanPlus) {

			// Not using KahnObject because we already lost some of that precision anyway in
			// quantization.
			if (op.indexFn instanceof ReduceAll)
				computeSum(result);
			else if (op.indexFn instanceof ReduceCol)
				computeRowSums(result, rl, ru);
			else if (op.indexFn instanceof ReduceRow)
				computeColSums(result);
		} else if (op.aggOp.increOp.fn instanceof KahanPlusSq) {
			if (op.indexFn instanceof ReduceAll)
				computeSumSq(result);
			else if (op.indexFn instanceof ReduceCol)
				computeRowSumsSq(result, rl, ru);
			else if (op.indexFn instanceof ReduceRow)
				computeColSumsSq(result);
		} else if (op.aggOp.increOp.fn instanceof Builtin
				&& (((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAX
						|| ((Builtin) op.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MIN)) {
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;
			// min and max (reduceall/reducerow over tuples only)

			if (op.indexFn instanceof ReduceAll)
				computeMxx(result, builtin, _zeros);
			else if (op.indexFn instanceof ReduceCol)
				computeRowMxx(result, builtin, rl, ru);
			else if (op.indexFn instanceof ReduceRow)
				computeColMxx(result, builtin, _zeros);
		} else {
			throw new DMLScriptException("Unknown UnaryAggregate operator on CompressedMatrixBlock");
		}
	}

	protected void computeSum(MatrixBlock result) {
		// TODO Potential speedup use vector instructions/group in batches of 32
		long sum = 0L;
		for (int i = 0; i < _values.length; i++) {
			sum += (long) _values[i];
		}
		result.quickSetValue(0, 0, result.getValue(0, 0) + (double) sum * _scale);
	}

	protected void computeSumSq(MatrixBlock result) {

		double sumsq = 0;
		for (int i = 0; i < _values.length; i++) {
			double v =  _values[i] * _scale;
			sumsq += v*v;
		}
		result.quickSetValue(0, 0, result.getValue(0, 0) + sumsq);
	}

	protected void computeRowSums(MatrixBlock result, int rl, int ru) {
		if (_colIndexes.length < 256) {
			short[] rowSums = new short[ru - rl];
			for (int row = rl; row < ru; row++) {
				for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
					rowSums[row - rl] += _values[row * colIx + row];
				}
			}
			for (int row = rl; row < ru; row++) {
				result.quickSetValue(row, 0, result.getValue(row, 0) + (double) rowSums[row - rl] * _scale);
			}
		} else {
			throw new NotImplementedException("Not Implemented number of columns in ColGroupQuan row sum");
		}
	}

	protected void computeRowSumsSq(MatrixBlock result, int rl, int ru) {
		if (_colIndexes.length < 256) {
			float[] rowSumSq = new float[ru - rl];
			for (int row = rl; row < ru; row++) {
				for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
					double v = (double) _values[row * colIx + row] * _scale;
					rowSumSq[row - rl] += v*v;
				}
			}

			for (int row = rl; row < ru; row++) {
				result.quickSetValue(row, 0, result.getValue(row, 0) + rowSumSq[row - rl]);
			}

		} else {
			throw new NotImplementedException("Not Implemented number of columns in ColGroupQuan row sum");
		}
	}

	protected void computeColSums(MatrixBlock result) {
		if (_numRows < 256) {
			short[] colSums = new short[_colIndexes.length];
			for (int i = 0; i < _values.length; i++) {
				colSums[i / _numRows] += _values[i];
			}

			for (int col = 0; col < _colIndexes.length; col++) {
				result.quickSetValue(0, _colIndexes[col], colSums[col] * _scale);
			}
		} else if (_numRows < 16777216) { // (Int max + 1) / (short max + 1)
			int[] colSums = new int[_colIndexes.length];
			for (int i = 0; i < _values.length; i++) {
				colSums[i / _numRows] += _values[i];
			}

			for (int col = 0; col < _colIndexes.length; col++) {
				result.quickSetValue(0, _colIndexes[col], colSums[col] * _scale);
			}
		} else {
			double[] colSums = new double[_colIndexes.length];
			for (int i = 0; i < _values.length; i++) {
				colSums[i / _numRows] += _values[i];
			}

			for (int col = 0; col < _colIndexes.length; col++) {
				result.quickSetValue(0, _colIndexes[col], colSums[col] * _scale);
			}
		}
	}

	protected void computeColSumsSq(MatrixBlock result) {
	
		double[] sumsq = new double[_colIndexes.length];
		for (int i = 0; i < _values.length; i++) {
			double v =  _values[i] * _scale;
			sumsq[i / _numRows] += v*v;
		}
		
		for (int col = 0; col < _colIndexes.length; col++) {
			result.quickSetValue(0, _colIndexes[col], sumsq[col]);
		}
		
	}

	protected void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru) {
		double[] c = result.getDenseBlockValues();
		for (int row = rl; row < ru; row++) {
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				double v = ((double)_values[row * colIx + row]) * _scale;
				// System.out.println(v);
				c[row] = builtin.execute(c[row], v);
			}
		}
		
	}

	protected void computeMxx(MatrixBlock result, Builtin builtin, boolean zeros) {

		double res = 0;
		for (int i = 0; i < _values.length; i++) {
			res = builtin.execute(res, _values[i] * _scale);
		}
		result.quickSetValue(0, 0, res);
	}

	protected void computeColMxx(MatrixBlock result, Builtin builtin, boolean zeros) {
		double[] colRes = new double[_colIndexes.length];
		for (int i = 0; i < _values.length; i++) {
			colRes[i / _numRows] = builtin.execute(colRes[i / _numRows], _values[i] * _scale);
		}

		for (int col = 0; col < _colIndexes.length; col++) {
			result.quickSetValue(0, _colIndexes[col], colRes[col]);
		}
	}

	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		return new QuanValueIterator();
	}

	private class QuanValueIterator implements Iterator<IJV> {

		@Override
		public boolean hasNext() {
			throw new NotImplementedException("Not Implemented");
		}

		@Override
		public IJV next() {
			throw new NotImplementedException("Not Implemented");
		}

	}

	@Override
	public ColGroupRowIterator getRowIterator(int rl, int ru) {

		return new QuanRowIterator();
	}

	private class QuanRowIterator extends ColGroupRowIterator {

		@Override
		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			throw new NotImplementedException("Not Implemented");
		}

	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		// TODO Auto-generated method stub
		for (int row = rl; row < ru; row++) {
			int lnnz = 0;
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				lnnz += (_values[row * colIx + row] != 0) ? 1 : 0;
			}
			rnnz[row - rl] += lnnz;
		}
	}

	@Override
	public MatrixBlock getValuesAsBlock() {
		// TODO Auto-generated method stub
		MatrixBlock target = new MatrixBlock(_numRows, _colIndexes.length, 0.0);
		decompressToBlock(target, _colIndexes);
		return target;
	}

	@Override
	public int[] getCounts() {
		throw new DMLCompressionException(
				"Invalid function call, the counts in Uncompressed Col Group is always 1 for each value");
	}

	@Override
	public int[] getCounts(boolean includeZero) {
		throw new DMLCompressionException(
				"Invalid function call, the counts in Uncompressed Col Group is always 1 for each value");
	}

}