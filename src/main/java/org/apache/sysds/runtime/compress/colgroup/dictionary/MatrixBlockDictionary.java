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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class MatrixBlockDictionary extends ADictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	private MatrixBlock _data;

	public MatrixBlockDictionary(double[] values, int nCol) {
		_data = Util.matrixBlockFromDenseArray(values, nCol);
		if(_data.isEmpty())
			throw new DMLCompressionException("Invalid construction of empty dictionary");
	}

	public MatrixBlockDictionary(MatrixBlock data) {
		_data = data;
		if(_data.isEmpty())
			throw new DMLCompressionException("Invalid construction of empty dictionary");
	}

	public MatrixBlock getMatrixBlock() {
		return _data;
	}

	@Override
	public double[] getValues() {
		LOG.warn("Inefficient call to getValues for a MatrixBlockDictionary");
		if(_data.isInSparseFormat())
			_data.sparseToDense();
		return _data.getDenseBlockValues();
	}

	@Override
	public double getValue(int i) {
		final int nCol = _data.getNumColumns();
		final int row = i / nCol;
		if(row > _data.getNumRows())
			return 0;
		final int col = i % nCol;
		return _data.quickGetValue(row, col);
	}

	@Override
	public long getInMemorySize() {
		return 8 + _data.estimateSizeInMemory();
	}

	public static long getInMemorySize(int numberValues, int numberColumns, double sparsity) {
		return 8 + MatrixBlock.estimateSizeInMemory(numberValues, numberColumns, sparsity);
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		if(fn.getBuiltinCode() == BuiltinCode.MAX)
			return fn.execute(init, _data.max());
		else if(fn.getBuiltinCode() == BuiltinCode.MIN)
			return fn.execute(init, _data.min());
		else
			throw new NotImplementedException();
	}

	@Override
	public double aggregate(double init, Builtin fn, double[] reference) {
		final int nCol = reference.length;
		final int nRows = _data.getNumRows();
		double ret = init;

		for(int i = 0; i < nCol; i++)
			ret = fn.execute(ret, reference[i]);

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRows; i++) {
				if(sb.isEmpty(i))
					continue;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				for(int k = apos; k < alen; k++) {
					final double v = avals[k] + reference[aix[k]];
					ret = fn.execute(ret, v);
				}
			}
		}
		else if(!_data.isEmpty()) {
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < nRows; k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret = fn.execute(ret, v);
				}
			}
		}

		return ret;
	}

	@Override
	public double[] aggregateRows(Builtin fn, int nCol) {
		double[] ret = new double[_data.getNumRows()];
		if(_data.isEmpty())
			return ret;
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					ret[i] = avals[apos];
					for(int j = apos + 1; j < alen; j++)
						ret[i] = fn.execute(ret[i], avals[j]);

					if(sb.size(i) < _data.getNumColumns())
						ret[i] = fn.execute(ret[i], 0);
				}
				else
					ret[i] = fn.execute(ret[i], 0);
			}
		}
		else if(nCol == 1)
			return _data.getDenseBlockValues();
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				ret[k] = values[off++];
				for(int j = 1; j < _data.getNumColumns(); j++)
					ret[k] = fn.execute(ret[k], values[off++]);
			}
		}
		return ret;
	}

	@Override
	public double[] aggregateRows(Builtin fn, double[] reference) {
		final int nCol = reference.length;
		final int nRows = _data.getNumRows();
		final double[] ret = new double[nRows + 1];

		ret[nRows] = reference[0];
		for(int i = 1; i < nCol; i++)
			ret[nRows] = fn.execute(ret[nRows], reference[i]);

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRows; i++) {
				if(sb.isEmpty(i))
					ret[i] = ret[nRows];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 1;
					ret[i] = (aix[k] == 0) ? avals[k++] + reference[0] : reference[0];
					for(; j < _data.getNumColumns() && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						ret[i] = fn.execute(ret[i], v);
					}
					for(; j < _data.getNumColumns(); j++)
						ret[i] = fn.execute(ret[i], reference[j]);
				}
			}
		}
		else if(!_data.isEmpty()) {
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < nRows; k++) {
				ret[k] = values[off++] + reference[0];
				for(int j = 1; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret[k] = fn.execute(ret[k], v);
				}
			}
		}

		return ret;
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes) {
		if(_data.isEmpty()) {
			for(int j = 0; j < colIndexes.length; j++) {
				final int idx = colIndexes[j];
				c[idx] = fn.execute(c[idx], 0);
			}
		}
		else if(_data.isInSparseFormat()) {
			MatrixBlock t = LibMatrixReorg.transposeInPlace(_data, 1);
			if(!t.isInSparseFormat()) {
				throw new NotImplementedException();
			}
			SparseBlock sbt = t.getSparseBlock();

			for(int i = 0; i < _data.getNumColumns(); i++) {
				final int idx = colIndexes[i];
				if(!sbt.isEmpty(i)) {
					final int apos = sbt.pos(i);
					final int alen = sbt.size(i) + apos;
					final double[] avals = sbt.values(i);
					for(int j = apos; j < alen; j++)
						c[idx] = fn.execute(c[idx], avals[j]);
					if(alen != _data.getNumRows())
						c[idx] = fn.execute(c[idx], 0);
				}
				else
					c[idx] = fn.execute(c[idx], 0);
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final int idx = colIndexes[j];
					c[idx] = fn.execute(c[idx], values[off++]);
				}
			}
		}
	}

	@Override
	public void aggregateCols(double[] c, Builtin fn, int[] colIndexes, double[] reference) {
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();

		for(int j = 0; j < colIndexes.length; j++) {
			final int idx = colIndexes[j];
			c[idx] = fn.execute(c[idx], reference[j]);
		}
		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					continue;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final double[] avals = sb.values(i);
				final int[] aix = sb.indexes(i);
				// This is a cool trick but it only works with min / max.
				for(int k = apos; k < alen; k++) {
					final int idx = colIndexes[aix[k]];
					c[idx] = fn.execute(c[idx], avals[k] + reference[aix[k]]);
				}
			}
		}
		else if(!_data.isEmpty()) {
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < nRow; k++) {
				for(int j = 0; j < nCol; j++) {
					final int idx = colIndexes[j];
					c[idx] = fn.execute(c[idx], values[off++] + reference[j]);
				}
			}
		}
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op) {
		MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
		if(res.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(res);
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op, double[] reference, double[] newReference) {
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();
		final MatrixBlock ret = new MatrixBlock(nRow, nCol, false);
		ret.allocateDenseBlock();
		final double[] retV = ret.getDenseBlockValues();
		int off = 0;
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					for(int j = 0; j < nCol; j++)
						retV[off++] = op.executeScalar(reference[j]) - newReference[j];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						retV[off++] = op.executeScalar(v) - newReference[j];
					}
					for(; j < nCol; j++)
						retV[off++] = op.executeScalar(reference[j]) - newReference[j];
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.executeScalar(values[off] + reference[j]) - newReference[j];
					off++;
				}
			}
		}

		ret.recomputeNonZeros();
		ret.examSparsity();
		if(ret.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(ret);

	}

	@Override
	public ADictionary inplaceScalarOp(ScalarOperator op) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
		final int lastRow = res.getNumRows();
		MatrixBlock res2 = new MatrixBlock(lastRow + 1, res.getNumColumns(), true);
		if(res.isEmpty())
			for(int i = 0; i < numCols; i++)
				res2.appendValue(lastRow, i, newVal);
		else
			res.append(new MatrixBlock(1, numCols, newVal), res2, false);

		if(res2.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(res2);
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		throw new NotImplementedException("Binary row op left is not supported for Uncompressed Matrix, "
			+ "Implement support for VMr in MatrixBLock Binary Cell operations");
		// MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		// return new MatrixBlockDictionary(rowVector.binaryOperations(op, _data, null));
	}

	@Override
	public Dictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyBinaryRowOpLeftAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		MatrixBlock tmp = _data.append(new MatrixBlock(1, _data.getNumColumns(), 0), null, false);
		return new MatrixBlockDictionary(rowVector.binaryOperations(op, tmp, null));
	}

	@Override
	public ADictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes) {
		MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		return new MatrixBlockDictionary(_data.binaryOperations(op, rowVector, null));
	}

	@Override
	public Dictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes, double[] reference,
		double[] newReference) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary applyBinaryRowOpRightAppendNewEntry(BinaryOperator op, double[] v, int[] colIndexes) {
		MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		MatrixBlock tmp = _data.append(new MatrixBlock(1, _data.getNumColumns(), 0), null, false);
		return new MatrixBlockDictionary(tmp.binaryOperations(op, rowVector, null));
	}

	@Override
	public ADictionary clone() {
		MatrixBlock ret = new MatrixBlock();
		ret.copy(_data);
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary cloneAndExtend(int len) {
		throw new NotImplementedException();
	}

	@Override
	public boolean isLossy() {
		return false;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return _data.getNumRows();
	}

	@Override
	public double[] sumAllRowsToDouble(int nrColumns) {
		double[] ret = new double[_data.getNumRows()];

		if(_data.isEmpty())
			return ret;
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						ret[i] += avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					ret[k] += v;
				}
			}
		}
		return ret;
	}

	@Override
	public double[] sumAllRowsToDouble(double[] reference){
		final int nCol = reference.length;
		final int numVals = _data.getNumRows();
		final double[] ret = new double[numVals + 1];

		final int finalIndex = numVals;
		for(int i = 0; i < nCol; i++)
			ret[finalIndex] += reference[i];

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < numVals; i++) {
				if(sb.isEmpty(i))
					ret[i] = ret[finalIndex];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 0;
					for(; j < _data.getNumColumns() && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						ret[i] += v ;
					}
					for(; j < _data.getNumColumns(); j++)
						ret[i] += reference[j];
				}

			}
		}
		else if(!_data.isEmpty()) {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < numVals; k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret[k] += v ;
				}
			}
		}

		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		final double[] ret = new double[_data.getNumRows()];

		if(_data.isEmpty())
			return ret;
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						ret[i] += avals[j] * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					ret[k] += v * v;
				}
			}
		}
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(double[] reference) {
		final int nCol = reference.length;
		final int numVals = _data.getNumRows();
		final double[] ret = new double[numVals + 1];

		final int finalIndex = numVals;
		for(int i = 0; i < nCol; i++)
			ret[finalIndex] += reference[i] * reference[i];

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < numVals; i++) {
				if(sb.isEmpty(i))
					ret[i] = ret[finalIndex];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 0;
					for(; j < _data.getNumColumns() && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						ret[i] += v * v;
					}
					for(; j < _data.getNumColumns(); j++)
						ret[i] += reference[j] * reference[j];
				}

			}
		}
		else if(!_data.isEmpty()) {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < numVals; k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret[k] += v * v;
				}
			}
		}

		return ret;
	}

	@Override
	public double sumRow(int k, int nrColumns) {
		throw new NotImplementedException();
	}

	@Override
	public double sumRowSq(int k, int nrColumns) {
		throw new NotImplementedException();
	}

	@Override
	public double sumRowSq(int k, int nrColumns, double[] reference) {
		throw new NotImplementedException();
	}

	@Override
	public double[] colSum(int[] counts, int nCol) {
		if(_data.isEmpty())
			return null;
		double[] ret = new double[nCol];
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					// double tmpSum = 0;
					final int count = counts[i];
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						ret[aix[j]] += count * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					ret[j] += v * countK;
				}
			}
		}
		return ret;
	}

	@Override
	public void colSum(double[] c, int[] counts, int[] colIndexes) {
		if(_data.isEmpty())
			return;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					// double tmpSum = 0;
					final int count = counts[i];
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						c[colIndexes[aix[j]]] += count * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					c[colIndexes[j]] += v * countK;
				}
			}
		}
	}

	@Override
	public void colSumSq(double[] c, int[] counts, int[] colIndexes) {
		if(_data.isEmpty())
			return;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					// double tmpSum = 0;
					final int count = counts[i];
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						c[colIndexes[aix[j]]] += count * avals[j] * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					c[colIndexes[j]] += v * v * countK;
				}
			}
		}
	}

	@Override
	public void colSumSq(double[] c, int[] counts, int[] colIndexes, double[] reference) {
		final int nCol = reference.length;
		final int nRow = _data.getNumRows();
		for(int i = 0; i < nCol; i++)
			c[colIndexes[i]] += reference[i] * reference[i] * counts[nRow];

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				final int countK = counts[i];
				if(sb.isEmpty(i))
					for(int j = 0; j < nCol; j++)
						c[colIndexes[j]] += reference[j] * reference[j] * countK;
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 0;
					for(; j < _data.getNumColumns() && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						c[colIndexes[j]] += v * v * countK;
					}
					for(; j < _data.getNumColumns(); j++)
						c[colIndexes[j]] += reference[j] * reference[j] * countK;
				}
			}
		}
		else if(!_data.isEmpty()) {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < nRow; k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					c[colIndexes[j]] += v * v * countK;
				}
			}
		}
	}

	@Override
	public double sum(int[] counts, int ncol) {
		double tmpSum = 0;
		if(_data.isEmpty())
			return tmpSum;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					final int count = counts[i];
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						tmpSum += count * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					tmpSum += v * countK;
				}
			}
		}
		return tmpSum;
	}

	@Override
	public double sumSq(int[] counts, int ncol) {
		double tmpSum = 0;
		if(_data.isEmpty())
			return tmpSum;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i)) {
					final int count = counts[i];
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					for(int j = apos; j < alen; j++) {
						tmpSum += count * avals[j] * avals[j];
					}
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					tmpSum += v * v * countK;
				}
			}
		}
		return tmpSum;
	}

	@Override
	public double sumSq(int[] counts, double[] reference) {
		final int nCol = reference.length;
		final int numVals = _data.getNumRows();
		double ret = 0;
		for(int i = 0; i < nCol; i++)
			ret += reference[i] * reference[i];
		final double ref = ret;
		ret *= counts[numVals];

		if(!_data.isEmpty() && _data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < numVals; i++) {
				final int countK = counts[i];
				if(sb.isEmpty(i))
					ret += ref * countK;
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 0;
					for(; j < _data.getNumColumns() && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						ret += v * v * countK;
					}
					for(; j < _data.getNumColumns(); j++)
						ret += reference[j] * reference[j] * countK;
				}

			}
		}
		else if(!_data.isEmpty()) {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < numVals; k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret += v * v * countK;
				}
			}
		}

		return ret;
	}

	@Override
	public void addMaxAndMin(double[] ret, int[] colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		MatrixBlock retBlock = _data.slice(0, _data.getNumRows() - 1, idxStart, idxEnd - 1);
		return new MatrixBlockDictionary(retBlock);
	}

	@Override
	public ADictionary reExpandColumns(int max) {
		throw new NotImplementedException();
	}

	@Override
	public boolean containsValue(double pattern) {
		return _data.containsValue(pattern);
	}

	@Override
	public boolean containsValue(double pattern, double[] reference) {

		if(_data.isEmpty()) {
			for(double d : reference)
				if(pattern == d)
					return true;
			return false;
		}
		else if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(sb.isEmpty(i))
					continue;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				int k = apos;
				int j = 0;
				for(; j < _data.getNumColumns() && k < alen; j++) {
					if(aix[k] == j) {
						if(reference[j] + avals[k++] == pattern)
							return true;
					}
					else {
						if(reference[j] == pattern)
							return true;
					}
				}
				for(; j < _data.getNumColumns(); j++)
					if(reference[j] == pattern)
						return true;

			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			final int nCol = reference.length;
			for(int i = 0; i < values.length; i++)
				if(values[i] + reference[i % nCol] == pattern)
					return true;

		}
		return false;
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		if(_data.isEmpty())
			return 0;

		long nnz = 0;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++)
				if(!sb.isEmpty(i))
					nnz += sb.size(i) * counts[i];
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int i = 0; i < _data.getNumRows(); i++) {
				int countThisTuple = 0;
				for(int j = 0; j < _data.getNumColumns(); j++) {
					double v = values[off++];
					if(v != 0)
						countThisTuple++;
				}
				nnz += countThisTuple * counts[i];
			}
		}
		return nnz;
	}

	@Override
	public long getNumberNonZeros(int[] counts, double[] reference, int nRows) {
		long nnz = 0;
		for(double d : reference)
			if(d != 0)
				nnz++;
		if(_data.isEmpty()) {
			// sum counts
			return nnz * nRows;
		}
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			long emptyRowNNZ = nnz;
			nnz *= counts[counts.length - 1]; // multiply count with the common value count in reference.
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(sb.isEmpty(i))
					nnz += emptyRowNNZ * counts[i];
				else {
					int countThis = 0;
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int k = apos;
					int j = 0;
					for(; j < _data.getNumColumns() && k < alen; j++) {
						if(aix[k] == j) {
							if(reference[j] + avals[k++] != 0)
								countThis++;
						}
						else {
							if(reference[j] != 0)
								countThis++;
						}
					}
					for(; j < _data.getNumColumns(); j++)
						if(reference[j] != 0)
							countThis++;

					nnz += countThis * counts[i];
				}
			}
		}
		else {
			nnz *= counts[counts.length - 1]; // multiply count with the common value count in reference.
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int i = 0; i < _data.getNumRows(); i++) {
				int countThisTuple = 0;
				for(int j = 0; j < _data.getNumColumns(); j++)
					if(values[off++] + reference[j] != 0)
						countThisTuple++;
				nnz += countThisTuple * counts[i];
			}
		}
		return nnz;
	}

	@Override
	public void addToEntry(Dictionary d, int fr, int to, int nCol) {
		double[] v = d.getValues();
		if(_data.isEmpty())
			return;
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			if(sb.isEmpty(fr))
				return;
			final int apos = sb.pos(fr);
			final int alen = sb.size(fr) + apos;
			final int[] aix = sb.indexes(fr);
			final double[] avals = sb.values(fr);
			final int offsetTo = nCol * to;
			for(int j = apos; j < alen; j++) {
				v[offsetTo + aix[j]] += avals[j];
			}
		}
		else {
			final int sf = nCol * fr; // start from
			final int ef = sf + nCol; // end from
			final double[] thisV = _data.getDenseBlockValues();
			for(int i = sf, j = nCol * to; i < ef; i++, j++) {
				v[j] += thisV[i];
			}
		}
	}

	@Override
	public double[] getTuple(int index, int nCol) {
		if(_data.isEmpty() || index >= _data.getNumRows())
			return null;

		final double[] tuple = new double[nCol];
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			if(sb.isEmpty(index))
				return null;
			final int apos = sb.pos(index);
			final int alen = sb.size(index) + apos;
			final int[] aix = sb.indexes(index);
			final double[] avals = sb.values(index);
			for(int j = apos; j < alen; j++)
				tuple[aix[j]] = avals[j];

			return tuple;
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int offset = index * nCol;
			for(int i = 0; i < nCol; i++, offset++)
				tuple[i] = values[offset];
			return tuple;
		}
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		if(_data.isEmpty())
			throw new NotImplementedException("Should not extract from empty matrix");
		else if(_data.isInSparseFormat()) {
			throw new NotImplementedException("Not supporting extracting from sparse matrix yet");
		}
		else {
			final int nRow = _data.getNumRows() - 1;
			final int nCol = _data.getNumColumns();
			double[] values = _data.getDenseBlockValues();
			MatrixBlock res = new MatrixBlock(nCol, nRow, false);
			res.allocateBlock();
			double[] resVals = res.getDenseBlockValues();
			for(int i = 0, off = 0; i < nRow; i++)
				for(int j = 0; j < nCol; j++, off++)
					resVals[off] = values[off] - tuple[j];

			res.examSparsity();
			if(res.isEmpty())
				return null;
			return new MatrixBlockDictionary(res);
		}
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		// Simply return this.
		return this;
	}

	@Override
	public String getString(int colIndexes) {
		if(_data.isInSparseFormat() || _data.getNumColumns() > 1)
			return "\n" + _data.toString();
		else
			return Arrays.toString(_data.getDenseBlockValues());
	}

	@Override
	public String toString() {
		if(_data.isInSparseFormat() || _data.getNumColumns() > 1)
			return "MatrixBlock Dictionary :\n" + _data.toString();
		else
			return "MatrixBlock Dictionary : " + Arrays.toString(_data.getDenseBlockValues());
	}

	@Override
	public ADictionary scaleTuples(int[] scaling, int nCol) {
		if(_data.isEmpty()) {
			throw new NotImplementedException("could return null here? or empty DictionaryMatrixBlock...");
		}
		else if(_data.isInSparseFormat()) {
			MatrixBlock retBlock = new MatrixBlock(_data.getNumRows(), _data.getNumColumns(), true);
			retBlock.allocateSparseRowsBlock(true);
			SparseBlock sbRet = retBlock.getSparseBlock();
			SparseBlock sbThis = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sbThis.isEmpty(i)) {
					sbRet.set(i, sbThis.get(i), true);

					final int count = scaling[i];
					final int apos = sbRet.pos(i);
					final int alen = sbRet.size(i) + apos;
					final double[] avals = sbRet.values(i);
					for(int j = apos; j < alen; j++)
						avals[j] = count * avals[j];
				}
			}
			retBlock.setNonZeros(_data.getNonZeros());
			return new MatrixBlockDictionary(retBlock);
		}
		else {
			final double[] _values = _data.getDenseBlockValues();
			final double[] scaledValues = new double[_values.length];
			int off = 0;
			for(int tuple = 0; tuple < _values.length / nCol; tuple++) {
				final int scale = scaling[tuple];
				for(int v = 0; v < nCol; v++) {
					scaledValues[off] = _values[off] * scale;
					off++;
				}
			}
			DenseBlockFP64 db = new DenseBlockFP64(new int[] {_data.getNumRows(), _data.getNumColumns()}, scaledValues);
			MatrixBlock retBlock = new MatrixBlock(_data.getNumRows(), _data.getNumColumns(), db);
			retBlock.setNonZeros(_data.getNonZeros());
			return new MatrixBlockDictionary(retBlock);
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(DictionaryFactory.Type.MATRIX_BLOCK_DICT.ordinal());
		_data.write(out);
	}

	public static MatrixBlockDictionary read(DataInput in) throws IOException {
		MatrixBlock ret = new MatrixBlock();
		ret.readFields(in);
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public long getExactSizeOnDisk() {
		return 1 + _data.getExactSizeOnDisk();
	}

	@Override
	public MatrixBlockDictionary preaggValuesFromDense(final int numVals, final int[] colIndexes,
		final int[] aggregateColumns, final double[] b, final int cut) {

		double[] ret = new double[numVals * aggregateColumns.length];
		if(_data.isEmpty())
			return null;
		else if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(sb.isEmpty(i))
					continue;
				final int off = aggregateColumns.length * i;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final double[] avals = sb.values(i);
				final int[] aix = sb.indexes(i);
				for(int j = apos; j < alen; j++) {
					final int idb = colIndexes[aix[j]] * cut;
					final double v = avals[j];
					for(int h = 0; h < aggregateColumns.length; h++)
						ret[off + h] += v * b[idb + aggregateColumns[h]];
				}
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			for(int k = 0, off = 0;
				k < numVals * colIndexes.length;
				k += colIndexes.length, off += aggregateColumns.length) {
				for(int h = 0; h < colIndexes.length; h++) {
					int idb = colIndexes[h] * cut;
					double v = values[k + h];
					if(v != 0)
						for(int i = 0; i < aggregateColumns.length; i++)
							ret[off + i] += v * b[idb + aggregateColumns[i]];
				}
			}
		}

		DenseBlock dictV = new DenseBlockFP64(new int[] {numVals, aggregateColumns.length}, ret);
		MatrixBlock dictM = new MatrixBlock(numVals, aggregateColumns.length, dictV);
		dictM.recomputeNonZeros();
		dictM.examSparsity();
		return new MatrixBlockDictionary(dictM);

	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		final MatrixBlock ret = _data.replaceOperations(new MatrixBlock(), pattern, replace);
		if(ret.isEmpty())
			return null;
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary replace(double pattern, double replace, double[] reference) {
		final int nRow = _data.getNumRows();
		final int nCol = _data.getNumColumns();
		final MatrixBlock ret = new MatrixBlock(nRow, nCol, false);
		ret.allocateDenseBlock();
		final double[] retV = ret.getDenseBlockValues();
		int off = 0;
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					for(int j = 0; j < nCol; j++)
						retV[off++] = pattern == reference[j] ? replace - reference[j] : 0;
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						retV[off++] = pattern == v ? replace - reference[j] : v - reference[j];
					}
					for(; j < nCol; j++)
						retV[off++] = pattern == reference[j] ? replace - reference[j] : 0;
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					final double v = values[off];
					retV[off++] = pattern == v + reference[j] ? replace - reference[j] : v;
				}
			}
		}

		ret.recomputeNonZeros();
		ret.examSparsity();
		if(ret.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(ret);

	}

	@Override
	public ADictionary replaceZeroAndExtend(double replace, int nCol) {
		final int nRows = _data.getNumRows();
		final int nCols = _data.getNumColumns();
		final long nonZerosOut = (nRows + 1) * nCols;
		final MatrixBlock ret = new MatrixBlock(nRows + 1, nCols, false);
		ret.allocateBlock();
		ret.setNonZeros(nonZerosOut);
		final double[] retValues = ret.getDenseBlockValues();
		if(_data.isEmpty())
			Arrays.fill(retValues, replace);
		else if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRows; i++) {
				for(int h = i * nCols; h < i * nCols + nCols; h++)
					retValues[h] = replace;
				if(sb.isEmpty(i))
					continue;
				final int off = nCol * i;
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final double[] avals = sb.values(i);
				final int[] aix = sb.indexes(i);
				for(int j = apos; j < alen; j++) {
					final int idb = aix[j];
					final double v = avals[j];
					retValues[off + idb] = v;
				}
			}
			for(int h = nRows * nCols; h < nonZerosOut; h++)
				retValues[h] = replace;
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int k = 0; k < nRows; k++)
				for(int h = k * nCols; h < k * nCols + nCols; h++)
					retValues[h] = values[h] == 0 ? replace : values[h];

			for(int h = nRows * nCols; h < nonZerosOut; h++)
				retValues[h] = replace;
		}
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public double product(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public void colProduct(double[] res, int[] counts, int[] colIndexes) {
		throw new NotImplementedException();
	}
}
