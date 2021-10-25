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
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

public class MatrixBlockDictionary extends ADictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	private MatrixBlock _data;

	public MatrixBlockDictionary(double[] values, int nCol) {
		_data = Util.matrixBlockFromDenseArray(values, nCol);
	}

	public MatrixBlockDictionary(MatrixBlock data) {
		_data = data;
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
	public double[] aggregateTuples(Builtin fn, int nCol) {
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
	public ADictionary inplaceScalarOp(ScalarOperator op) {
		MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
		return new MatrixBlockDictionary(res);
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		MatrixBlock res = _data.scalarOperations(op, new MatrixBlock());
		final int lastRow = res.getNumRows();
		MatrixBlock res2 = new MatrixBlock(lastRow + 1, res.getNumColumns(), true);
		if(res.isEmpty()) {
			for(int i = 0; i < numCols; i++)
				res2.appendValue(lastRow, i, newVal);
			return new MatrixBlockDictionary(res2);
		}
		else {
			res.append(new MatrixBlock(1, numCols, newVal), res2, false);
			return new MatrixBlockDictionary(res2);
		}
	}

	@Override
	public ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		return new MatrixBlockDictionary(rowVector.binaryOperations(op, _data, null));
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
	public double[] sumAllRowsToDouble(boolean square, int nrColumns) {
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
						ret[i] += (square) ? avals[j] * avals[j] : avals[j];
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
					ret[k] += (square) ? v * v : v;
				}
			}
		}
		return ret;
	}

	@Override
	public double sumRow(int k, boolean square, int nrColumns) {
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
	public void colSum(double[] c, int[] counts, int[] colIndexes, boolean square) {
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
						c[colIndexes[aix[j]]] += square ? count * avals[j] * avals[j] : count * avals[j];
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
					c[colIndexes[j]] += square ? v * v * countK : v * countK;
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
	public double sumsq(int[] counts, int ncol) {
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
	public String getString(int colIndexes) {
		return _data.toString();
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
		DenseBlockFP64 b = new DenseBlockFP64(new int[] {1, tuple.length}, tuple);
		MatrixBlock rowVector = new MatrixBlock(1, tuple.length, b);
		MatrixBlock res = new MatrixBlock(_data.getNumColumns(), _data.getNumRows(), _data.isInSparseFormat());
		_data.binaryOperations(new BinaryOperator(Minus.getMinusFnObject()), rowVector, res);
		return new MatrixBlockDictionary(res);
	}

	@Override
	public MatrixBlockDictionary getMBDict(int nCol) {
		// Simply return this.
		return this;
	}

	@Override
	public String toString() {
		return "MatrixBlock Dictionary :" + _data.toString();
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
		dictM.getNonZeros();
		dictM.examSparsity();
		return new MatrixBlockDictionary(dictM);

	}

	@Override
	public ADictionary replace(double pattern, double replace, int nCol) {
		MatrixBlock ret = _data.replaceOperations(new MatrixBlock(), pattern, replace);
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary replaceZeroAndExtend(double replace, int nCol) {
		final int nRows = _data.getNumRows();
		final int nCols = _data.getNumColumns();
		final long nonZerosOut = (nRows + 1) * nCols;
		final MatrixBlock ret = new MatrixBlock(_data.getNumRows() + 1, _data.getNumColumns(), false);
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
