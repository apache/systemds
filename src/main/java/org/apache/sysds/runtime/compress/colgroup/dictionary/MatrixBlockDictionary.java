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
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class MatrixBlockDictionary extends ADictionary {

	private static final long serialVersionUID = 2535887782150955098L;

	final private MatrixBlock _data;

	/**
	 * Unsafe private constructor that does not check the data validity. USE WITH CAUTION.
	 * 
	 * @param data The matrix block data.
	 */
	public MatrixBlockDictionary(MatrixBlock data) {
		this(data, true);
	}

	/**
	 * Unsafe private constructor that does not check the data validity. USE WITH CAUTION.
	 * 
	 * @param data  The matrix block data.
	 * @param check Check the nonZeros in the dict
	 */
	public MatrixBlockDictionary(MatrixBlock data, boolean check) {
		if(check) {
			data.examSparsity(true);
			if(data.isEmpty())
				throw new DMLCompressionException("Invalid construction of empty dictionary");
			else if(data.isInSparseFormat() && data.getSparseBlock() instanceof SparseBlockMCSR) {
				SparseBlock csr = SparseBlockFactory.copySparseBlock(SparseBlock.Type.CSR, data.getSparseBlock(), false);
				data.setSparseBlock(csr);
			}
		}
		_data = data;
	}

	public static MatrixBlockDictionary createDictionary(double[] values, int nCol, boolean check) {
		final MatrixBlock mb = Util.matrixBlockFromDenseArray(values, nCol, check);
		if(mb.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(mb, check);
	}

	public MatrixBlock getMatrixBlock() {
		return _data;
	}

	@Override
	public double[] getValues() {
		if(_data.isInSparseFormat()) {
			LOG.warn("Inefficient call to getValues for a MatrixBlockDictionary because it was sparse");
			throw new DMLCompressionException("Should not call this function");
			// _data.sparseToDense();
		}
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
	public final double getValue(int r, int c, int nCol) {
		return _data.quickGetValue(r, c);
	}

	@Override
	public long getInMemorySize() {
		// object reference to a matrix block + matrix block size.
		return 8 + _data.estimateSizeInMemory();
	}

	public static long getInMemorySize(int numberValues, int numberColumns, double sparsity) {
		// object reference to a matrix block + matrix block size.
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
	public double aggregateWithReference(double init, Builtin fn, double[] reference, boolean def) {
		final int nCol = reference.length;
		final int nRows = _data.getNumRows();
		double ret = init;

		if(def)
			for(int i = 0; i < nCol; i++)
				ret = fn.execute(ret, reference[i]);

		if(_data.isInSparseFormat()) {
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
			if(!def) {
				final int[] nnz = LibMatrixReorg.countNnzPerColumn(_data);
				for(int i = 0; i < nnz.length; i++)
					if(nnz[i] < nRows)
						ret = fn.execute(ret, reference[i]);

			}
		}
		else {
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

		if(_data.isInSparseFormat()) {
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
	public double[] aggregateRowsWithDefault(Builtin fn, double[] defaultTuple) {
		final int nRow = _data.getNumRows();
		final int nCol = defaultTuple.length;
		double[] ret = new double[_data.getNumRows() + 1];
		if(_data.isInSparseFormat()) {
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
		else if(nCol == 1) {
			System.arraycopy(_data.getDenseBlockValues(), 0, ret, 0, _data.getNumRows());
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				ret[k] = values[off++];
				for(int j = 1; j < _data.getNumColumns(); j++)
					ret[k] = fn.execute(ret[k], values[off++]);
			}
		}

		ret[nRow] = defaultTuple[0];
		for(int i = 1; i < defaultTuple.length; i++)
			ret[nRow] = fn.execute(ret[nRow], defaultTuple[i]);

		return ret;
	}

	@Override
	public double[] aggregateRowsWithReference(Builtin fn, double[] reference) {
		final int nCol = reference.length;
		final int nRows = _data.getNumRows();
		final double[] ret = new double[nRows + 1];

		ret[nRows] = reference[0];
		for(int i = 1; i < nCol; i++)
			ret[nRows] = fn.execute(ret[nRows], reference[i]);

		if(_data.isInSparseFormat()) {
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
		else {
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
		if(_data.isInSparseFormat()) {
			MatrixBlock t = LibMatrixReorg.transpose(_data);
			if(!t.isInSparseFormat()) // highly unlikely.
				throw new NotImplementedException("Not implemented aggregate Cols on dense transposed dict.");

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
	public void aggregateColsWithReference(double[] c, Builtin fn, int[] colIndexes, double[] reference, boolean def) {
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();

		if(def)
			for(int j = 0; j < colIndexes.length; j++) {
				final int idx = colIndexes[j];
				c[idx] = fn.execute(c[idx], reference[j]);
			}
		if(_data.isInSparseFormat()) {
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
			if(!def) {
				final int[] nnz = LibMatrixReorg.countNnzPerColumn(_data);
				for(int i = 0; i < nnz.length; i++)
					if(nnz[i] < nRow) {
						final int idx = colIndexes[i];
						c[idx] = fn.execute(c[idx], reference[i]);
					}
			}
		}
		else {
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
	public ADictionary applyScalarOpAndAppend(ScalarOperator op, double v0, int nCol) {
		// guaranteed to be densifying since this only is called if op(0) is != 0
		// set the entire output to v0.
		final MatrixBlock ret = new MatrixBlock(_data.getNumRows() + 1, _data.getNumColumns(), v0);
		final double[] retV = ret.getDenseBlockValues();

		if(_data.isInSparseFormat()) {
			final int nRow = _data.getNumRows();
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					continue;

				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				for(int k = apos; k < alen; k++)
					retV[i * nCol + aix[k]] = op.executeScalar(avals[i]);
			}
		}
		else {
			final double[] v = _data.getDenseBlockValues();
			for(int i = 0; i < v.length; i++)
				retV[i] = op.executeScalar(v[i]);
		}

		ret.recomputeNonZeros();
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary applyUnaryOp(UnaryOperator op) {
		MatrixBlock res = _data.unaryOperations(op, new MatrixBlock());
		if(res.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(res);
	}

	@Override
	public ADictionary applyUnaryOpAndAppend(UnaryOperator op, double v0, int nCol) {
		// guaranteed to be densifying since this only is called if op(0) is != 0
		// set the entire output to v0.
		final MatrixBlock ret = new MatrixBlock(_data.getNumRows() + 1, _data.getNumColumns(), v0);
		final double[] retV = ret.getDenseBlockValues();

		if(_data.isInSparseFormat()) {
			final int nRow = _data.getNumRows();
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					continue;

				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				for(int k = apos; k < alen; k++)
					retV[i * nCol + aix[k]] = op.fn.execute(avals[i]);
			}
		}
		else {
			final double[] v = _data.getDenseBlockValues();
			for(int i = 0; i < v.length; i++)
				retV[i] = op.fn.execute(v[i]);
		}

		ret.recomputeNonZeros();
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary applyScalarOpWithReference(ScalarOperator op, double[] reference, double[] newReference) {
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
	public ADictionary applyUnaryOpWithReference(UnaryOperator op, double[] reference, double[] newReference) {
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
						retV[off++] = op.fn.execute(reference[j]) - newReference[j];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double v = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						retV[off++] = op.fn.execute(v) - newReference[j];
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(reference[j]) - newReference[j];
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(values[off] + reference[j]) - newReference[j];
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
	public ADictionary binOpLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		LOG.warn("Binary row op left is not supported for Uncompressed Matrix, "
			+ "Implement support for VMr in MatrixBlock Binary Cell operations");
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
						retV[off++] = op.fn.execute(v[j], 0);
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double vx = aix[k] == j ? avals[k++] : 0;
						retV[off++] = op.fn.execute(v[j], vx);
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(v[j], 0);
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(v[j], values[off]);
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
	public ADictionary binOpLeftAndAppend(BinaryOperator op, double[] v, int[] colIndexes) {
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();

		final MatrixBlock ret = new MatrixBlock(nRow + 1, nCol, false);
		ret.allocateDenseBlock();
		final double[] retV = ret.getDenseBlockValues();

		int off = 0;
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					for(int j = 0; j < nCol; j++)
						retV[off++] = op.fn.execute(v[j], 0);
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double vx = aix[k] == j ? avals[k++] : 0;
						retV[off++] = op.fn.execute(v[j], vx);
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(v[j], 0);
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(v[j], values[off]);
					off++;
				}
			}
		}
		for(int j = 0; j < nCol; j++) {
			retV[off] = op.fn.execute(v[j], 0);
			off++;
		}

		ret.recomputeNonZeros();
		ret.examSparsity();
		if(ret.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(ret);
	}

	@Override
	public MatrixBlockDictionary binOpLeftWithReference(BinaryOperator op, double[] v, int[] colIndexes,
		double[] reference, double[] newReference) {
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
						retV[off++] = op.fn.execute(v[j], reference[j]) - newReference[j];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double vx = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						retV[off++] = op.fn.execute(v[j], vx) - newReference[j];
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(v[j], reference[j]) - newReference[j];
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(v[j], values[off] + reference[j]) - newReference[j];
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
	public MatrixBlockDictionary binOpRight(BinaryOperator op, double[] v, int[] colIndexes) {
		final MatrixBlock rowVector = Util.extractValues(v, colIndexes);
		final MatrixBlock ret = _data.binaryOperations(op, rowVector, null);
		if(ret.isEmpty())
			return null;
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public ADictionary binOpRightAndAppend(BinaryOperator op, double[] v, int[] colIndexes) {
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();

		final MatrixBlock ret = new MatrixBlock(nRow + 1, nCol, false);
		ret.allocateDenseBlock();
		final double[] retV = ret.getDenseBlockValues();

		int off = 0;
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < nRow; i++) {
				if(sb.isEmpty(i))
					for(int j = 0; j < nCol; j++)
						retV[off++] = op.fn.execute(0, v[j]);
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double vx = aix[k] == j ? avals[k++] : 0;
						retV[off++] = op.fn.execute(vx, v[j]);
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(0, v[j]);
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(values[off], v[j]);
					off++;
				}
			}
		}
		for(int j = 0; j < nCol; j++) {
			retV[off] = op.fn.execute(0, v[j]);
			off++;
		}

		ret.recomputeNonZeros();
		ret.examSparsity();
		if(ret.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(ret);
	}

	@Override
	public MatrixBlockDictionary binOpRight(BinaryOperator op, double[] v) {
		final MatrixBlock rowVector = new MatrixBlock(1, v.length, v);
		final MatrixBlock ret = _data.binaryOperations(op, rowVector, null);
		if(ret.isEmpty())
			return null;
		return new MatrixBlockDictionary(ret);
	}

	@Override
	public MatrixBlockDictionary binOpRightWithReference(BinaryOperator op, double[] v, int[] colIndexes,
		double[] reference, double[] newReference) {
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
						retV[off++] = op.fn.execute(reference[j], v[j]) - newReference[j];
				else {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final int[] aix = sb.indexes(i);
					final double[] avals = sb.values(i);
					int j = 0;
					for(int k = apos; j < nCol && k < alen; j++) {
						final double vx = aix[k] == j ? avals[k++] + reference[j] : reference[j];
						retV[off++] = op.fn.execute(vx, v[j]) - newReference[j];
					}
					for(; j < nCol; j++)
						retV[off++] = op.fn.execute(reference[j], v[j]) - newReference[j];
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					retV[off] = op.fn.execute(values[off] + reference[j], v[j]) - newReference[j];
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
	public ADictionary clone() {
		MatrixBlock ret = new MatrixBlock();
		ret.copy(_data);
		return new MatrixBlockDictionary(ret);
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

		if(_data.isInSparseFormat()) {
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
	public double[] sumAllRowsToDoubleWithDefault(double[] defaultTuple) {
		final int numVals = _data.getNumRows();
		double[] ret = new double[numVals + 1];

		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < numVals; i++) {
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
			for(int k = 0; k < numVals; k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					ret[k] += v;
				}
			}
		}

		for(double v : defaultTuple)
			ret[numVals] += v;
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleWithReference(double[] reference) {
		final int nCol = reference.length;
		final int numVals = _data.getNumRows();
		final double[] ret = new double[numVals + 1];

		final int finalIndex = numVals;
		for(int i = 0; i < nCol; i++)
			ret[finalIndex] += reference[i];

		if(_data.isInSparseFormat()) {
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
						ret[i] += v;
					}
					for(; j < _data.getNumColumns(); j++)
						ret[i] += reference[j];
				}

			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < numVals; k++) {
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					ret[k] += v;
				}
			}
		}

		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSq(int nrColumns) {
		final double[] ret = new double[_data.getNumRows()];
		sumAllRowsToDoubleSq(ret);
		return ret;
	}

	private void sumAllRowsToDoubleSq(double[] ret) {
		if(_data.isInSparseFormat()) {
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
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithDefault(double[] defaultTuple) {
		final double[] ret = new double[_data.getNumRows() + 1];
		sumAllRowsToDoubleSq(ret);
		int defIdx = ret.length - 1;
		for(int j = 0; j < _data.getNumColumns(); j++) {
			final double v = defaultTuple[j];
			ret[defIdx] += v * v;
		}
		return ret;
	}

	@Override
	public double[] sumAllRowsToDoubleSqWithReference(double[] reference) {
		final int nCol = reference.length;
		final int numVals = _data.getNumRows();
		final double[] ret = new double[numVals + 1];

		final int finalIndex = numVals;
		for(int i = 0; i < nCol; i++)
			ret[finalIndex] += reference[i] * reference[i];

		if(_data.isInSparseFormat()) {
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
		else {
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
	public double[] productAllRowsToDouble(int nCol) {
		double[] ret = new double[_data.getNumRows()];
		productAllRowsToDouble(ret, nCol);
		return ret;
	}

	private final void productAllRowsToDouble(double[] ret, int nCol) {
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				if(!sb.isEmpty(i) && sb.size(i) == nCol) {
					// if not equal to nCol ... skip
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					ret[i] = 1;
					for(int j = apos; j < alen; j++) {
						ret[i] *= avals[j];
					}
				}
				else
					ret[i] = 0;
			}
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < _data.getNumRows(); k++) {
				ret[k] = 1;
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					ret[k] *= v;
				}
			}
		}
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[_data.getNumRows() + 1];
		productAllRowsToDouble(ret, defaultTuple.length);
		ret[_data.getNumRows()] = defaultTuple[0];
		for(int j = 1; j < defaultTuple.length; j++)
			ret[_data.getNumRows()] *= defaultTuple[j];

		return ret;
	}

	private final void productAllRowsToDoubleWithReference(double[] ret, int nCol, double[] reference) {
		final int nRow = _data.getNumRows();
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < _data.getNumRows(); i++) {
				ret[i] = 1;
				if(!sb.isEmpty(i)) {
					final int apos = sb.pos(i);
					final int alen = sb.size(i) + apos;
					final double[] avals = sb.values(i);
					final int[] aix = sb.indexes(i);
					for(int j = 0, jj = apos; j < nCol; j++) {
						if(jj < alen && aix[jj] == j)
							ret[i] *= avals[jj++] + reference[j];
						else {
							if(reference[j] == 0) {
								ret[i] = 0;
								continue;
							}
							ret[i] *= reference[j];
						}

					}
				}
				else {
					// empty row
					for(int j = 0; j < nCol; j++)
						ret[i] *= reference[j];
				}
			}
		}
		else {
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < nRow; k++) {
				ret[k] = 1;
				for(int j = 0; j < nCol; j++)
					ret[k] *= values[off++] + reference[j];
			}
		}
	}

	@Override
	public double[] productAllRowsToDoubleWithReference(double[] reference) {
		double[] ret = new double[_data.getNumRows() + 1];
		productAllRowsToDoubleWithReference(ret, reference.length, reference);
		ret[_data.getNumRows()] = reference[0];
		for(int j = 1; j < reference.length; j++)
			ret[_data.getNumRows()] *= reference[j];

		return ret;
	}

	@Override
	public void colSum(double[] c, int[] counts, int[] colIndexes) {
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < counts.length; i++) {
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
			for(int k = 0; k < counts.length; k++) {
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
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < counts.length; i++) {
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
			for(int k = 0; k < counts.length; k++) {
				final int countK = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					c[colIndexes[j]] += v * v * countK;
				}
			}
		}
	}

	@Override
	public void colProduct(double[] res, int[] counts, int[] colIndexes) {
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			final int[] cnt = new int[colIndexes.length];
			for(int i = 0; i < counts.length; i++) {
				if(sb.isEmpty(i))
					continue;
				final int count = counts[i];
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				for(int j = apos; j < alen; j++)
					res[colIndexes[aix[j]]] *= Math.pow(avals[j], count);

				LibMatrixAgg.countAgg(avals, cnt, aix, apos, sb.size(i));

			}
			final int nVal = getNumberOfValues(colIndexes.length);
			for(int j = 0; j < colIndexes.length; j++)
				if(cnt[j] < nVal)
					res[colIndexes[j]] = 0;
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < counts.length; k++) {
				final int count = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++];
					res[colIndexes[j]] *= Math.pow(v, count);
				}
			}
		}
		correctNan(res, colIndexes);
	}

	@Override
	public void colProductWithReference(double[] res, int[] counts, int[] colIndexes, double[] reference) {

		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			final int[] cnt = new int[colIndexes.length];
			for(int i = 0; i < counts.length; i++) {
				if(sb.isEmpty(i))
					continue;
				final int count = counts[i];
				final int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				final double[] avals = sb.values(i);
				for(int j = apos; j < alen; j++)
					res[colIndexes[aix[j]]] *= Math.pow(avals[j] + reference[aix[j]], count);

				LibMatrixAgg.countAgg(avals, cnt, aix, apos, sb.size(i));

			}
			final int nVal = getNumberOfValues(colIndexes.length);
			for(int j = 0; j < colIndexes.length; j++)
				if(cnt[j] < nVal && cnt[j] - nVal != 0)
					res[colIndexes[j]] *= Math.pow(reference[j], cnt[j]);
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int k = 0; k < counts.length; k++) {
				final int count = counts[k];
				for(int j = 0; j < _data.getNumColumns(); j++) {
					final double v = values[off++] + reference[j];
					res[colIndexes[j]] *= Math.pow(v, count);
				}
			}
		}
		correctNan(res, colIndexes);
	}

	@Override
	public void colSumSqWithReference(double[] c, int[] counts, int[] colIndexes, double[] reference) {
		final int nCol = reference.length;
		final int nRow = counts.length;
		if(_data.isInSparseFormat()) {
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
		else {
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
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < counts.length; i++) {
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
			for(int k = 0; k < counts.length; k++) {
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
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < counts.length; i++) {
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
			for(int k = 0; k < counts.length; k++) {
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
	public double sumSqWithReference(int[] counts, double[] reference) {
		final int nCol = reference.length;
		final int numVals = counts.length;
		double ret = 0;

		if(_data.isInSparseFormat()) {
			double ref = 0;
			for(int i = 0; i < nCol; i++)
				ref += reference[i] * reference[i];
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
		else {
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
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		final MatrixBlock retBlock = _data.slice(0, _data.getNumRows() - 1, idxStart, idxEnd - 1);
		if(retBlock.isEmpty())
			return null;
		return new MatrixBlockDictionary(retBlock);
	}

	@Override
	public boolean containsValue(double pattern) {
		return _data.containsValue(pattern);
	}

	@Override
	public boolean containsValueWithReference(double pattern, double[] reference) {
		if(_data.isInSparseFormat()) {
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

		long nnz = 0;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			for(int i = 0; i < counts.length; i++)
				if(!sb.isEmpty(i))
					nnz += sb.size(i) * counts[i];
		}
		else {
			double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int i = 0; i < counts.length; i++) {
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
	public long getNumberNonZerosWithReference(int[] counts, double[] reference, int nRows) {
		long nnz = 0;
		if(_data.isInSparseFormat()) {
			SparseBlock sb = _data.getSparseBlock();
			long emptyRowNNZ = nnz;
			for(int i = 0; i < counts.length; i++) {
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
			final double[] values = _data.getDenseBlockValues();
			int off = 0;
			for(int i = 0; i < counts.length; i++) {
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
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol) {
		if(_data.isInSparseFormat())
			addToEntrySparse(_data.getSparseBlock(), v, fr, to * nCol, nCol);
		else
			addToEntryDense(_data.getDenseBlockValues(), v, fr * nCol, to * nCol, nCol);
	}

	private static final void addToEntrySparse(final SparseBlock sb, final double[] v, final int fr, final int st,
		final int nCol) {
		if(sb.isEmpty(fr))
			return;
		final int apos = sb.pos(fr);
		final int alen = sb.size(fr) + apos;
		final int[] aix = sb.indexes(fr);
		final double[] avals = sb.values(fr);
		for(int j = apos; j < alen; j++)
			v[st + aix[j]] += avals[j];
	}

	private static final void addToEntryDense(final double[] thisV, final double[] v, final int sf, final int st,
		final int nCol) {
		for(int i = sf, j = st; i < sf + nCol; i++, j++)
			v[j] += thisV[i];
	}

	@Override
	public void addToEntry(final double[] v, final int fr, final int to, final int nCol, int rep) {
		if(_data.isInSparseFormat())
			addToEntrySparse(_data.getSparseBlock(), v, fr, to * nCol, nCol, rep);
		else
			addToEntryDense(_data.getDenseBlockValues(), v, fr * nCol, to * nCol, nCol, rep);
	}

	private static final void addToEntrySparse(final SparseBlock sb, final double[] v, final int fr, final int st,
		final int nCol, final int rep) {
		if(sb.isEmpty(fr))
			return;
		final int apos = sb.pos(fr);
		final int alen = sb.size(fr) + apos;
		final int[] aix = sb.indexes(fr);
		final double[] avals = sb.values(fr);
		for(int j = apos; j < alen; j++)
			v[st + aix[j]] += avals[j] * rep;
	}

	private static final void addToEntrySparseCSR(final SparseBlockCSR sb, final double[] v, final int fr, final int st,
		final int nCol, final int[] aix, final double[] avals) {

		final int apos = sb.pos(fr);
		final int alen = sb.size(fr) + apos;
		for(int j = apos; j < alen; j++)
			v[st + aix[j]] += avals[j];
	}

	private static final void addToEntryDense(final double[] thisV, final double[] v, final int sf, final int st,
		final int nCol, final int rep) {
		for(int i = sf, j = st; i < sf + nCol; i++, j++)
			v[j] += thisV[i] * rep;
	}

	@Override
	public void addToEntryVectorized(double[] v, int f1, int f2, int f3, int f4, int f5, int f6, int f7, int f8, int t1,
		int t2, int t3, int t4, int t5, int t6, int t7, int t8, int nCol) {
		if(_data.isInSparseFormat()) {
			final SparseBlock sb = _data.getSparseBlock();
			if(sb instanceof SparseBlockCSR) {
				final SparseBlockCSR csr = (SparseBlockCSR) sb;
				final int[] aix = csr.indexes();
				final double[] avals = csr.values();
				addToEntrySparseCSR(csr, v, f1, t1 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f2, t2 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f3, t3 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f4, t4 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f5, t5 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f6, t6 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f7, t7 * nCol, nCol, aix, avals);
				addToEntrySparseCSR(csr, v, f8, t8 * nCol, nCol, aix, avals);
			}
			else {
				addToEntrySparse(sb, v, f1, t1 * nCol, nCol);
				addToEntrySparse(sb, v, f2, t2 * nCol, nCol);
				addToEntrySparse(sb, v, f3, t3 * nCol, nCol);
				addToEntrySparse(sb, v, f4, t4 * nCol, nCol);
				addToEntrySparse(sb, v, f5, t5 * nCol, nCol);
				addToEntrySparse(sb, v, f6, t6 * nCol, nCol);
				addToEntrySparse(sb, v, f7, t7 * nCol, nCol);
				addToEntrySparse(sb, v, f8, t8 * nCol, nCol);
			}
		}
		else {
			final double[] thisV = _data.getDenseBlockValues();
			addToEntryDense(thisV, v, f1 * nCol, t1 * nCol, nCol);
			addToEntryDense(thisV, v, f2 * nCol, t2 * nCol, nCol);
			addToEntryDense(thisV, v, f3 * nCol, t3 * nCol, nCol);
			addToEntryDense(thisV, v, f4 * nCol, t4 * nCol, nCol);
			addToEntryDense(thisV, v, f5 * nCol, t5 * nCol, nCol);
			addToEntryDense(thisV, v, f6 * nCol, t6 * nCol, nCol);
			addToEntryDense(thisV, v, f7 * nCol, t7 * nCol, nCol);
			addToEntryDense(thisV, v, f8 * nCol, t8 * nCol, nCol);
		}
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		MatrixBlock v = new MatrixBlock(1, tuple.length, tuple);
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		MatrixBlock ret = _data.binaryOperations(op, v, null);
		if(ret.isEmpty())
			return null;
		return new MatrixBlockDictionary(ret);
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
		if(_data.isInSparseFormat()) {
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
		if(_data.isInSparseFormat()) {
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
		if(dictM.isEmpty())
			return null;
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
	public ADictionary replaceWithReference(double pattern, double replace, double[] reference) {
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
	public void product(double[] ret, int[] counts, int nCol) {
		if(_data.isInSparseFormat())
			ret[0] = 0; // if we are sparse there is a zero
		else if(_data.getNonZeros() < _data.getNumColumns() * _data.getNumRows())
			ret[0] = 0; // if the number of zeros are not equal number of cells.
		else {
			final MathContext cont = MathContext.DECIMAL128;
			final int nRow = _data.getNumRows();
			final double[] values = _data.getDenseBlockValues();
			BigDecimal tmp = BigDecimal.ONE;
			int off = 0;
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					final double v = values[off++];
					tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
				}
			}
			if(Math.abs(tmp.doubleValue()) == 0)
				ret[0] = 0;
			else if(!Double.isInfinite(ret[0]))
				ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();
		}
	}

	@Override
	public void productWithDefault(double[] ret, int[] counts, double[] def, int defCount) {
		if(_data.isInSparseFormat())
			ret[0] = 0; // if we are sparse there is a zero
		else if(_data.getNonZeros() < _data.getNumColumns() * _data.getNumRows())
			ret[0] = 0; // if the number of zeros are not equal number of cells.
		else {
			final MathContext cont = MathContext.DECIMAL128;
			final int nRow = _data.getNumRows();
			final int nCol = def.length;
			final double[] values = _data.getDenseBlockValues();
			BigDecimal tmp = BigDecimal.ONE;
			int off = 0;
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; j < nCol; j++) {
					final double v = values[off++];
					tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
				}
			}
			for(int x = 0; x < def.length; x++)
				tmp = tmp.multiply(new BigDecimal(def[x]).pow(defCount, cont), cont);
			if(Math.abs(tmp.doubleValue()) == 0)
				ret[0] = 0;
			else if(!Double.isInfinite(ret[0]))
				ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();
		}
	}

	@Override
	public void productWithReference(double[] ret, int[] counts, double[] reference, int refCount) {
		final MathContext cont = MathContext.DECIMAL128;
		final int nCol = _data.getNumColumns();
		final int nRow = _data.getNumRows();
		// force dense ... if this ever is a bottleneck i will be surprised
		_data.sparseToDense();
		final double[] values = _data.getDenseBlockValues();
		BigDecimal tmp = BigDecimal.ONE;
		int off = 0;
		for(int i = 0; i < nRow; i++) {
			for(int j = 0; j < nCol; j++) {
				final double v = values[off++] + reference[j];
				if(v == 0) {
					ret[0] = 0;
					return;
				}
				tmp = tmp.multiply(new BigDecimal(v).pow(counts[i], cont), cont);
			}
		}
		for(int x = 0; x < reference.length; x++)
			tmp = tmp.multiply(new BigDecimal(reference[x]).pow(refCount, cont), cont);
		if(Math.abs(tmp.doubleValue()) == 0)
			ret[0] = 0;
		else if(!Double.isInfinite(ret[0]))
			ret[0] = new BigDecimal(ret[0]).multiply(tmp, MathContext.DECIMAL128).doubleValue();
	}

	@Override
	public CM_COV_Object centralMoment(CM_COV_Object ret, ValueFunction fn, int[] counts, int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		if(_data.isInSparseFormat())
			throw new DMLCompressionException("The dictionary should not be sparse with one column");
		double[] vals = _data.getDenseBlockValues();
		for(int i = 0; i < vals.length; i++)
			fn.execute(ret, vals[i], counts[i]);
		if(ret.getWeight() < nRows)
			fn.execute(ret, 0, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CM_COV_Object centralMomentWithDefault(CM_COV_Object ret, ValueFunction fn, int[] counts, double def,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		if(_data.isInSparseFormat())
			throw new DMLCompressionException("The dictionary should not be sparse with one column");
		double[] vals = _data.getDenseBlockValues();
		for(int i = 0; i < vals.length; i++)
			fn.execute(ret, vals[i], counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, def, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public CM_COV_Object centralMomentWithReference(CM_COV_Object ret, ValueFunction fn, int[] counts, double reference,
		int nRows) {
		// should be guaranteed to only contain one value per tuple in dictionary.
		if(_data.isInSparseFormat())
			throw new DMLCompressionException("The dictionary should not be sparse with one column");
		double[] vals = _data.getDenseBlockValues();
		for(int i = 0; i < vals.length; i++)
			fn.execute(ret, vals[i] + reference, counts[i]);

		if(ret.getWeight() < nRows)
			fn.execute(ret, reference, nRows - ret.getWeight());
		return ret;
	}

	@Override
	public ADictionary rexpandCols(int max, boolean ignore, boolean cast, int nCol) {
		MatrixBlock ex = LibMatrixReorg.rexpand(_data, new MatrixBlock(), max, false, cast, ignore, 1);
		if(ex.isEmpty())
			return null;
		else
			return new MatrixBlockDictionary(ex);
	}

	@Override
	public ADictionary rexpandColsWithReference(int max, boolean ignore, boolean cast, int reference) {
		ADictionary a = applyScalarOp(new LeftScalarOperator(Plus.getPlusFnObject(), reference));
		if(a == null)
			return null;
		else
			return a.rexpandCols(max, ignore, cast, 1);
	}

	@Override
	public double getSparsity() {
		return _data.getSparsity();
	}

	@Override
	public void multiplyScalar(double v, double[] ret, int off, int dictIdx, int[] cols) {
		if(_data.isInSparseFormat())
			multiplyScalarSparse(v, ret, off, dictIdx, cols);
		else
			multiplyScalarDense(v, ret, off, dictIdx, cols);
	}

	private void multiplyScalarSparse(double v, double[] ret, int off, int dictIdx, int[] cols) {
		final SparseBlock sb = _data.getSparseBlock();
		if(sb.isEmpty(dictIdx))
			return;
		final int apos = sb.pos(dictIdx);
		final int alen = sb.size(dictIdx) + apos;
		final int[] aix = sb.indexes(dictIdx);
		final double[] aval = sb.values(dictIdx);
		for(int i = apos; i < alen; i++)
			ret[off + cols[aix[i]]] += v * aval[i];
	}

	private void multiplyScalarDense(double v, double[] ret, int off, int dictIdx, int[] cols) {
		final double[] dV = _data.getDenseBlockValues();
		final int offD = dictIdx * cols.length;
		for(int i = 0; i < cols.length; i++)
			ret[off + cols[i]] += v * dV[offD + i];
	}

	@Override
	protected void TSMMWithScaling(int[] counts, int[] rows, int[] cols, MatrixBlock ret) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.TSMMDictsSparseWithScaling(_data.getSparseBlock(), rows, cols, counts, ret);
		else
			DictLibMatrixMult.TSMMDictsDenseWithScaling(_data.getDenseBlockValues(), rows, cols, counts, ret);
	}

	@Override
	protected void MMDict(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		if(_data.isInSparseFormat())
			right.MMDictSparse(_data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			right.MMDictDense(_data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void MMDictDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.MMDictsDenseSparse(left, _data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			DictLibMatrixMult.MMDictsDenseDense(left, _data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void MMDictSparse(SparseBlock left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {

		if(_data.isInSparseFormat())
			DictLibMatrixMult.MMDictsSparseSparse(left, _data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			DictLibMatrixMult.MMDictsSparseDense(left, _data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangle(ADictionary right, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		if(_data.isInSparseFormat())
			right.TSMMToUpperTriangleSparse(_data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			right.TSMMToUpperTriangleDense(_data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleDense(double[] left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.MMToUpperTriangleDenseSparse(left, _data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			DictLibMatrixMult.MMToUpperTriangleDenseDense(left, _data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparse(SparseBlock left, int[] rowsLeft, int[] colsRight, MatrixBlock result) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.MMToUpperTriangleSparseSparse(left, _data.getSparseBlock(), rowsLeft, colsRight, result);
		else
			DictLibMatrixMult.MMToUpperTriangleSparseDense(left, _data.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	@Override
	protected void TSMMToUpperTriangleScaling(ADictionary right, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		if(_data.isInSparseFormat())
			right.TSMMToUpperTriangleSparseScaling(_data.getSparseBlock(), rowsLeft, colsRight, scale, result);
		else
			right.TSMMToUpperTriangleDenseScaling(_data.getDenseBlockValues(), rowsLeft, colsRight, scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleDenseScaling(double[] left, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.TSMMToUpperTriangleDenseSparseScaling(left, _data.getSparseBlock(), rowsLeft, colsRight,
				scale, result);
		else
			DictLibMatrixMult.TSMMToUpperTriangleDenseDenseScaling(left, _data.getDenseBlockValues(), rowsLeft, colsRight,
				scale, result);
	}

	@Override
	protected void TSMMToUpperTriangleSparseScaling(SparseBlock left, int[] rowsLeft, int[] colsRight, int[] scale,
		MatrixBlock result) {
		if(_data.isInSparseFormat())
			DictLibMatrixMult.TSMMToUpperTriangleSparseSparseScaling(left, _data.getSparseBlock(), rowsLeft, colsRight,
				scale, result);
		else
			DictLibMatrixMult.TSMMToUpperTriangleSparseDenseScaling(left, _data.getDenseBlockValues(), rowsLeft, colsRight,
				scale, result);
	}

}
