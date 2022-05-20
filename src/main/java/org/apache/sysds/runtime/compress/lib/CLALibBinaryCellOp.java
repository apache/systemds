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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibBinaryCellOp {

	private static final Log LOG = LogFactory.getLog(CLALibBinaryCellOp.class.getName());

	public static MatrixBlock binaryOperationsRight(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that,
		MatrixBlock result) {

		if(that.getNumRows() == 1 && that.getNumColumns() == 1) {
			ScalarOperator sop = new RightScalarOperator(op.fn, that.getValue(0, 0), op.getNumThreads());
			return CLALibScalar.scalarOperations(sop, m1, result);
		}
		if(that.isEmpty())
			return binaryOperationsEmpty(op, m1, that, result);

		LibMatrixBincell.isValidDimensionsBinaryExtended(m1, that);

		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(m1, that);
		if(that instanceof CompressedMatrixBlock && that.getInMemorySize() < m1.getInMemorySize()) {
			MatrixBlock m1uc = CompressedMatrixBlock.getUncompressed(m1, "Decompressing left side in BinaryOps");
			return selectProcessingBasedOnAccessType(op, (CompressedMatrixBlock) that, m1uc, result, atype, true);
		}
		else {
			that = CompressedMatrixBlock.getUncompressed(that, "Decompressing right side in BinaryOps");
			return selectProcessingBasedOnAccessType(op, m1, that, result, atype, false);
		}
	}

	public static MatrixBlock binaryOperationsLeft(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that,
		MatrixBlock result) {
		if(that.getNumRows() == 1 && that.getNumColumns() == 1) {
			ScalarOperator sop = new LeftScalarOperator(op.fn, that.getValue(0, 0), op.getNumThreads());
			return CLALibScalar.scalarOperations(sop, m1, result);
		}
		if(that.isEmpty())
			throw new NotImplementedException("Not handling left empty yet");

		that = CompressedMatrixBlock.getUncompressed(that, "Decompressing left side in BinaryOps");
		LibMatrixBincell.isValidDimensionsBinaryExtended(that, m1);
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(that, m1);
		return selectProcessingBasedOnAccessType(op, m1, that, result, atype, true);

	}

	private static MatrixBlock binaryOperationsEmpty(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that,
		MatrixBlock result) {
		final int m1Col = m1.getNumColumns();
		final int m1Row = m1.getNumRows();

		final ValueFunction fn = op.fn;
		if(fn instanceof Multiply)
			result = CompressedMatrixBlockFactory.createConstant(m1Row, m1Col, 0);
		else if(fn instanceof Minus1Multiply)
			result = CompressedMatrixBlockFactory.createConstant(m1Row, m1Col, 1);
		else if(fn instanceof Minus || fn instanceof Plus || fn instanceof MinusMultiply || fn instanceof PlusMultiply) {
			CompressedMatrixBlock ret = new CompressedMatrixBlock();
			ret.copy(m1);
			return ret;
		}
		else
			throw new NotImplementedException("Function Type: " + fn);
		return result;
	}

	private static MatrixBlock selectProcessingBasedOnAccessType(BinaryOperator op, CompressedMatrixBlock m1,
		MatrixBlock that, MatrixBlock result, BinaryAccessType atype, boolean left) {

		if(atype == BinaryAccessType.MATRIX_COL_VECTOR || atype == BinaryAccessType.COL_VECTOR_MATRIX) {
			// Column vector access
			MatrixBlock d_compressed = m1.getCachedDecompressed();
			if(d_compressed != null) {
				if(left && atype == BinaryAccessType.COL_VECTOR_MATRIX)
					throw new NotImplementedException("Binary row op left is not supported for Uncompressed Matrix, "
						+ "Implement support for VMr in MatrixBlock Binary Cell operations");
				if(left)
					return that.binaryOperations(op, d_compressed);
				else
					return d_compressed.binaryOperations(op, that);
			}
			return binaryMVCol(m1, that, op, left);
		}
		else if(atype == BinaryAccessType.MATRIX_MATRIX) {
			// Full matrix access.
			MatrixBlock d_compressed = m1.getCachedDecompressed();// m1.getUncompressed("MatrixMatrix " + op);
			if(d_compressed != null) {
				if(left)
					return that.binaryOperations(op, d_compressed);
				else
					return d_compressed.binaryOperations(op, that);
			}
			return binaryMM(m1, that, op, left);
		}
		else if(isSupportedBinaryCellOp(op.fn) && atype == BinaryAccessType.MATRIX_ROW_VECTOR ||
			atype == BinaryAccessType.ROW_VECTOR_MATRIX)
			// Row matrix access.
			return rowBinCellOp(m1, that, result, op, left);
		else
			// All other, fallback to default execution.
			return CompressedMatrixBlock.getUncompressed(m1, "BinaryOp: " + op.fn).binaryOperations(op, that, result);

	}

	private static boolean isSupportedBinaryCellOp(ValueFunction fn) {
		return fn instanceof Multiply || fn instanceof Divide || fn instanceof Plus || fn instanceof Minus ||
			fn instanceof MinusMultiply || fn instanceof PlusMultiply;
	}

	private static CompressedMatrixBlock setupCompressedReturnMatrixBlock(CompressedMatrixBlock m1, MatrixValue result) {
		CompressedMatrixBlock ret = null;
		if(result == null || !(result instanceof CompressedMatrixBlock))
			ret = new CompressedMatrixBlock(m1.getNumRows(), m1.getNumColumns());
		else {
			ret = (CompressedMatrixBlock) result;
			ret.reset(m1.getNumRows(), m1.getNumColumns());
		}
		return ret;
	}

	private static MatrixBlock rowBinCellOp(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		boolean left) {
		CompressedMatrixBlock cRet = setupCompressedReturnMatrixBlock(m1, ret);
		if(isValidForOverlappingBinaryCellOperations(m1, op))
			overlappingBinaryCellOp(m1, m2, cRet, op, left);
		else
			nonOverlappingBinaryCellOp(m1, m2, cRet, op, left);
		cRet.recomputeNonZeros();
		return cRet;
	}

	private static void nonOverlappingBinaryCellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) {

		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, m2);
		switch(atype) {
			case MATRIX_ROW_VECTOR:
				// Verify if it is okay to include all OuterVectorVector ops here.
				binaryMVRow(m1, m2, ret, op, left);
				return;
			default:
				LOG.warn("Inefficient Decompression for " + op + "  " + atype);
				m1.decompress().binaryOperations(op, m2, ret);
		}
	}

	private static boolean isValidForOverlappingBinaryCellOperations(CompressedMatrixBlock m1, BinaryOperator op) {
		return m1.isOverlapping() && (op.fn instanceof Plus || op.fn instanceof Minus);
	}

	private static void overlappingBinaryCellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) {
		binaryMVPlusStack(m1, m2, ret, op, left);
	}

	private static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, double[] v, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) {

		final List<AColGroup> oldColGroups = m1.getColGroups();

		final int k = op.getNumThreads();
		final List<AColGroup> newColGroups = new ArrayList<>(oldColGroups.size());
		final boolean isRowSafe = left ? op.isRowSafeLeft(v) : op.isRowSafeRight(v);

		if(k <= 1)
			binaryMVRowSingleThread(oldColGroups, v, op, left, newColGroups, isRowSafe);
		else
			binaryMVRowMultiThread(oldColGroups, v, op, left, newColGroups, isRowSafe, k);

		ret.allocateColGroupList(newColGroups);
		ret.setNonZeros(m1.getNumColumns() * m1.getNumRows());
		return ret;
	}

	private static void binaryMVRowSingleThread(List<AColGroup> oldColGroups, double[] v, BinaryOperator op,
		boolean left, List<AColGroup> newColGroups, boolean isRowSafe) {
		if(left)
			for(AColGroup grp : oldColGroups)
				newColGroups.add(grp.binaryRowOpLeft(op, v, isRowSafe));
		else
			for(AColGroup grp : oldColGroups)
				newColGroups.add(grp.binaryRowOpRight(op, v, isRowSafe));
	}

	private static void binaryMVRowMultiThread(List<AColGroup> oldColGroups, double[] v, BinaryOperator op, boolean left,
		List<AColGroup> newColGroups, boolean isRowSafe, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		final ArrayList<BinaryMVRowTask> tasks = new ArrayList<>();
		try {
			if(left)
				for(AColGroup grp : oldColGroups)
					tasks.add(new BinaryMVRowTaskLeft(grp, v, op, isRowSafe));
			else
				for(AColGroup grp : oldColGroups)
					tasks.add(new BinaryMVRowTaskRight(grp, v, op, isRowSafe));

			for(Future<AColGroup> f : pool.invokeAll(tasks))
				newColGroups.add(f.get());

		}
		catch(InterruptedException | ExecutionException e) {
			pool.shutdown();
			throw new DMLRuntimeException(e);
		}
		pool.shutdown();
	}

	private static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) {
		return binaryMVRow(m1, forceMatrixBlockToDense(m2), ret, op, left);
	}

	private static double[] forceMatrixBlockToDense(MatrixBlock m2) {
		double[] v;
		if(m2.isInSparseFormat()) {
			SparseBlock sb = m2.getSparseBlock();
			if(sb == null) {
				throw new DMLRuntimeException("Unknown matrix block type");
			}
			else {
				double[] spV = sb.values(0);
				int[] spI = sb.indexes(0);
				v = new double[m2.getNumColumns()];
				for(int i = sb.pos(0); i < sb.size(0); i++) {
					v[spI[i]] = spV[i];
				}
			}
		}
		else
			v = m2.getDenseBlockValues();

		return v;

	}

	protected static CompressedMatrixBlock binaryMVPlusStack(CompressedMatrixBlock m1, MatrixBlock m2,
		CompressedMatrixBlock ret, BinaryOperator op, boolean left) {
		final List<AColGroup> oldColGroups = m1.getColGroups();
		final int size = oldColGroups.size();
		final List<AColGroup> newColGroups = new ArrayList<>(size);
		final int nCol = m1.getNumColumns();

		// find the smallest colgroup to modify.
		int smallestIndex = 0;
		int smallestSize = Integer.MAX_VALUE;
		for(int i = 0; i < size; i++) {
			final AColGroup g = oldColGroups.get(i);
			final int newSize = g.getNumValues();
			newColGroups.add(g);
			if(newSize < smallestSize && g.getNumCols() == nCol && !(g instanceof ASDCZero)) {
				smallestIndex = i;
				smallestSize = newSize;
			}
		}

		// apply overlap
		if(smallestSize == Integer.MAX_VALUE) {
			// if there was no smallest colgroup
			ADictionary newDict = new MatrixBlockDictionary(m2);
			newColGroups.add(ColGroupConst.create(nCol, newDict));
		}
		else {
			// apply to the found group
			final double[] row = m2.getDenseBlockValues();
			AColGroup g;
			if(left)
				g = newColGroups.get(smallestIndex).binaryRowOpLeft(op, row, op.isRowSafeLeft(row));
			else
				g = newColGroups.get(smallestIndex).binaryRowOpRight(op, row, op.isRowSafeRight(row));

			newColGroups.set(smallestIndex, g);
		}

		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(true);
		ret.setNonZeros(-1);
		return ret;
	}

	private static MatrixBlock binaryMVCol(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left) {

		final int nCols = m1.getNumColumns();
		final int nRows = m1.getNumRows();
		m1 = morph(m1);

		MatrixBlock ret = new MatrixBlock(nRows, nCols, false, -1).allocateBlock();

		final int k = op.getNumThreads();
		long nnz = 0;

		if(k <= 1)
			nnz = binaryMVColSingleThread(m1, m2, op, left, ret);
		else
			nnz = binaryMVColMultiThread(m1, m2, op, left, ret);

		ret.setNonZeros(nnz);
		ret.examSparsity();
		return ret;
	}

	private static long binaryMVColSingleThread(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left, MatrixBlock ret) {
		final int nRows = m1.getNumRows();
		long nnz = 0;
		if(left)
			nnz += new BinaryMVColLeftTask(m1, m2, ret, 0, nRows, op).call();
		else
			nnz += new BinaryMVColTask(m1, m2, ret, 0, nRows, op).call();
		return nnz;
	}

	private static long binaryMVColMultiThread(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left,
		MatrixBlock ret) {
		final int nRows = m1.getNumRows();
		final int k = op.getNumThreads();
		final int blkz = ret.getNumRows() / k;
		long nnz = 0;
		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		final ArrayList<Callable<Integer>> tasks = new ArrayList<>();
		try {
			for(int i = 0; i < nRows; i += blkz) {
				if(left)
					tasks.add(new BinaryMVColLeftTask(m1, m2, ret, i, Math.min(nRows, i + blkz), op));
				else
					tasks.add(new BinaryMVColTask(m1, m2, ret, i, Math.min(nRows, i + blkz), op));
			}
			for(Future<Integer> f : pool.invokeAll(tasks))
				nnz += f.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
		return nnz;
	}

	private static MatrixBlock binaryMM(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left) {
		final int nCols = m1.getNumColumns();
		final int nRows = m1.getNumRows();
		m1 = morph(m1);

		MatrixBlock ret = new MatrixBlock(nRows, nCols, false, -1).allocateBlock();

		// final int k = op.getNumThreads();
		long nnz = binaryMMMultiThread(m1, m2, op, left, ret);

		ret.setNonZeros(nnz);
		ret.examSparsity();
		return ret;
	}

	private static long binaryMMMultiThread(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left,
		MatrixBlock ret) {
		final int nRows = m1.getNumRows();
		final int k = op.getNumThreads();
		final int blkz = ret.getNumRows() / k;
		long nnz = 0;
		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		final ArrayList<Callable<Long>> tasks = new ArrayList<>();
		try {
			for(int i = 0; i < nRows; i += blkz)
				tasks.add(new BinaryMMTask(m1, m2, ret, i, Math.min(nRows, i + blkz), op, left));

			for(Future<Long> f : pool.invokeAll(tasks))
				nnz += f.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException(e);
		}
		return nnz;
	}

	private static CompressedMatrixBlock morph(CompressedMatrixBlock m) {
		final List<AColGroup> groups = m.getColGroups();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			CompressedMatrixBlock mf1 = new CompressedMatrixBlock(m);
			final int nCols = m.getNumColumns();
			double[] constV = new double[nCols];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);
			filteredGroups.add(ColGroupConst.create(constV));
			mf1.allocateColGroupList(filteredGroups);
			return mf1;
		}
		else
			return m;
	}

	private static class BinaryMVColTask implements Callable<Integer> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		protected BinaryMVColTask(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Integer call() {
			final int _blklen = 32768 / _ret.getNumColumns();
			final List<AColGroup> groups = _m1.getColGroups();

			final AIterator[] its = new AIterator[groups.size()];

			for(int i = 0; i < groups.size(); i++)
				if(groups.get(i) instanceof ASDCZero)
					its[i] = ((ASDCZero) groups.get(i)).getIterator(_rl);

			for(int r = _rl; r < _ru; r += _blklen)
				processBlock(r, Math.min(r + _blklen, _ru), groups, its);

			return _ret.getNumColumns() * _ret.getNumRows();
		}

		private final void processBlock(final int rl, final int ru, final List<AColGroup> groups, final AIterator[] its) {
			// unsafe decompress, since we count nonzeros afterwards.
			final DenseBlock db = _ret.getDenseBlock();
			for(int i = 0; i < groups.size(); i++) {
				final AColGroup g = groups.get(i);
				// AColGroup g = _groups.get(i);
				if(g instanceof ASDCZero)
					((ASDCZero) g).decompressToDenseBlock(db, rl, ru, 0, 0, its[i]);
				else
					g.decompressToDenseBlock(db, rl, ru, 0, 0);
			}

			if(_m2.isInSparseFormat())
				throw new NotImplementedException("Not Implemented sparse Format execution for MM.");
			else {
				int offset = rl * _m1.getNumColumns();
				double[] _retDense = _ret.getDenseBlockValues();
				double[] _m2Dense = _m2.getDenseBlockValues();
				for(int row = rl; row < ru; row++) {
					double vr = _m2Dense[row];
					for(int col = 0; col < _m1.getNumColumns(); col++) {
						double v = _op.fn.execute(_retDense[offset], vr);
						_retDense[offset] = v;
						offset++;
					}
				}

			}
		}
	}

	private static class BinaryMMTask implements Callable<Long> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final boolean _left;
		private final BinaryOperator _op;

		protected BinaryMMTask(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op, boolean left) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
			_left = left;
		}

		@Override
		public Long call() {
			final List<AColGroup> groups = _m1.getColGroups();
			final int _blklen = Math.max(65536 * 2 / _ret.getNumColumns() / groups.size(), 64);

			final AIterator[] its = new AIterator[groups.size()];

			for(int i = 0; i < groups.size(); i++)
				if(groups.get(i) instanceof ASDCZero)
					its[i] = ((ASDCZero) groups.get(i)).getIterator(_rl);

			long nnz = 0;
			for(int r = _rl; r < _ru; r += _blklen) {
				final int re = Math.min(r + _blklen, _ru);
				processBlock(r, re, groups, its);
				nnz += _ret.recomputeNonZeros(r, re - 1);
			}

			return nnz;
		}

		private final void processBlock(final int rl, final int ru, final List<AColGroup> groups, final AIterator[] its) {
			// unsafe decompress, since we count nonzeros afterwards.
			final DenseBlock db = _ret.getDenseBlock();
			for(int i = 0; i < groups.size(); i++) {
				final AColGroup g = groups.get(i);
				// AColGroup g = _groups.get(i);
				if(g instanceof ASDCZero)
					((ASDCZero) g).decompressToDenseBlock(db, rl, ru, 0, 0, its[i]);
				else
					g.decompressToDenseBlock(db, rl, ru, 0, 0);
			}

			final DenseBlock rv = _ret.getDenseBlock();
			final int cols = _ret.getNumColumns();
			if(_left) {
				// all exec should have ret on right side
				if(_m2.isInSparseFormat()) {
					final SparseBlock m2sb = _m2.getSparseBlock();
					for(int r = rl; r < ru; r++) {
						final double[] retV = rv.values(r);
						int off = rv.pos(r);
						if(m2sb.isEmpty(r)) {
							for(int c = off; c < cols + off; c++)
								retV[c] = _op.fn.execute(retV[c], 0);
						}
						else {
							final int apos = m2sb.pos(r);
							final int alen = m2sb.size(r) + apos;
							final int[] aix = m2sb.indexes(r);
							final double[] avals = m2sb.values(r);
							int j = 0;
							for(int k = apos; j < cols && k < alen; j++, off++) {
								final double v = aix[k] == j ? avals[k++] : 0;
								retV[off] = _op.fn.execute(v, retV[off]);
							}

							for(; j < cols; j++)
								retV[off] = _op.fn.execute(0, retV[off]);
						}
					}
				}
				else {
					DenseBlock m2db = _m2.getDenseBlock();
					for(int r = rl; r < ru; r++) {
						double[] retV = rv.values(r);
						double[] m2V = m2db.values(r);

						int off = rv.pos(r);
						for(int c = off; c < cols + off; c++)
							retV[c] = _op.fn.execute(m2V[c], retV[c]);
					}
				}
			}
			else {
				// all exec should have ret on left side
				if(_m2.isInSparseFormat()) {
					final SparseBlock m2sb = _m2.getSparseBlock();
					for(int r = rl; r < ru; r++) {
						final double[] retV = rv.values(r);
						int off = rv.pos(r);
						if(m2sb.isEmpty(r)) {
							for(int c = off; c < cols + off; c++)
								retV[c] = _op.fn.execute(retV[c], 0);
						}
						else {
							final int apos = m2sb.pos(r);
							final int alen = m2sb.size(r) + apos;
							final int[] aix = m2sb.indexes(r);
							final double[] avals = m2sb.values(r);
							int j = 0;
							for(int k = apos; j < cols && k < alen; j++, off++) {
								final double v = aix[k] == j ? avals[k++] : 0;
								retV[off] = _op.fn.execute(retV[off], v);
							}

							for(; j < cols; j++)
								retV[off] = _op.fn.execute(retV[off], 0);
						}
					}
				}
				else {
					final DenseBlock m2db = _m2.getDenseBlock();
					for(int r = rl; r < ru; r++) {
						final double[] retV = rv.values(r);
						final double[] m2V = m2db.values(r);

						int off = rv.pos(r);
						for(int c = off; c < cols + off; c++)
							retV[c] = _op.fn.execute(retV[c], m2V[c]);
					}
				}
			}
		}
	}

	private static class BinaryMVColLeftTask implements Callable<Integer> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		protected BinaryMVColLeftTask(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Integer call() {
			// unsafe decompress, since we count nonzeros afterwards.
			for(AColGroup g : _m1.getColGroups())
				g.decompressToDenseBlock(_ret.getDenseBlock(), _rl, _ru);

			if(_m2.isInSparseFormat())
				throw new NotImplementedException("Not Implemented sparse Format execution for MM.");
			else {
				int offset = _rl * _m1.getNumColumns();
				double[] _retDense = _ret.getDenseBlockValues();
				double[] _m2Dense = _m2.getDenseBlockValues();
				for(int row = _rl; row < _ru; row++) {
					double vr = _m2Dense[row];
					for(int col = 0; col < _m1.getNumColumns(); col++) {
						double v = _op.fn.execute(vr, _retDense[offset]);
						_retDense[offset] = v;
						offset++;
					}
				}

				return _ret.getNumColumns() * _ret.getNumRows();
			}
		}
	}

	private static abstract class BinaryMVRowTask implements Callable<AColGroup> {
		protected final AColGroup _group;
		protected final double[] _v;
		protected final BinaryOperator _op;
		protected final boolean _isRowSafe;

		protected BinaryMVRowTask(AColGroup group, double[] v, BinaryOperator op, boolean isRowSafe) {
			_group = group;
			_v = v;
			_op = op;
			_isRowSafe = isRowSafe;
		}
	}

	private static class BinaryMVRowTaskLeft extends BinaryMVRowTask {
		protected BinaryMVRowTaskLeft(AColGroup group, double[] v, BinaryOperator op, boolean isRowSafe) {
			super(group, v, op, isRowSafe);
		}

		@Override
		public AColGroup call() {
			return _group.binaryRowOpLeft(_op, _v, _isRowSafe);
		}
	}

	private static class BinaryMVRowTaskRight extends BinaryMVRowTask {
		protected BinaryMVRowTaskRight(AColGroup group, double[] v, BinaryOperator op, boolean isRowSafe) {
			super(group, v, op, isRowSafe);
		}

		@Override
		public AColGroup call() {
			return _group.binaryRowOpRight(_op, _v, _isRowSafe);
		}
	}
}
