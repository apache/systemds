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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.ValueComparisonFunction;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.apache.sysds.utils.stats.Timing;

public final class CLALibBinaryCellOp {
	private static final Log LOG = LogFactory.getLog(CLALibBinaryCellOp.class.getName());

	private CLALibBinaryCellOp() {
		// empty private constructor.
	}

	public static MatrixBlock binaryOperationsRight(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that) {

		try {
			op = LibMatrixBincell.replaceOpWithSparseSafeIfApplicable(m1, that, op);
			if((that.getNumRows() == 1 && that.getNumColumns() == 1) || that.isEmpty()) {
				ScalarOperator sop = new RightScalarOperator(op.fn, that.get(0, 0), op.getNumThreads());

				return CLALibScalar.scalarOperations(sop, m1, null);
			}
			else
				return binaryOperationsRightFiltered(op, m1, that);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed Right Binary Compressed Operation", e);
		}
	}

	public static MatrixBlock binaryOperationsLeft(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that) {

		try {
			op = LibMatrixBincell.replaceOpWithSparseSafeIfApplicable(m1, that, op);
			if((that.getNumRows() == 1 && that.getNumColumns() == 1) || that.isEmpty()) {
				ScalarOperator sop = new LeftScalarOperator(op.fn, that.get(0, 0), op.getNumThreads());
				return CLALibScalar.scalarOperations(sop, m1, null);
			}
			that = CompressedMatrixBlock.getUncompressed(that, "Decompressing left side in BinaryOps");
			BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(that, m1);
			return selectProcessingBasedOnAccessType(op, m1, that, atype, true);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed Left Binary Compressed Operation", e);
		}
	}

	private static MatrixBlock binaryOperationsRightFiltered(BinaryOperator op, CompressedMatrixBlock m1,
		MatrixBlock that) throws Exception {
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessTypeExtended(m1, that);
		if(that instanceof CompressedMatrixBlock && that.getInMemorySize() < m1.getInMemorySize()) {
			MatrixBlock m1uc = CompressedMatrixBlock.getUncompressed(m1, "Decompressing left side in BinaryOps");
			return selectProcessingBasedOnAccessType(op, (CompressedMatrixBlock) that, m1uc, atype, true);
		}
		else {
			that = CompressedMatrixBlock.getUncompressed(that, "Decompressing right side in BinaryOps");
			return selectProcessingBasedOnAccessType(op, m1, that, atype, false);
		}
	}

	private static MatrixBlock selectProcessingBasedOnAccessType(BinaryOperator op, CompressedMatrixBlock m1,
		MatrixBlock that, BinaryAccessType atype, boolean left) throws Exception {

		switch(atype) {
			case MATRIX_MATRIX:
				return mm(op, m1, that, left);
			case COL_VECTOR_MATRIX:
			case MATRIX_COL_VECTOR:
				return mvCol(op, m1, that, left);
			case MATRIX_ROW_VECTOR:
			case ROW_VECTOR_MATRIX:
				return mvRow(m1, that, op, left);
			case OUTER_VECTOR_VECTOR:
				return CompressedMatrixBlock.getUncompressed(m1, "OVV BinaryOp: " + op.fn).binaryOperations(op, that);
			case INVALID:
			default:
				final int rlen1 = m1.getNumRows();
				final int rlen2 = that.getNumRows();
				final int clen1 = m1.getNumColumns();
				final int clen2 = that.getNumColumns();
				throw new RuntimeException("Block sizes are not matched for binary cell operations: " + rlen1 + "x" + clen1
					+ " vs " + rlen2 + "x" + clen2);
		}
	}

	private static MatrixBlock mm(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that, boolean left)
		throws Exception {
		// Full matrix access.
		MatrixBlock d_compressed = m1.getCachedDecompressed();
		if(d_compressed != null) {
			if(left)
				return that.binaryOperations(op, d_compressed);
			else
				return d_compressed.binaryOperations(op, that);
		}
		return mmCompressed(m1, that, op, left);
	}

	private static MatrixBlock mvCol(BinaryOperator op, CompressedMatrixBlock m1, MatrixBlock that, boolean left)
		throws Exception {
		// Column vector access
		MatrixBlock d_compressed = m1.getCachedDecompressed();
		if(d_compressed != null) {
			if(left)
				throw new NotImplementedException("Binary row op left is not supported for Uncompressed Matrix, "
					+ "Implement support for VMr in MatrixBlock Binary Cell operations");
			// return that.binaryOperations(op, d_compressed);
			else
				return d_compressed.binaryOperations(op, that);
		}
		// make sure that the right side matrix really is dense!
		// it is a hard requirement, however normal that it is since it is a column vector
		that.sparseToDense(op.getNumThreads());
		return mvColCompressed(m1, that, op, left);
	}

	private static MatrixBlock mvRow(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left)
		throws Exception {
		CompressedMatrixBlock cRet = new CompressedMatrixBlock(m1.getNumRows(), m1.getNumColumns());
		if(isValidForOverlappingBinaryCellOperations(m1, op))
			return binaryMVPlusStack(m1, m2, cRet, op, left);
		else if(isSupportedOverlappingBinCell(m1, m2, op.fn))
			return binaryMVRow(m1, m2, cRet, op, left);
		else// TODO add decompress into and apply for row operations.
			return CompressedMatrixBlock.getUncompressed(m1, "BinaryOp: " + op.fn).binaryOperations(op, m2);
	}

	private static boolean isSupportedOverlappingBinCell(CompressedMatrixBlock m1, MatrixBlock m2, ValueFunction fn) {
		return (!m1.isOverlapping() // should not be overlapping
			&& !(fn instanceof Power && m1.getSparsity() < 1.0 && containsNegative(m2))) // and no zeros on power.
		// or the operation is
			|| fn instanceof Multiply //
			|| fn instanceof Divide//

		;
		// || fn instanceof MinusMultiply //
		// || fn instanceof PlusMultiply;
	}

	private static boolean containsNegative(MatrixBlock rm) {
		final int clen = rm.getNumColumns();
		for(int i = 0; i < clen; i++) {
			if(rm.get(0, i) < 0)
				return true;
		}
		return false;
	}

	private static boolean isValidForOverlappingBinaryCellOperations(CompressedMatrixBlock m1, BinaryOperator op) {
		return m1.isOverlapping() && (op.fn instanceof Plus || op.fn instanceof Minus);
	}

	private static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, double[] v, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) throws Exception {
		final List<AColGroup> oldColGroups = m1.getColGroups();

		final int k = op.getNumThreads();
		final List<AColGroup> newColGroups = new ArrayList<>(oldColGroups.size());
		final boolean isRowSafe = left ? op.isRowSafeLeft(v) : op.isRowSafeRight(v);

		if(k <= 1 || oldColGroups.size() <= 1)
			binaryMVRowSingleThread(oldColGroups, v, op, left, newColGroups, isRowSafe);
		else
			binaryMVRowMultiThread(oldColGroups, v, op, left, newColGroups, isRowSafe, k);

		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(m1.isOverlapping());
		ret.examSparsity(op.getNumThreads());
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
		List<AColGroup> newColGroups, boolean isRowSafe, int k) throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<BinaryMVRowTask> tasks = new ArrayList<>();
			if(left)
				for(AColGroup grp : oldColGroups)
					tasks.add(new BinaryMVRowTaskLeft(grp, v, op, isRowSafe));
			else
				for(AColGroup grp : oldColGroups)
					tasks.add(new BinaryMVRowTaskRight(grp, v, op, isRowSafe));

			for(Future<AColGroup> f : pool.invokeAll(tasks))
				newColGroups.add(f.get());

		}
		finally {
			pool.shutdown();
		}
	}

	private static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) throws Exception {
		return binaryMVRow(m1, forceRowToDense(m2), ret, op, left);
	}

	private static double[] forceRowToDense(MatrixBlock m2) {
		final double[] v;
		if(m2.isInSparseFormat()) {
			final SparseBlock sb = m2.getSparseBlock();
			final double[] spV = sb.values(0);
			final int[] spI = sb.indexes(0);
			v = new double[m2.getNumColumns()];
			for(int i = sb.pos(0); i < sb.size(0); i++)
				v[spI[i]] = spV[i];
		}
		else
			v = m2.getDenseBlockValues();

		return v;

	}

	protected static MatrixBlock binaryMVPlusStack(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op, boolean left) {
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
			// select the best!
			if(newSize < smallestSize && g.getNumCols() == nCol && !(g instanceof ASDCZero)) {
				smallestIndex = i;
				smallestSize = newSize;
			}
		}

		if(smallestSize == Integer.MAX_VALUE)
			stackConstGroup(m2, op, left, newColGroups, nCol, m2.getDenseBlockValues());
		else
			stackModifiedGroup(op, left, newColGroups, smallestIndex, m2.getDenseBlockValues());

		if(newColGroups.size() == 0)
			return new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), 0.0);
		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(true);
		ret.setNonZeros(-1); // set unknown non zeros.
		return ret;
	}

	private static void stackModifiedGroup(BinaryOperator op, boolean left, final List<AColGroup> newColGroups,
		int smallestIndex, final double[] row) {
		AColGroup g;
		// select the smallest group to modify
		g = newColGroups.get(smallestIndex);
		if(left)
			g = g.binaryRowOpLeft(op, row, op.isRowSafeLeft(row));
		else
			g = g.binaryRowOpRight(op, row, op.isRowSafeRight(row));
		if(!(g instanceof ColGroupEmpty))
			newColGroups.set(smallestIndex, g); // overwrite the modified group.
		else
			newColGroups.remove(smallestIndex); // remove the element from the groups.
	}

	private static void stackConstGroup(MatrixBlock m2, BinaryOperator op, boolean left,
		final List<AColGroup> newColGroups, final int nCol, final double[] row) {
		AColGroup g;
		if(row == null) {
			// m2 must be sparse
			final double[] gVals = new double[nCol];
			final SparseBlock sb = m2.getSparseBlock();
			final double[] avals = sb.values(0);
			final int[] aix = sb.indexes(0);
			final int alen = sb.size(0);
			if(left)
				for(int i = 0; i < alen; i++)
					gVals[aix[i]] = op.fn.execute(avals[i], 0);
			else
				for(int i = 0; i < alen; i++)
					gVals[aix[i]] = op.fn.execute(0, avals[i]);
			g = ColGroupConst.create(gVals);
		}
		else {
			// if there is no appropriate group create a const.
			g = ColGroupConst.create(nCol, 0.0);
			if(left)
				g = g.binaryRowOpLeft(op, row, op.isRowSafeLeft(row));
			else
				g = g.binaryRowOpRight(op, row, op.isRowSafeRight(row));
		}
		if(!(g instanceof ColGroupEmpty))
			newColGroups.add(g); // add the const group.
	}

	private static MatrixBlock mvColCompressed(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left)
		throws Exception {

		final int nCols = m1.getNumColumns();
		final int nRows = m1.getNumRows();
		m1 = morph(m1);

		final int k = op.getNumThreads();
		long nnz = 0;

		boolean shouldBeSparseOut = false;
		if(op.fn.isBinary()) {
			// maybe it is good if this is a sparse output.
			// evaluate if it is good
			double est = evaluateSparsityMVCol(m1, m2, op, left);
			shouldBeSparseOut = MatrixBlock.evalSparseFormatInMemory(nRows, nCols, (long) (est * nRows * nCols));

		}
		MatrixBlock ret = new MatrixBlock(nRows, nCols, shouldBeSparseOut, -1).allocateBlock();

		if(shouldBeSparseOut) {
			if(k <= 1)
				nnz = binaryMVColSingleThreadSparse(m1, m2, op, left, ret);
			else
				nnz = binaryMVColMultiThreadSparse(m1, m2, op, left, ret);
		}
		else {
			if(k <= 1)
				nnz = binaryMVColSingleThreadDense(m1, m2, op, left, ret);
			else
				nnz = binaryMVColMultiThreadDense(m1, m2, op, left, ret);
		}

		if(op.fn instanceof ValueComparisonFunction) {
			if(nnz == (long) nRows * nCols)// all was 1
				return CompressedMatrixBlockFactory.createConstant(nRows, nCols, 1.0);
			else if(nnz == 0) // all was 0 -> return empty.
				return new MatrixBlock(nRows, nCols, 0.0);
		}

		ret.setNonZeros(nnz);
		ret.examSparsity(op.getNumThreads());

		return ret;
	}

	private static long binaryMVColSingleThreadDense(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left, MatrixBlock ret) {
		final int nRows = m1.getNumRows();
		long nnz = 0;
		if(left)
			nnz += new BinaryMVColLeftTaskDense(m1, m2, ret, 0, nRows, op).call();
		else
			nnz += new BinaryMVColTaskDense(m1, m2, ret, 0, nRows, op).call();
		return nnz;
	}

	private static long binaryMVColSingleThreadSparse(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left, MatrixBlock ret) {
		final int nRows = m1.getNumRows();
		long nnz = 0;
		if(left)
			throw new NotImplementedException();
		else
			nnz += new BinaryMVColTaskSparse(m1, m2, ret, 0, nRows, op).call();
		return nnz;
	}

	private static long binaryMVColMultiThreadDense(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left, MatrixBlock ret) throws Exception {
		final int nRows = m1.getNumRows();
		final int k = op.getNumThreads();
		final int blkz = ret.getNumRows() / k;
		long nnz = 0;
		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		try {
			final ArrayList<Callable<Long>> tasks = new ArrayList<>();
			for(int i = 0; i < nRows; i += blkz) {
				if(left)
					tasks.add(new BinaryMVColLeftTaskDense(m1, m2, ret, i, Math.min(nRows, i + blkz), op));
				else
					tasks.add(new BinaryMVColTaskDense(m1, m2, ret, i, Math.min(nRows, i + blkz), op));
			}
			for(Future<Long> f : pool.invokeAll(tasks))
				nnz += f.get();
		}
		finally {
			pool.shutdown();
		}
		return nnz;
	}

	private static long binaryMVColMultiThreadSparse(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left, MatrixBlock ret) throws Exception {
		final int nRows = m1.getNumRows();
		final int k = op.getNumThreads();
		final int blkz = Math.max(nRows / k, 64);
		long nnz = 0;
		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		try {
			final ArrayList<Callable<Long>> tasks = new ArrayList<>();
			for(int i = 0; i < nRows; i += blkz) {
				if(left)
					throw new NotImplementedException();
				else
					tasks.add(new BinaryMVColTaskSparse(m1, m2, ret, i, Math.min(nRows, i + blkz), op));
			}
			for(Future<Long> f : pool.invokeAll(tasks))
				nnz += f.get();
		}
		finally {
			pool.shutdown();
		}
		return nnz;
	}

	private static MatrixBlock mmCompressed(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left)
		throws Exception {
		final int nCols = m1.getNumColumns();
		final int nRows = m1.getNumRows();
		m1 = morph(m1);

		MatrixBlock ret = new MatrixBlock(nRows, nCols, false, -1).allocateBlock();

		long nnz = binaryMMExec(m1, m2, op, left, ret);

		ret.setNonZeros(nnz);
		ret.examSparsity(op.getNumThreads());
		return ret;
	}

	private static long binaryMMExec(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left,
		MatrixBlock ret) throws Exception {
		final int nRows = m1.getNumRows();
		final int k = op.getNumThreads();
		final int blkz = Math.max(ret.getNumRows() / k, 10);
		final long nnz;
		if(k <= 1)
			nnz = binaryMMSingleThread(m1, m2, op, left, ret, nRows, blkz);
		else
			nnz = binaryMMParallel(m1, m2, op, left, ret, nRows, blkz);
		return nnz;
	}

	private static long binaryMMParallel(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left,
		MatrixBlock ret, final int nRows, final int blkz) throws InterruptedException, ExecutionException {
		final ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
		long nnz = 0;
		try {
			final ArrayList<Callable<Long>> tasks = new ArrayList<>();
			for(int i = 0; i < nRows; i += blkz)
				tasks.add(new BinaryMMTask(m1, m2, ret, i, Math.min(nRows, i + blkz), op, left));

			for(Future<Long> f : pool.invokeAll(tasks))
				nnz += f.get();
		}
		finally {
			pool.shutdown();
		}
		return nnz;
	}

	private static long binaryMMSingleThread(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op, boolean left,
		MatrixBlock ret, final int nRows, final int blkz) {
		long nnz = 0;
		for(int i = 0; i < nRows; i += blkz)
			nnz += new BinaryMMTask(m1, m2, ret, i, Math.min(nRows, i + blkz), op, left).call();
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

	private static class BinaryMVColTaskDense implements Callable<Long> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		protected BinaryMVColTaskDense(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Long call() {
			final int _blklen = Math.max(16384 / _ret.getNumColumns(), 64);
			final List<AColGroup> groups = _m1.getColGroups();

			final AIterator[] its = getIterators(groups, _rl);

			for(int r = _rl; r < _ru; r += _blklen)
				processBlock(r, Math.min(r + _blklen, _ru), groups, its);

			return _ret.recomputeNonZeros(_rl, _ru - 1);
		}

		private final void processBlock(final int rl, final int ru, final List<AColGroup> groups, final AIterator[] its) {
			// unsafe decompress, since we count nonzeros afterwards.
			final DenseBlock db = _ret.getDenseBlock();
			decompressToSubBlock(rl, ru, db, groups, its);
			processGenericDense(rl, ru);
		}

		private final void processGenericDense(final int rl, final int ru) {
			final int ncol = _m1.getNumColumns();
			final DenseBlock rd = _ret.getDenseBlock();
			// m2 is a vector therefore guaranteed continuous.
			final double[] _m2Dense = _m2.getDenseBlockValues();
			for(int row = rl; row < ru; row++) {
				final double[] retDense = rd.values(row);
				final int posR = rd.pos(row);
				final double vr = _m2Dense[row];
				processRow(ncol, retDense, posR, vr);
			}
		}

		private void processRow(final int ncol, final double[] ret, final int posR, final double vr) {
			for(int col = 0; col < ncol; col++)
				ret[posR + col] = _op.fn.execute(ret[posR + col], vr);
		}

	}

	private static class BinaryMVColTaskSparse implements Callable<Long> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		private MatrixBlock tmp;

		protected BinaryMVColTaskSparse(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Long call() {
			final int _blklen = Math.max(16384 / _ret.getNumColumns(), 64);
			final List<AColGroup> groups = _m1.getColGroups();
			final AIterator[] its = getIterators(groups, _rl);
			tmp = new MatrixBlock(_blklen, _m1.getNumColumns(), false);
			tmp.allocateBlock();

			for(int r = _rl; r < _ru; r += _blklen)
				processBlock(r, Math.min(r + _blklen, _ru), groups, its);

			return _ret.recomputeNonZeros(_rl, _ru - 1);
		}

		private final void processBlock(final int rl, final int ru, final List<AColGroup> groups, final AIterator[] its) {
			decompressToTmpBlock(rl, ru, tmp.getDenseBlock(), groups, its);
			processDense(rl, ru);
			tmp.reset();
		}

		private final void processDense(final int rl, final int ru) {
			final int nCol = _m1.getNumColumns();
			final SparseBlock sb = _ret.getSparseBlock();
			final double[] _tmpDense = tmp.getDenseBlockValues();
			final double[] _m2Dense = _m2.getDenseBlockValues();
			for(int row = rl; row < ru; row++) {
				final double vr = _m2Dense[row];
				final int tmpOff = (row - rl) * nCol;
				for(int col = 0; col < nCol; col++)
					sb.append(row, col, _op.fn.execute(_tmpDense[tmpOff + col], vr));

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
			final int _blklen = Math.max(16384 / _ret.getNumColumns() / groups.size(), 64);
			final AIterator[] its = getIterators(groups, _rl);

			long nnz = 0;
			for(int r = _rl; r < _ru; r += _blklen) {
				final int re = Math.min(r + _blklen, _ru);
				processBlock(r, re, groups, its);
				nnz += _ret.recomputeNonZeros(r, re - 1);
			}

			return nnz;
		}

		private final void processBlock(final int rl, final int ru, final List<AColGroup> groups, final AIterator[] its) {
			final DenseBlock db = _ret.getDenseBlock();
			decompressToSubBlock(rl, ru, db, groups, its);

			if(_left)
				processLeft(rl, ru);
			else
				processRight(rl, ru);
		}

		private final void processLeft(final int rl, final int ru) {
			// all exec should have ret on right side
			if(_m2.isInSparseFormat())
				processLeftSparse(rl, ru);
			else
				processLeftDense(rl, ru);
		}

		private final void processLeftSparse(final int rl, final int ru) {
			final DenseBlock rv = _ret.getDenseBlock();
			final int cols = _ret.getNumColumns();
			final SparseBlock m2sb = _m2.getSparseBlock();
			for(int r = rl; r < ru; r++) {
				final double[] retV = rv.values(r);
				int off = rv.pos(r);
				if(m2sb.isEmpty(r)) {
					for(int c = off; c < cols + off; c++)
						retV[c] = _op.fn.execute(0, retV[c]);
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

					for(; j < cols; j++, off++)
						retV[off] = _op.fn.execute(0, retV[off]);
				}
			}
		}

		private final void processLeftDense(final int rl, final int ru) {
			final DenseBlock rv = _ret.getDenseBlock();
			final int cols = _ret.getNumColumns();
			DenseBlock m2db = _m2.getDenseBlock();
			for(int r = rl; r < ru; r++) {
				double[] retV = rv.values(r);
				double[] m2V = m2db.values(r);

				int off = rv.pos(r);
				for(int c = off; c < cols + off; c++)
					retV[c] = _op.fn.execute(m2V[c], retV[c]);
			}
		}

		private final void processRight(final int rl, final int ru) {
			if(_m2.isInSparseFormat())
				processRightSparse(rl, ru);
			else
				processRightDense(rl, ru);
		}

		private final void processRightSparse(final int rl, final int ru) {
			final DenseBlock rv = _ret.getDenseBlock();
			final int cols = _ret.getNumColumns();

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

					for(; j < cols; j++, off++)
						retV[off] = _op.fn.execute(retV[off], 0);
				}
			}

		}

		private final void processRightDense(final int rl, final int ru) {
			final DenseBlock rv = _ret.getDenseBlock();
			final int cols = _ret.getNumColumns();
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

	private static class BinaryMVColLeftTaskDense implements Callable<Long> {
		private final int _rl;
		private final int _ru;
		private final CompressedMatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		protected BinaryMVColLeftTaskDense(CompressedMatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Long call() {
			for(AColGroup g : _m1.getColGroups())
				g.decompressToDenseBlock(_ret.getDenseBlock(), _rl, _ru);

			// m2 is never sparse or empty. always dense here.
			final int ncol = _m1.getNumColumns();
			int offset = _rl * ncol;
			double[] _retDense = _ret.getDenseBlockValues();
			double[] _m2Dense = _m2.getDenseBlockValues();
			for(int row = _rl; row < _ru; row++) {
				double vr = _m2Dense[row];
				for(int col = 0; col < ncol; col++) {
					double v = _op.fn.execute(vr, _retDense[offset]);
					_retDense[offset] = v;
					offset++;
				}
			}

			return _ret.recomputeNonZeros(_rl, _ru - 1);

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

	protected static void decompressToSubBlock(final int rl, final int ru, final DenseBlock db,
		final List<AColGroup> groups, final AIterator[] its) {
		Timing time = new Timing(true);
		for(int i = 0; i < groups.size(); i++) {
			final AColGroup g = groups.get(i);
			if(g.getCompType() == CompressionType.SDC)
				((ASDCZero) g).decompressToDenseBlock(db, rl, ru, 0, 0, its[i]);
			else
				g.decompressToDenseBlock(db, rl, ru, 0, 0);
		}

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressToBlockTime(t, 1);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + 1 + " in " + t + "ms.");
		}
	}

	protected static void decompressToTmpBlock(final int rl, final int ru, final DenseBlock db,
		final List<AColGroup> groups, final AIterator[] its) {
		Timing time = new Timing(true);
		for(int i = 0; i < groups.size(); i++) {
			final AColGroup g = groups.get(i);
			if(g.getCompType() == CompressionType.SDC)
				((ASDCZero) g).decompressToDenseBlock(db, rl, ru, -rl, 0, its[i]);
			else
				g.decompressToDenseBlock(db, rl, ru, -rl, 0);
		}

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressToBlockTime(t, 1);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + 1 + " in " + t + "ms.");
		}
	}

	protected static AIterator[] getIterators(final List<AColGroup> groups, final int rl) {
		final AIterator[] its = new AIterator[groups.size()];
		for(int i = 0; i < groups.size(); i++) {

			final AColGroup g = groups.get(i);
			if(g.getCompType() == CompressionType.SDC)
				its[i] = ((ASDCZero) g).getIterator(rl);
		}
		return its;
	}

	private static double evaluateSparsityMVCol(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op,
		boolean left) {
		final List<AColGroup> groups = m1.getColGroups();
		final int nCol = m1.getNumColumns();
		final int nRow = m1.getNumRows();
		// sample max 5 rows, to analyze if we want sparse outputs.
		// we limit this much because we want the overhead of the
		// analysis very short.
		final int sampleRow = Math.min(nRow, 5);
		final int sampleCol = nCol;
		final int sampleNCells = sampleRow * sampleCol;
		final double[] dv = new double[sampleRow * sampleCol];

		// get vector dense values, guaranteed continuous.
		final double[] m2v = m2.getDenseBlockValues();

		decompressToDense(groups, sampleRow, sampleCol, dv);

		int nnz = 0;
		// m2v guaranteed to be dense and not empty.
		// if empty then we defaulted to scalar operations.
		if(left) {
			for(int r = 0; r < sampleRow; r++) {
				final double m = m2v[r];
				final int off = r * sampleCol;
				for(int c = 0; c < sampleCol; c++)
					nnz += op.fn.execute(m, dv[off + c]) != 0 ? 1 : 0;
			}
		}
		else {
			for(int r = 0; r < sampleRow; r++) {
				final double m = m2v[r];
				final int off = r * sampleCol;
				for(int c = 0; c < sampleCol; c++)
					nnz += op.fn.execute(dv[off + c], m) != 0 ? 1 : 0;
			}
		}

		return (double) nnz / (sampleNCells);

	}

	private static void decompressToDense(final List<AColGroup> groups, final int sampleRow, final int sampleCol,
		final double[] dv) {
		final DenseBlock db = new DenseBlockFP64(new int[] {sampleRow, sampleCol}, dv);
		for(int i = 0; i < groups.size(); i++) // decompress all to sample block
			groups.get(i).decompressToDenseBlock(db, 0, sampleRow);
	}
}
