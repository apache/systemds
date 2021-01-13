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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.AbstractCompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.Dictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
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
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibBinaryCellOp {

	private static final Log LOG = LogFactory.getLog(LibBinaryCellOp.class.getName());

	public static MatrixBlock binaryOperations(BinaryOperator op, CompressedMatrixBlock m1, MatrixValue thatValue,
		MatrixValue result) {
		MatrixBlock that = AbstractCompressedMatrixBlock.getUncompressed(thatValue);
		LibMatrixBincell.isValidDimensionsBinary(m1, that);

		return selectProcessingBasedOnAccessType(op, m1, that, thatValue, result);
	}

	private static MatrixBlock selectProcessingBasedOnAccessType(BinaryOperator op, CompressedMatrixBlock m1,
		MatrixBlock that, MatrixValue thatValue, MatrixValue result) {
		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, that);

		if(atype == BinaryAccessType.MATRIX_COL_VECTOR || atype == BinaryAccessType.MATRIX_MATRIX)
			return binaryMVCol(m1, that, op);
		else if(isSupportedBinaryCellOp(op.fn))
			return bincellOp(m1, that, setupCompressedReturnMatrixBlock(m1, result), op);
		else {
			LOG.warn("Decompressing since Binary Ops" + op.fn + " is not supported compressed");
			return AbstractCompressedMatrixBlock.getUncompressed(m1).binaryOperations(op, thatValue, result);
		}
	}

	private static boolean isSupportedBinaryCellOp(ValueFunction fn) {
		return fn instanceof Multiply || fn instanceof Divide || fn instanceof Plus || fn instanceof Minus ||
			fn instanceof MinusMultiply || fn instanceof PlusMultiply;
	}

	private static CompressedMatrixBlock setupCompressedReturnMatrixBlock(CompressedMatrixBlock m1,
		MatrixValue result) {
		CompressedMatrixBlock ret = null;
		if(result == null || !(result instanceof CompressedMatrixBlock))
			ret = new CompressedMatrixBlock(m1.getNumRows(), m1.getNumColumns(), false);
		else {
			ret = (CompressedMatrixBlock) result;
			ret.reset(m1.getNumRows(), m1.getNumColumns());
		}
		return ret;
	}

	private static MatrixBlock bincellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op) {
		if(isValidForOverlappingBinaryCellOperations(m1, op))
			overlappingBinaryCellOp(m1, m2, ret, op);
		else
			nonOverlappingBinaryCellOp(m1, m2, ret, op);
		return ret;

	}

	private static void nonOverlappingBinaryCellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op) {

		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, m2);
		switch(atype) {
			case MATRIX_ROW_VECTOR:
				// Verify if it is okay to include all OuterVectorVector ops here.
				binaryMVRow(m1, m2, ret, op);
				return;
			case OUTER_VECTOR_VECTOR:
				if(m2.getNumRows() == 1 && m2.getNumColumns() == 1) {
					LibScalar.scalarOperations(new RightScalarOperator(op.fn, m2.quickGetValue(0, 0)), m1, ret);
				}
				return;
			default:
				LOG.warn("Inefficient Decompression for " + op + "  " + atype);
				m1.decompress().binaryOperations(op, m2, ret);

		}
	}

	private static boolean isValidForOverlappingBinaryCellOperations(CompressedMatrixBlock m1, BinaryOperator op) {
		return m1.isOverlapping() && !(op.fn instanceof Multiply || op.fn instanceof Divide);
	}

	private static void overlappingBinaryCellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op) {
		if(op.fn instanceof Plus || op.fn instanceof Minus) {

			binaryMVPlusStack(m1, m2, ret, op);
		}
		else {
			throw new NotImplementedException(op + " not implemented for CLA");
		}
	}

	protected static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, MatrixBlock m2,
		CompressedMatrixBlock ret, BinaryOperator op) {

		List<ColGroup> oldColGroups = m1.getColGroups();
		double[] v = forceMatrixBlockToDense(m2);
		boolean sparseSafe = true;
		for(double x : v) {
			if(op.fn.execute(0.0, x) != 0.0) {
				sparseSafe = false;
				break;
			}
		}

		List<ColGroup> newColGroups = new ArrayList<>(oldColGroups.size());
		int k = OptimizerUtils.getConstrainedNumThreads(-1);
		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<BinaryMVRowTask> tasks = new ArrayList<>();
		try {
			for(ColGroup grp : oldColGroups) {
				if(grp instanceof ColGroupUncompressed) {
					throw new DMLCompressionException("Not supported uncompressed Col Group for Binary MV");
				}
				else {
					tasks.add(new BinaryMVRowTask(grp, v, sparseSafe, op));

				}
			}

			for(Future<ColGroup> f : pool.invokeAll(tasks))
				newColGroups.add(f.get());

			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}

		ret.allocateColGroupList(newColGroups);
		ret.setNonZeros(m1.getNumColumns() * m1.getNumRows());
		return ret;

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
		CompressedMatrixBlock ret, BinaryOperator op) {
		List<ColGroup> oldColGroups = m1.getColGroups();

		List<ColGroup> newColGroups = (m2.isEmpty()) ? new ArrayList<>(oldColGroups.size()) : new ArrayList<>(
			oldColGroups.size() + 1);
		boolean foundConst = false;
		for(ColGroup grp : m1.getColGroups()) {
			if(!m2.isEmpty() && !foundConst && grp instanceof ColGroupConst) {
				ADictionary newDict = ((ColGroupValue) grp).applyBinaryRowOp(op.fn, m2.getDenseBlockValues(), false);
				newColGroups.add(new ColGroupConst(grp.getColIndices(), m1.getNumRows(), newDict));
				foundConst = true;
			}
			else {
				newColGroups.add(grp);
			}
		}
		if(!m2.isEmpty() && !foundConst) {
			int[] colIndexes = oldColGroups.get(0).getColIndices();
			double[] v = m2.getDenseBlockValues();
			ADictionary newDict = new Dictionary(new double[colIndexes.length]);
			newDict = newDict.applyBinaryRowOp(op.fn, v, true, colIndexes);
			newColGroups.add(new ColGroupConst(colIndexes, m1.getNumRows(), newDict));
		}
		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(true);
		ret.setNonZeros(-1);
		return ret;
	}

	private static MatrixBlock binaryMVCol(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op) {

		MatrixBlock ret = new MatrixBlock(m1.getNumRows(), m1.getNumColumns(), false, -1).allocateBlock();

		final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
		int k = OptimizerUtils.getConstrainedNumThreads(-1);
		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<BinaryMVColTask> tasks = new ArrayList<>();

		try {
			for(int i = 0; i * blkz < m1.getNumRows(); i++) {
				tasks.add(new BinaryMVColTask(m1, m2, ret, i * blkz, Math.min(m1.getNumRows(), (i + 1) * blkz), op));
			}
			long nnz = 0;
			for(Future<Integer> f : pool.invokeAll(tasks))
				nnz += f.get();
			ret.setNonZeros(nnz);
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}

		return ret;
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
			for(ColGroup g : _m1.getColGroups()) {
				// unsafe decompress, since we count nonzeros afterwards.
				g.decompressToBlockSafe(_ret, _rl, _ru, g.getValues(), false);
			}

			int nnz = 0;
			DenseBlock db = _ret.getDenseBlock();
			for(int row = _rl; row < _ru; row++) {
				double vr = _m2.quickGetValue(row, 0);
				for(int col = 0; col < _m1.getNumColumns(); col++) {
					double v = _op.fn.execute(_ret.quickGetValue(row, col), vr);
					nnz += (v != 0) ? 1 : 0;
					db.set(row, col, v);
				}
			}

			return nnz;
		}
	}

	private static class BinaryMVRowTask implements Callable<ColGroup> {
		private final ColGroup _group;
		private final double[] _v;
		private final boolean _sparseSafe;
		private final BinaryOperator _op;

		protected BinaryMVRowTask(ColGroup group, double[] v, boolean sparseSafe, BinaryOperator op) {
			_group = group;
			_v = v;
			_op = op;
			_sparseSafe = sparseSafe;
		}

		@Override
		public ColGroup call() {
			return _group.binaryRowOp(_op, _v, _sparseSafe);
		}
	}
}
