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
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.Dictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell;
import org.apache.sysds.runtime.matrix.data.LibMatrixBincell.BinaryAccessType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class LibBinaryCellOp {

	private static final Log LOG = LogFactory.getLog(LibBinaryCellOp.class.getName());

	/**
	 * matrix-matrix binary operations, MM, MV
	 * 
	 * @param m1  Input matrix in compressed format to perform the operator on using m2
	 * @param m2  Input matrix 2 to use cell-wise on m1
	 * @param ret Result matrix to be returned in the operation.
	 * @param op  Binary operator such as multiply, add, less than, etc.
	 * @return The ret matrix, modified appropriately.
	 */
	public static MatrixBlock bincellOp(CompressedMatrixBlock m1, MatrixBlock m2, CompressedMatrixBlock ret,
		BinaryOperator op) {
		if(op.fn instanceof Minus) {
			ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), -1);
			m2 = m2.scalarOperations(sop, new MatrixBlock());
			return LibBinaryCellOp.bincellOp(m1, m2, ret, new BinaryOperator(Plus.getPlusFnObject()));
		}

		BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, m2);

		if(m1.isOverlapping() && !(op.fn instanceof Multiply || op.fn instanceof Divide)) {
			if(op.fn instanceof Plus || op.fn instanceof Minus) {
				return binaryMVPlusStack(m1, m2, ret, op);
			}
			else {
				throw new NotImplementedException(op + " not implemented for CLA");
			}

		}
		else {
			switch(atype) {
				case MATRIX_ROW_VECTOR:
					// Verify if it is okay to include all OuterVectorVector ops here.
					return binaryMVRow(m1, m2, ret, op);

				case OUTER_VECTOR_VECTOR:
					if(m2.getNumRows() == 1 && m2.getNumColumns() == 1) {
						return LibScalar.scalarOperations(new RightScalarOperator(op.fn, m2.quickGetValue(0, 0)),
							m1,
							ret,
							m1.isOverlapping());
					}
				default:
					LOG.warn("Inefficient Decompression for " + op + "  " + atype);
					MatrixBlock m1d = m1.decompress();
					return m1d.binaryOperations(op, m2, ret);

			}
		}

	}

	protected static CompressedMatrixBlock binaryMVRow(CompressedMatrixBlock m1, MatrixBlock m2,
		CompressedMatrixBlock ret, BinaryOperator op) {

		// Apply the operation to each of the column groups.
		// Most implementations will only modify metadata.
		List<ColGroup> oldColGroups = m1.getColGroups();
		List<ColGroup> newColGroups = new ArrayList<>(oldColGroups.size());
		double[] v = m2.getDenseBlockValues();
		boolean sparseSafe = true;
		for(double x : v) {
			if(op.fn.execute(x, 0.0) != 0.0) {
				sparseSafe = false;
				break;
			}
		}

		for(ColGroup grp : oldColGroups) {
			if(grp instanceof ColGroupUncompressed) {
				throw new DMLCompressionException("Not supported Binary MV");
			}
			else {
				if(grp.getNumCols() == 1) {
					ScalarOperator sop = new LeftScalarOperator(op.fn, m2.getValue(0, grp.getColIndices()[0]), 1);
					newColGroups.add(grp.scalarOperation(sop));
				}
				else {
					ColGroup ncg = grp.binaryRowOp(op, v, sparseSafe);
					newColGroups.add(ncg);
				}
			}
		}
		ret.allocateColGroupList(newColGroups);
		ret.setNonZeros(m1.getNumColumns() * m1.getNumRows());
		return ret;

	}

	protected static CompressedMatrixBlock binaryMVPlusStack(CompressedMatrixBlock m1, MatrixBlock m2,
		CompressedMatrixBlock ret, BinaryOperator op) {
		List<ColGroup> oldColGroups = m1.getColGroups();
		List<ColGroup> newColGroups = new ArrayList<>(oldColGroups.size() + 1);
		for(ColGroup grp : m1.getColGroups()) {
			newColGroups.add(grp);
		}
		int[] colIndexes = oldColGroups.get(0).getColIndices();
		double[] v = m2.getDenseBlockValues();
		ADictionary newDict = new Dictionary(new double[colIndexes.length]);
		newDict = newDict.applyBinaryRowOp(op.fn, v, true, colIndexes);
		newColGroups.add(new ColGroupConst(colIndexes, m1.getNumRows(), newDict));
		ret.allocateColGroupList(newColGroups);
		ret.setOverlapping(true);
		ret.setNonZeros(-1);
		return ret;
	}

	public static MatrixBlock binaryMVPlusCol(CompressedMatrixBlock m1, MatrixBlock m2, BinaryOperator op) {
		MatrixBlock ret = new MatrixBlock(m1.getNumRows(), m1.getNumColumns(), false, -1).allocateBlock();

		final int blkz = CompressionSettings.BITMAP_BLOCK_SZ;
		int k = OptimizerUtils.getConstrainedNumThreads(-1);
		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<BinaryMVColTask> tasks = new ArrayList<>();

		try {
			for(int i = 0; i * blkz < m1.getNumRows(); i++) {
				BinaryMVColTask rt = new BinaryMVColTask(m1.getColGroups(), m2, ret, i * blkz,
					Math.min(m1.getNumRows(), (i + 1) * blkz), op);
				tasks.add(rt);
			}
			List<Future<Integer>> futures = pool.invokeAll(tasks);
			pool.shutdown();
			long nnz = 0;
			for(Future<Integer> f : futures)
				nnz += f.get();
			ret.setNonZeros(nnz);
		}
		catch(InterruptedException | ExecutionException e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}

		return ret;
	}

	private static class BinaryMVColTask implements Callable<Integer> {
		private final List<ColGroup> _groups;
		private final int _rl;
		private final int _ru;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _op;

		protected BinaryMVColTask(List<ColGroup> groups, MatrixBlock m2, MatrixBlock ret, int rl, int ru,
			BinaryOperator op) {
			_groups = groups;
			_m2 = m2;
			_ret = ret;
			_op = op;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Integer call() {

			for(ColGroup g : _groups) {
				g.decompressToBlock(_ret, _rl, _ru, _rl, g.getValues());
			}

			int nnz = 0;
			DenseBlock db = _ret.getDenseBlock();
			for(int row = _rl; row < _ru; row++) {
				double vr = _m2.quickGetValue(row, 0);
				for(int col = 0; col < _ret.getNumColumns(); col++) {
					double v = _op.fn.execute(_ret.quickGetValue(row, col), vr);
					nnz += (v != 0) ? 1 : 0;
					db.set(row, col, v);
				}
			}

			return nnz;
		}
	}
}
