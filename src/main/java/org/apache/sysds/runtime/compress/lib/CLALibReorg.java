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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class CLALibReorg {

	protected static final Log LOG = LogFactory.getLog(CLALibReorg.class.getName());

	public static boolean warned = false;

	public static MatrixBlock reorg(CompressedMatrixBlock cmb, ReorgOperator op, MatrixBlock ret, int startRow,
		int startColumn, int length) {
		// SwapIndex is transpose
		if(op.fn instanceof SwapIndex && cmb.getNumColumns() == 1) {
			MatrixBlock tmp = cmb.decompress(op.getNumThreads());
			long nz = tmp.setNonZeros(tmp.getNonZeros());
			if(tmp.isInSparseFormat())
				return LibMatrixReorg.transpose(tmp); // edge case...
			else
				tmp = new MatrixBlock(tmp.getNumColumns(), tmp.getNumRows(), tmp.getDenseBlockValues());
			tmp.setNonZeros(nz);
			return tmp;
		}
		else if(op.fn instanceof SwapIndex) {
			if(cmb.getCachedDecompressed() != null)
				return cmb.getCachedDecompressed().reorgOperations(op, ret, startRow, startColumn, length);

			// Allow transpose to be compressed output. In general we need to have a transposed flag on
			// the compressed matrix. https://issues.apache.org/jira/browse/SYSTEMDS-3025
			return transpose(cmb, ret, op.getNumThreads());
		}
		else {
			String message = !warned ? op.getClass().getSimpleName() + " -- " + op.fn.getClass().getSimpleName() : null;
			MatrixBlock tmp = cmb.getUncompressed(message, op.getNumThreads());
			warned = true;
			return tmp.reorgOperations(op, ret, startRow, startColumn, length);
		}
	}

	private static MatrixBlock transpose(CompressedMatrixBlock cmb, MatrixBlock ret, int k) {

		final long nnz = cmb.getNonZeros();
		final int nRow = cmb.getNumRows();
		final int nCol = cmb.getNumColumns();
		final boolean sparseOut = MatrixBlock.evalSparseFormatInMemory(nRow, nCol, nnz);
		if(sparseOut)
			return transposeSparse(cmb, ret, k, nRow, nCol, nnz);
		else
			return transposeDense(cmb, ret, k, nRow, nCol, nnz);
	}

	private static MatrixBlock transposeSparse(CompressedMatrixBlock cmb, MatrixBlock ret, int k, int nRow, int nCol,
		long nnz) {
		if(ret == null)
			ret = new MatrixBlock(nCol, nRow, true, nnz);
		else
			ret.reset(nCol, nRow, true, nnz);

		ret.allocateAndResetSparseBlock(true, SparseBlock.Type.MCSR);

		final int nColOut = ret.getNumColumns();

		if(k > 1)
			decompressToTransposedSparseParallel((SparseBlockMCSR) ret.getSparseBlock(), cmb.getColGroups(), nColOut, k);
		else
			decompressToTransposedSparseSingleThread((SparseBlockMCSR) ret.getSparseBlock(), cmb.getColGroups(), nColOut);

		return ret;
	}

	private static MatrixBlock transposeDense(CompressedMatrixBlock cmb, MatrixBlock ret, int k, int nRow, int nCol,
		long nnz) {
		if(ret == null)
			ret = new MatrixBlock(nCol, nRow, false, nnz);
		else
			ret.reset(nCol, nRow, false, nnz);

		// TODO: parallelize
		ret.allocateDenseBlock();

		decompressToTransposedDense(ret.getDenseBlock(), cmb.getColGroups(), nRow, 0, nRow);
		return ret;
	}

	private static void decompressToTransposedDense(DenseBlock ret, List<AColGroup> groups, int rlen, int rl, int ru) {
		for(int i = 0; i < groups.size(); i++) {
			AColGroup g = groups.get(i);
			g.decompressToDenseBlockTransposed(ret, rl, ru);
		}
	}

	private static void decompressToTransposedSparseSingleThread(SparseBlockMCSR ret, List<AColGroup> groups,
		int nColOut) {
		for(int i = 0; i < groups.size(); i++) {
			AColGroup g = groups.get(i);
			g.decompressToSparseBlockTransposed(ret, nColOut);
		}
	}

	private static void decompressToTransposedSparseParallel(SparseBlockMCSR ret, List<AColGroup> groups, int nColOut,
		int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final List<Future<?>> tasks = new ArrayList<>(groups.size());

			for(int i = 0; i < groups.size(); i++) {
				final AColGroup g = groups.get(i);
				tasks.add(pool.submit(() -> g.decompressToSparseBlockTransposed(ret, nColOut)));
			}

			for(Future<?> f : tasks)
				f.get();

		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed to parallel decompress transpose sparse", e);
		}
		finally {
			pool.shutdown();
		}
	}
}
