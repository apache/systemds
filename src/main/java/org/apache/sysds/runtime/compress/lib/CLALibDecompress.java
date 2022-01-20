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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.DMLCompressionStatistics;

/**
 * Library to decompress a list of column groups into a matrix.
 */
public class CLALibDecompress {
	private static final Log LOG = LogFactory.getLog(CLALibDecompress.class.getName());

	public static MatrixBlock decompress(CompressedMatrixBlock cmb, int k) {
		Timing time = new Timing(true);
		MatrixBlock ret = decompressExecute(cmb, k);
		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressTime(t, k);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + k + " in " + t + "ms.");
		}
		return ret;
	}

	public static void decompressTo(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset, int colOffset, int k) {
		Timing time = new Timing(true);
		final boolean outSparse = ret.isInSparseFormat();
		if(!cmb.isEmpty()) {
			if(outSparse && cmb.isOverlapping())
				throw new DMLCompressionException("Not supported decompression into sparse block from overlapping state");
			else if(outSparse)
				decompressToSparseBlock(cmb, ret, rowOffset, colOffset);
			else
				decompressToDenseBlock(cmb, ret.getDenseBlock(), rowOffset, colOffset);
		}

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressToBlockTime(t, k);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + k + " in " + t + "ms.");
		}
	}

	private static void decompressToSparseBlock(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset,
		int colOffset) {

		final SparseBlock sb = ret.getSparseBlock();
		final List<AColGroup> groups = cmb.getColGroups();
		final int nRows = cmb.getNumRows();
		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			final MatrixBlock tmp = cmb.getUncompressed("Decompression to put into Sparse Block");
			tmp.putInto(ret, rowOffset, colOffset, false);
		}
		else
			for(AColGroup g : groups)
				g.decompressToSparseBlock(sb, 0, nRows, rowOffset, colOffset);
	}

	private static void decompressToDenseBlock(CompressedMatrixBlock cmb, DenseBlock ret, int rowOffset, int colOffset) {
		final List<AColGroup> groups = cmb.getColGroups();
		// final int nCols = cmb.getNumColumns();
		final int nRows = cmb.getNumRows();

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter) {
			final double[] constV = new double[cmb.getNumColumns()];
			final List<AColGroup> filteredGroups = CLALibUtils.filterGroups(groups, constV);
			for(AColGroup g : filteredGroups)
				g.decompressToDenseBlock(ret, 0, nRows, rowOffset, colOffset);
			AColGroup cRet = ColGroupFactory.genColGroupConst(constV);
			cRet.decompressToDenseBlock(ret, 0, nRows, rowOffset, colOffset);
		}
		else {
			for(AColGroup g : groups)
				g.decompressToDenseBlock(ret, 0, nRows, rowOffset, colOffset);
		}
	}

	private static MatrixBlock decompressExecute(CompressedMatrixBlock cmb, int k) {

		if(cmb.isEmpty())
			return new MatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), true);
		// Copy column groups to make sure we can modify the list if we want to.
		final List<AColGroup> groups = new ArrayList<>(cmb.getColGroups());
		final int nRows = cmb.getNumRows();
		final int nCols = cmb.getNumColumns();
		final boolean overlapping = cmb.isOverlapping();
		final long nonZeros = cmb.getNonZeros();

		MatrixBlock ret = getUncompressedColGroupAndRemoveFromListOfColGroups(groups, overlapping, nRows, nCols);

		if(ret != null && groups.size() == 0) {
			ret.setNonZeros(ret.recomputeNonZeros());
			return ret; // if uncompressedColGroup is only colGroup.
		}

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		double[] constV = shouldFilter ? new double[nCols] : null;
		final List<AColGroup> filteredGroups = shouldFilter ? CLALibUtils.filterGroups(groups, constV) : groups;

		if(ret == null) { // There was no uncompressed group that fit the entire matrix.
			final boolean sparse = !shouldFilter && !overlapping &&
				MatrixBlock.evalSparseFormatInMemory(nRows, nCols, nonZeros);
			ret = new MatrixBlock(nRows, nCols, sparse);
			if(sparse)
				ret.allocateSparseRowsBlock();
			else
				ret.allocateDenseBlock();
		}

		final int blklen = Math.max(nRows / k, 512);

		// check if we are using filtered groups, and if we are not force constV to null
		if(groups == filteredGroups)
			constV = null;

		final double eps = getEps(constV);

		if(k == 1) {
			if(ret.isInSparseFormat()) {
				decompressSparseSingleThread(ret, filteredGroups, nRows, blklen);
				ret.setNonZeros(nonZeros);
			}
			else {
				decompressDenseSingleThread(ret, filteredGroups, nRows, blklen, constV, eps, nonZeros, overlapping);
				ret.setNonZeros(nonZeros == -1 || overlapping ? ret.recomputeNonZeros() : nonZeros);
			}
		}
		else if(ret.isInSparseFormat()) {
			decompressSparseMultiThread(ret, filteredGroups, nRows, blklen, k);
			ret.setNonZeros(nonZeros);
		}
		else
			decompressDenseMultiThread(ret, filteredGroups, nRows, blklen, constV, eps, k);

		ret.examSparsity();
		return ret;
	}

	private static MatrixBlock getUncompressedColGroupAndRemoveFromListOfColGroups(List<AColGroup> colGroups,
		boolean overlapping, int nRows, int nCols) {
		// If we have a uncompressed column group that covers all of the matrix,
		// it makes sense to use as the decompression target.
		MatrixBlock ret = null;

		// It is only relevant if we are in overlapping state, or we only have a Uncompressed ColumnGroup left.
		if(overlapping || colGroups.size() == 1) {
			for(int i = 0; i < colGroups.size(); i++) {
				AColGroup g = colGroups.get(i);
				if(g instanceof ColGroupUncompressed) {
					// Find an Uncompressed ColumnGroup
					ColGroupUncompressed guc = (ColGroupUncompressed) g;
					MatrixBlock gMB = guc.getData();
					// Make sure that it is the correct dimensions
					if(gMB.getNumColumns() == nCols && gMB.getNumRows() == nRows &&
						(!gMB.isInSparseFormat() || colGroups.size() == 1)) {
						colGroups.remove(i);
						LOG.debug("Using one of the uncompressed ColGroups as base for decompression");
						return gMB;
					}
				}
			}
		}

		return ret;
	}

	private static void decompressSparseSingleThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen,
		int blklen) {
		final SparseBlock sb = ret.getSparseBlock();
		for(int i = 0; i < rlen; i += blklen) {
			final int rl = i;
			final int ru = Math.min(i + blklen, rlen);
			for(AColGroup grp : filteredGroups)
				grp.decompressToSparseBlock(ret.getSparseBlock(), rl, ru);
			for(int j = rl; j < ru; j++)
				if(!sb.isEmpty(j))
					sb.sort(j);
		}

	}

	private static void decompressDenseSingleThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen,
		int blklen, double[] constV, double eps, long nonZeros, boolean overlapping) {
		for(int i = 0; i < rlen; i += blklen) {
			final int rl = i;
			final int ru = Math.min(i + blklen, rlen);
			for(AColGroup grp : filteredGroups)
				grp.decompressToDenseBlock(ret.getDenseBlock(), rl, ru);
			if(constV != null && !ret.isInSparseFormat())
				addVector(ret, constV, eps, rl, ru);
		}
	}

	protected static void decompressDenseMultiThread(MatrixBlock ret, List<AColGroup> groups, double[] constV, int k) {
		final int nRows = ret.getNumRows();
		final double eps = getEps(constV);
		final int blklen = Math.max(nRows / k, 512);
		decompressDenseMultiThread(ret, groups, nRows, blklen, constV, eps, k);
	}

	protected static void decompressDenseMultiThread(MatrixBlock ret, List<AColGroup> groups, double[] constV,
		double eps, int k) {
		final int nRows = ret.getNumRows();
		final int blklen = Math.max(nRows / k, 512);
		decompressDenseMultiThread(ret, groups, nRows, blklen, constV, eps, k);
	}

	private static void decompressDenseMultiThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen, int blklen,
		double[] constV, double eps, int k) {
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final ArrayList<DecompressDenseTask> tasks = new ArrayList<>();
			for(int i = 0; i < rlen; i += blklen)
				tasks.add(new DecompressDenseTask(filteredGroups, ret, eps, i, Math.min(i + blklen, rlen), constV));

			long nnz = 0;
			for(Future<Long> rt : pool.invokeAll(tasks))
				nnz += rt.get();
			pool.shutdown();
			ret.setNonZeros(nnz);
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLCompressionException("Parallel decompression failed", ex);
		}
	}

	private static void decompressSparseMultiThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen,
		int blklen, int k) {
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final ArrayList<DecompressSparseTask> tasks = new ArrayList<>();
			for(int i = 0; i < rlen; i += blklen)
				tasks.add(new DecompressSparseTask(filteredGroups, ret, i, Math.min(i + blklen, rlen)));

			for(Future<Object> rt : pool.invokeAll(tasks))
				rt.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLCompressionException("Parallel decompression failed", ex);
		}
	}

	/**
	 * Get a small epsilon from the constant group.
	 * 
	 * @param constV the constant vector.
	 * @return epsilon
	 */
	private static double getEps(double[] constV) {
		if(constV == null)
			return 0;
		else {
			double max = -Double.MAX_VALUE;
			double min = Double.MAX_VALUE;
			for(double v : constV) {
				if(v > max)
					max = v;
				if(v < min)
					min = v;
			}
			final double eps = (max + 1e-4 - min) * 1e-10;
			return eps;
		}
	}

	private static class DecompressDenseTask implements Callable<Long> {
		private final List<AColGroup> _colGroups;
		private final MatrixBlock _ret;
		private final double _eps;
		private final int _rl;
		private final int _ru;
		private final double[] _constV;

		protected DecompressDenseTask(List<AColGroup> colGroups, MatrixBlock ret, double eps, int rl, int ru,
			double[] constV) {
			_colGroups = colGroups;
			_ret = ret;
			_eps = eps;
			_rl = rl;
			_ru = ru;
			_constV = constV;
		}

		@Override
		public Long call() {
			final int blk = 1024;
			long nnz = 0;
			for(int b = _rl; b < _ru; b += blk) {
				int e = Math.min(b + blk, _ru);
				for(AColGroup grp : _colGroups)
					grp.decompressToDenseBlock(_ret.getDenseBlock(), b, e);

				if(_constV != null)
					addVector(_ret, _constV, _eps, b, e);
				nnz += _ret.recomputeNonZeros(b, e - 1);
			}

			return nnz;
		}
	}

	private static class DecompressSparseTask implements Callable<Object> {
		private final List<AColGroup> _colGroups;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;

		protected DecompressSparseTask(List<AColGroup> colGroups, MatrixBlock ret, int rl, int ru) {
			_colGroups = colGroups;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() {
			final SparseBlock sb = _ret.getSparseBlock();
			for(AColGroup grp : _colGroups)
				grp.decompressToSparseBlock(_ret.getSparseBlock(), _rl, _ru);
			for(int i = _rl; i < _ru; i++)
				if(!sb.isEmpty(i))
					sb.sort(i);
			return null;
		}
	}

	/**
	 * Add the rowV vector to each row in ret.
	 * 
	 * @param ret  matrix to add the vector to
	 * @param rowV The row vector to add
	 * @param eps  an epsilon defined, to round the output value to zero if the value is less than epsilon away from
	 *             zero.
	 * @param rl   The row to start at
	 * @param ru   The row to end at
	 */
	private static void addVector(final MatrixBlock ret, final double[] rowV, final double eps, final int rl,
		final int ru) {
		final int nCols = ret.getNumColumns();
		final DenseBlock db = ret.getDenseBlock();

		if(nCols == 1) {
			if(eps == 0)
				addValue(db.values(0), rowV[0], rl, ru);
			else
				addValueEps(db.values(0), rowV[0], eps, rl, ru);
		}
		else if(db.isContiguous()) {
			if(eps == 0)
				addVectorContiguousNoEps(db.values(0), rowV, nCols, rl, ru);
			else
				addVectorContiguousEps(db.values(0), rowV, nCols, eps, rl, ru);
		}
		else if(eps == 0)
			addVectorNoEps(db, rowV, nCols, rl, ru);
		else
			addVectorEps(db, rowV, nCols, eps, rl, ru);

	}

	private static void addValue(final double[] retV, final double v, final int rl, final int ru) {
		for(int off = rl; off < ru; off++)
			retV[off] += v;
	}

	private static void addValueEps(final double[] retV, final double v, final double eps, final int rl, final int ru) {
		for(int off = rl; off < ru; off++) {
			final double e = retV[off] + v;
			if(Math.abs(e) <= eps)
				retV[off] = 0;
			else
				retV[off] = e;
		}
	}

	private static void addVectorContiguousNoEps(final double[] retV, final double[] rowV, final int nCols, final int rl,
		final int ru) {
		for(int off = rl * nCols; off < ru * nCols; off += nCols) {
			for(int col = 0; col < nCols; col++) {
				final int out = off + col;
				retV[out] += rowV[col];
			}
		}
	}

	private static void addVectorContiguousEps(final double[] retV, final double[] rowV, final int nCols,
		final double eps, final int rl, final int ru) {
		for(int off = rl * nCols; off < ru * nCols; off += nCols) {
			for(int col = 0; col < nCols; col++) {
				final int out = off + col;
				retV[out] += rowV[col];
				if(Math.abs(retV[out]) <= eps)
					retV[out] = 0;
			}
		}
	}

	private static void addVectorNoEps(final DenseBlock db, final double[] rowV, final int nCols, final int rl,
		final int ru) {
		for(int row = rl; row < ru; row++) {
			final double[] _retV = db.values(row);
			final int off = db.pos(row);
			for(int col = 0; col < nCols; col++)
				_retV[off + col] += rowV[col];
		}
	}

	private static void addVectorEps(final DenseBlock db, final double[] rowV, final int nCols, final double eps,
		final int rl, final int ru) {
		for(int row = rl; row < ru; row++) {
			final double[] _retV = db.values(row);
			final int off = db.pos(row);
			for(int col = 0; col < nCols; col++) {
				final int out = off + col;
				_retV[out] += rowV[col];
				if(Math.abs(_retV[out]) <= eps)
					_retV[out] = 0;
			}
		}
	}
}
