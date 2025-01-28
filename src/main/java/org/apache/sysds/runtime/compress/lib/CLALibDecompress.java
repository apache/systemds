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
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.apache.sysds.utils.stats.Timing;

/**
 * Library to decompress a list of column groups into a matrix.
 */
public final class CLALibDecompress {
	private static final Log LOG = LogFactory.getLog(CLALibDecompress.class.getName());

	private CLALibDecompress() {
		// private constructor
	}

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

	public static void decompressTo(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset, int colOffset, int k,
		boolean countNNz) {
		decompressTo(cmb, ret, rowOffset, colOffset, k, countNNz, false);
	}

	public static void decompressTo(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset, int colOffset, int k,
		boolean countNNz, boolean reset) {
		Timing time = new Timing(true);
		if(cmb.getNumColumns() + colOffset > ret.getNumColumns() || cmb.getNumRows() + rowOffset > ret.getNumRows()) {
			LOG.warn(
				"Slow slicing off excess parts for decompressTo because decompression into is implemented for fitting blocks");
			MatrixBlock mbSliced = cmb.slice( //
				Math.min(Math.abs(rowOffset), 0), Math.min(cmb.getNumRows(), ret.getNumRows() - rowOffset) - 1, // Rows
				Math.min(Math.abs(colOffset), 0), Math.min(cmb.getNumColumns(), ret.getNumColumns() - colOffset) - 1); // Cols
			mbSliced.putInto(ret, rowOffset, colOffset, false);
			return;
		}

		final boolean outSparse = ret.isInSparseFormat();
		if(!cmb.isEmpty()) {
			if(outSparse && (cmb.isOverlapping() || reset))
				throw new DMLCompressionException("Not supported decompression into sparse block from overlapping state");
			else if(outSparse)
				decompressToSparseBlock(cmb, ret, rowOffset, colOffset);
			else
				decompressToDenseBlock(cmb, ret.getDenseBlock(), rowOffset, colOffset, k, reset);
		}

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressToBlockTime(t, k);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + k + " in " + t + "ms.");
		}

		if(countNNz)
			ret.recomputeNonZeros(k);
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
		sb.sort();
		ret.checkSparseRows();
	}

	private static void decompressToDenseBlock(CompressedMatrixBlock cmb, DenseBlock ret, int rowOffset, int colOffset,
		int k, boolean reset) {
		List<AColGroup> groups = cmb.getColGroups();
		// final int nCols = cmb.getNumColumns();
		final int nRows = cmb.getNumRows();

		final boolean shouldFilter = CLALibUtils.shouldPreFilter(groups);
		if(shouldFilter && !CLALibUtils.alreadyPreFiltered(groups, cmb.getNumColumns())) {
			final double[] constV = new double[cmb.getNumColumns()];
			groups = CLALibUtils.filterGroups(groups, constV);
			AColGroup cRet = ColGroupConst.create(constV);
			groups.add(cRet);
		}

		if(k > 1 && nRows > 1000)
			decompressToDenseBlockParallel(ret, groups, rowOffset, colOffset, nRows, k, reset);
		else
			decompressToDenseBlockSingleThread(ret, groups, rowOffset, colOffset, nRows, reset);
	}

	private static void decompressToDenseBlockSingleThread(DenseBlock ret, List<AColGroup> groups, int rowOffset,
		int colOffset, int nRows, boolean reset) {
		decompressToDenseBlockBlock(ret, groups, rowOffset, colOffset, 0, nRows, reset);
	}

	private static void decompressToDenseBlockBlock(DenseBlock ret, List<AColGroup> groups, int rowOffset, int colOffset,
		int rl, int ru, boolean reset) {
		if(reset) {
			if(ret.isContiguous()) {
				final int nCol = ret.getDim(1);
				ret.fillBlock(0, rl * nCol, ru * nCol, 0.0);
			}
			else
				throw new NotImplementedException();
		}
		for(AColGroup g : groups)
			g.decompressToDenseBlock(ret, rl, ru, rowOffset, colOffset);
	}

	private static void decompressToDenseBlockParallel(DenseBlock ret, List<AColGroup> groups, int rowOffset,
		int colOffset, int nRows, int k, boolean reset) {

		final int blklen = Math.max(nRows / k, 512);
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<Future<?>> tasks = new ArrayList<>(nRows / blklen);
			for(int r = 0; r < nRows; r += blklen) {
				final int start = r;
				final int end = Math.min(nRows, r + blklen);
				tasks.add(
					pool.submit(() -> decompressToDenseBlockBlock(ret, groups, rowOffset, colOffset, start, end, reset)));
			}

			for(Future<?> t : tasks)
				t.get();
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed parallel decompress to");
		}
		finally {
			pool.shutdown();
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
			ret.setNonZeros(ret.recomputeNonZeros(k));
			return ret; // if uncompressedColGroup is only colGroup.
		}

		final boolean shouldFilter = CLALibUtils.shouldPreFilterMorphOrRef(groups);
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

			if(MatrixBlock.evalSparseFormatInMemory(nRows, nCols, nonZeros) && !sparse)
				LOG.warn("Decompressing into dense but reallocating after to sparse: overlapping - " + overlapping
					+ ", filter - " + shouldFilter);
		}
		else{
			MatrixBlock tmp = new MatrixBlock();
			tmp.copy( ret);
			ret = tmp;
		}

		final int blklen = Math.max(nRows / k, 512);

		// check if we are using filtered groups, and if we are not force constV to null
		if(groups == filteredGroups)
			constV = null;

		final double eps = getEps(constV);
		if(k == 1) {
			if(ret.isInSparseFormat()) 
				decompressSparseSingleThread(ret, filteredGroups, nRows, blklen);
			else 
				decompressDenseSingleThread(ret, filteredGroups, nRows, blklen, constV, eps, overlapping);
		}
		else if(ret.isInSparseFormat()) 
			decompressSparseMultiThread(ret, filteredGroups, nRows, blklen, k);
		else 
			decompressDenseMultiThread(ret, filteredGroups, nRows, blklen, constV, eps, k, overlapping);

		ret.recomputeNonZeros(k);
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
		int blklen, double[] constV, double eps, boolean overlapping) {
		final DenseBlock db = ret.getDenseBlock();
		final int nCol = ret.getNumColumns();
		for(int i = 0; i < rlen; i += blklen) {
			final int rl = i;
			final int ru = Math.min(i + blklen, rlen);
			for(AColGroup grp : filteredGroups)
				grp.decompressToDenseBlock(db, rl, ru);
			if(constV != null)
				addVector(db, nCol, constV, eps, rl, ru);
		}
	}

	public static void decompressDense(MatrixBlock ret, List<AColGroup> groups, double[] constV,
		double eps, int k, boolean overlapping) {

		Timing time = new Timing(true);
		final int nRows = ret.getNumRows();
		final int blklen = Math.max(nRows / k, 512);
		if( k > 1)
			decompressDenseMultiThread(ret, groups, nRows, blklen, constV, eps, k, overlapping);
		else
			decompressDenseSingleThread(ret, groups, nRows, blklen, constV, eps, overlapping);
		
		ret.recomputeNonZeros(k);
		
		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressTime(t, k);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + k + " in " + t + "ms.");
		}
	}

	private static void decompressDenseMultiThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen, int blklen,
		double[] constV, double eps, int k, boolean overlapping) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<Callable<Long>> tasks = new ArrayList<>();
			if(overlapping || constV != null) {
				for(int i = 0; i < rlen; i += blklen)
					tasks.add(new DecompressDenseTask(filteredGroups, ret, eps, i, Math.min(i + blklen, rlen), constV));
			}
			else {
				for(int i = 0; i < rlen; i += blklen)
					for(AColGroup g : filteredGroups)
						tasks.add(new DecompressDenseSingleColTask(g, ret, eps, i, Math.min(i + blklen, rlen), null));
			}

			long nnz = 0;
			for(Future<Long> rt : pool.invokeAll(tasks))
				nnz += rt.get();
			ret.setNonZeros(nnz);
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLCompressionException("Parallel decompression failed", ex);
		}
		finally {
			pool.shutdown();
		}
	}

	private static void decompressSparseMultiThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen,
		int blklen, int k) {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			final ArrayList<DecompressSparseTask> tasks = new ArrayList<>();
			for(int i = 0; i < rlen; i += blklen)
				tasks.add(new DecompressSparseTask(filteredGroups, ret, i, Math.min(i + blklen, rlen)));

			for(Future<Object> rt : pool.invokeAll(tasks))
				rt.get();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLCompressionException("Parallel decompression failed", ex);
		}
		finally {
			pool.shutdown();
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
				if(v > max && Double.isFinite(v))
					max = v;
				if(v < min && Double.isFinite(v))
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
		private final int _blklen;
		private final double[] _constV;

		protected DecompressDenseTask(List<AColGroup> colGroups, MatrixBlock ret, double eps, int rl, int ru,
			double[] constV) {
			_colGroups = colGroups;
			_ret = ret;
			_eps = eps;
			_rl = rl;
			_ru = ru;
			_blklen = Math.max(32768 / ret.getNumColumns(), 128);
			_constV = constV;
		}

		@Override
		public Long call() {
			try {
				final DenseBlock db = _ret.getDenseBlock();
				final int nCol = _ret.getNumColumns();
				long nnz = 0;
				for(int b = _rl; b < _ru; b += _blklen) {
					final int e = Math.min(b + _blklen, _ru);
					for(AColGroup grp : _colGroups)
						grp.decompressToDenseBlock(db, b, e);

					if(_constV != null)
						addVector(db, nCol, _constV, _eps, b, e);
					nnz += _ret.recomputeNonZeros(b, e - 1);
				}

				return nnz;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLCompressionException("Failed dense decompression", e);
			}
		}
	}

	private static class DecompressDenseSingleColTask implements Callable<Long> {
		private final AColGroup _grp;
		private final MatrixBlock _ret;
		private final double _eps;
		private final int _rl;
		private final int _ru;
		private final int _blklen;
		private final double[] _constV;

		protected DecompressDenseSingleColTask(AColGroup grp, MatrixBlock ret, double eps, int rl, int ru,
			double[] constV) {
			_grp = grp;
			_ret = ret;
			_eps = eps;
			_rl = rl;
			_ru = ru;
			_blklen = Math.max(32768 / ret.getNumColumns(), 128);
			_constV = constV;
		}

		@Override
		public Long call() {
			try {
				final DenseBlock db = _ret.getDenseBlock();
				final int nCol = _ret.getNumColumns();
				long nnz = 0;
				for(int b = _rl; b < _ru; b += _blklen) {
					final int e = Math.min(b + _blklen, _ru);
					_grp.decompressToDenseBlock(db, b, e);

					if(_constV != null)
						addVector(db, nCol, _constV, _eps, b, e);
				}

				return nnz;
			}
			catch(Exception e) {
				e.printStackTrace();
				throw new DMLCompressionException("Failed dense decompression", e);
			}
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
		public Object call() throws Exception{
			try{

				final SparseBlock sb = _ret.getSparseBlock();
				for(AColGroup grp : _colGroups)
					grp.decompressToSparseBlock(_ret.getSparseBlock(), _rl, _ru);
				for(int i = _rl; i < _ru; i++)
					if(!sb.isEmpty(i))
						sb.sort(i);
				return null;
			}
			catch(Exception e){
				e.printStackTrace();
				throw new DMLRuntimeException(e);
			}
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
	private static final void addVector(final DenseBlock db, final int nCols, final double[] rowV, final double eps,
		final int rl, final int ru) {
		if(eps == 0)
			addVectorEps(db, nCols, rowV, eps, rl, ru);
		else
			addVectorNoEps(db, nCols, rowV, eps, rl, ru);
	}

	private static final void addVectorEps(final DenseBlock db, final int nCols, final double[] rowV, final double eps,
		final int rl, final int ru) {
		if(nCols == 1)
			addValue(db.values(0), rowV[0], rl, ru);
		else if(db.isContiguous())
			addVectorContiguousNoEps(db.values(0), rowV, nCols, rl, ru);
		else
			addVectorNoEps(db, rowV, nCols, rl, ru);
	}

	private static final void addVectorNoEps(final DenseBlock db, final int nCols, final double[] rowV, final double eps,
		final int rl, final int ru) {
		if(nCols == 1)
			addValueEps(db.values(0), rowV[0], eps, rl, ru);
		else if(db.isContiguous())
			addVectorContiguousEps(db.values(0), rowV, nCols, eps, rl, ru);
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
