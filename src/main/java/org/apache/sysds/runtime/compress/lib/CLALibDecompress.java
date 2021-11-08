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
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.DenseBlock;
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
		if(outSparse && cmb.isOverlapping())
			throw new DMLCompressionException("Not supported decompression into sparse block from overlapping state");
		else if(outSparse)
			decompressToSparseBlock(cmb, ret, rowOffset, colOffset);
		else
			decompressToDenseBlock(cmb, ret, rowOffset, colOffset);

		if(DMLScript.STATISTICS) {
			final double t = time.stop();
			DMLCompressionStatistics.addDecompressToBlockTime(t, k);
			if(LOG.isTraceEnabled())
				LOG.trace("decompressed block w/ k=" + k + " in " + t + "ms.");
		}
	}

	private static void decompressToSparseBlock(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset,
		int colOffset) {
		final List<AColGroup> groups = new ArrayList<>(cmb.getColGroups());
		final int nRows = cmb.getNumRows();

		for(AColGroup g : groups)
			g.decompressToBlock(ret, 0, nRows, rowOffset, colOffset);
	}

	private static void decompressToDenseBlock(CompressedMatrixBlock cmb, MatrixBlock ret, int rowOffset,
		int colOffset) {
		final List<AColGroup> groups = new ArrayList<>(cmb.getColGroups());
		// final int nCols = cmb.getNumColumns();
		final int nRows = cmb.getNumRows();

		final boolean containsSDC = CLALibUtils.containsSDCOrConst(groups);
		double[] constV = containsSDC ? new double[cmb.getNumColumns()] : null;
		final List<AColGroup> filteredGroups = containsSDC ? CLALibUtils.filterGroups(groups, constV) : groups;

		for(AColGroup g : filteredGroups)
			g.decompressToBlock(ret, 0, nRows, rowOffset, colOffset);

		if(constV != null) {
			AColGroup cRet = ColGroupFactory.genColGroupConst(constV);
			cRet.decompressToBlock(ret, 0, nRows, rowOffset, colOffset);
		}
	}

	private static MatrixBlock decompressExecute(CompressedMatrixBlock cmb, int k) {

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
		else if(ret == null) {
			ret = new MatrixBlock(nRows, nCols, false, -1);
			ret.allocateDenseBlock();
		}

		final int block = (int) Math.ceil((double) (CompressionSettings.BITMAP_BLOCK_SZ) / nCols);
		final int blklen = block > 1000 ? block + 1000 - block % 1000 : Math.max(64, block);

		final boolean containsSDC = CLALibUtils.containsSDCOrConst(groups);
		double[] constV = containsSDC ? new double[ret.getNumColumns()] : null;
		final List<AColGroup> filteredGroups = containsSDC ? CLALibUtils.filterGroups(groups, constV) : groups;
		if(LOG.isTraceEnabled())
			LOG.debug("Decompressing with block size: " + blklen);

		sortGroups(filteredGroups, overlapping);

		// check if we are using filtered groups, and if we are not force constV to null
		if(groups == filteredGroups)
			constV = null;

		final double eps = getEps(constV);
		if(k == 1)
			decompressSingleThread(ret, filteredGroups, nRows, blklen, constV, eps, nonZeros, overlapping);
		else
			decompressMultiThread(ret, filteredGroups, nRows, blklen, constV, eps, overlapping, k);

		if(overlapping)
			ret.recomputeNonZeros();

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

	private static void decompressSingleThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen, int blklen,
		double[] constV, double eps, long nonZeros, boolean overlapping) {
		for(int i = 0; i < rlen; i += blklen) {
			final int rl = i;
			final int ru = Math.min(i + blklen, rlen);
			for(AColGroup grp : filteredGroups)
				grp.decompressToBlock(ret, rl, ru);
			if(constV != null && !ret.isInSparseFormat())
				addVector(ret, constV, eps, rl, ru);
		}
		ret.setNonZeros(nonZeros == -1 || overlapping ? ret.recomputeNonZeros() : nonZeros);
	}

	private static void decompressMultiThread(MatrixBlock ret, List<AColGroup> filteredGroups, int rlen, int blklen,
		double[] constV, double eps, boolean overlapping, int k) {
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final ArrayList<DecompressTask> tasks = new ArrayList<>();
			for(int i = 0; i * blklen < rlen; i++)
				tasks.add(new DecompressTask(filteredGroups, ret, eps, i * blklen, Math.min((i + 1) * blklen, rlen),
					overlapping, constV));
			List<Future<Long>> rtasks = pool.invokeAll(tasks);
			pool.shutdown();

			long nnz = 0;
			for(Future<Long> rt : rtasks)
				nnz += rt.get();
			ret.setNonZeros(nnz);
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLCompressionException("Parallel decompression failed", ex);
		}
	}

	private static void sortGroups(List<AColGroup> groups, boolean overlapping) {
		if(overlapping) {
			// add a bit of stability in decompression
			Comparator<AColGroup> comp = Comparator.comparing(x -> effect(x));
			groups.sort(comp);
		}
	}

	/**
	 * Calculate an effect value for a column group. This is used to sort the groups before decompression to decompress
	 * the columns that have the smallest effect first.
	 * 
	 * @param x A Group
	 * @return A Effect double value.
	 */
	private static double effect(AColGroup x) {
		return (x instanceof ColGroupUncompressed) ? -Double.MAX_VALUE : -Math.max(x.getMax(), Math.abs(x.getMin()));
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

	private static class DecompressTask implements Callable<Long> {
		private final List<AColGroup> _colGroups;
		private final MatrixBlock _ret;
		private final double _eps;
		private final int _rl;
		private final int _ru;
		private final double[] _constV;
		private final boolean _overlapping;

		protected DecompressTask(List<AColGroup> colGroups, MatrixBlock ret, double eps, int rl, int ru,
			boolean overlapping, double[] constV) {
			_colGroups = colGroups;
			_ret = ret;
			_eps = eps;
			_rl = rl;
			_ru = ru;
			_overlapping = overlapping;
			_constV = constV;
		}

		@Override
		public Long call() {
			for(AColGroup grp : _colGroups)
				grp.decompressToBlock(_ret, _rl, _ru);

			if(_constV != null)
				addVector(_ret, _constV, _eps, _rl, _ru);

			return _overlapping ? 0 : _ret.recomputeNonZeros(_rl, _ru - 1);
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
		if(eps == 0) {
			for(int row = rl; row < ru; row++) {
				final double[] _retV = db.values(row);
				final int off = db.pos(row);
				for(int col = 0; col < nCols; col++)
					_retV[off + col] += rowV[col];
			}
		}
		else {
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
}
