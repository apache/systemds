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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Library to decompress a list of column groups into a matrix.
 */
public class CLALibDecompress {
	public static MatrixBlock decompress(MatrixBlock ret, List<AColGroup> groups, long nonZeros, boolean overlapping) {

		final int rlen = ret.getNumRows();
		final int clen = ret.getNumColumns();
		final int block = (int) Math.ceil((double) (CompressionSettings.BITMAP_BLOCK_SZ) / clen);
		final int blklen = block > 1000 ? block + 1000 - block % 1000 : Math.max(64, block);
		final boolean containsSDC = CLALibUtils.containsSDC(groups);
		double[] constV = containsSDC ? new double[ret.getNumColumns()] : null;
		final List<AColGroup> filteredGroups = containsSDC ? CLALibUtils.filterSDCGroups(groups, constV) : groups;

		sortGroups(filteredGroups, overlapping);
		// check if we are using filtered groups, and if we are not force constV to null
		if(groups == filteredGroups)
			constV = null;

		for(int i = 0; i < rlen; i += blklen) {
			final int rl = i;
			final int ru = Math.min(i + blklen, rlen);
			for(AColGroup grp : filteredGroups)
				grp.decompressToBlockUnSafe(ret, rl, ru);
			if(constV != null && !ret.isInSparseFormat())
				addVector(ret, constV, rl, ru);
		}

		ret.setNonZeros(nonZeros == -1 || overlapping ? ret.recomputeNonZeros() : nonZeros);

		return ret;
	}

	public static MatrixBlock decompress(MatrixBlock ret, List<AColGroup> groups, boolean overlapping, int k) {

		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final int rlen = ret.getNumRows();
			final int block = (int) Math.ceil((double) (CompressionSettings.BITMAP_BLOCK_SZ) / ret.getNumColumns());
			final int blklen = block > 1000 ? block + 1000 - block % 1000 : Math.max(64, block);

			final boolean containsSDC = CLALibUtils.containsSDC(groups);
			double[] constV = containsSDC ? new double[ret.getNumColumns()] : null;
			final List<AColGroup> filteredGroups = containsSDC ? CLALibUtils.filterSDCGroups(groups, constV) : groups;
			sortGroups(filteredGroups, overlapping);

			// check if we are using filtered groups, and if we are not force constV to null
			if(groups == filteredGroups)
				constV = null;

			final ArrayList<DecompressTask> tasks = new ArrayList<>();
			for(int i = 0; i * blklen < rlen; i++)
				tasks.add(new DecompressTask(filteredGroups, ret, i * blklen, Math.min((i + 1) * blklen, rlen),
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

		return ret;
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
		return -Math.max(x.getMax(), Math.abs(x.getMin()));
	}

	private static class DecompressTask implements Callable<Long> {
		private final List<AColGroup> _colGroups;
		private final MatrixBlock _ret;
		private final int _rl;
		private final int _ru;
		private final double[] _constV;
		private final boolean _overlapping;

		protected DecompressTask(List<AColGroup> colGroups, MatrixBlock ret, int rl, int ru, boolean overlapping,
			double[] constV) {
			_colGroups = colGroups;
			_ret = ret;
			_rl = rl;
			_ru = ru;
			_overlapping = overlapping;
			_constV = constV;
		}

		@Override
		public Long call() {

			// preallocate sparse rows to avoid repeated alloc
			// if(!_overlapping && _ret.isInSparseFormat()) {
			// int[] rnnz = new int[_ru - _rl];
			// for(AColGroup grp : _colGroups)
			// grp.countNonZerosPerRow(rnnz, _rl, _ru);
			// SparseBlock rows = _ret.getSparseBlock();
			// for(int i = _rl; i < _ru; i++)
			// rows.allocate(i, rnnz[i - _rl]);
			// }

			// decompress row partition
			for(AColGroup grp : _colGroups)
				grp.decompressToBlockUnSafe(_ret, _rl, _ru);

			if(_constV != null && !_ret.isInSparseFormat()) {
				addVector(_ret, _constV, _rl, _ru);
			}

			// post processing (sort due to append)
			// if(_ret.isInSparseFormat())
			// _ret.sortSparseRows(_rl, _ru);

			return _overlapping ? 0 : _ret.recomputeNonZeros(_rl, _ru - 1);
		}
	}

	private static void addVector(MatrixBlock ret, double[] rowV, int rl, int ru) {
		final int nCols = ret.getNumColumns();

		final DenseBlock db = ret.getDenseBlock();
		for(int row = rl; row < ru; row++) {
			final double[] _retV = db.values(row);
			final int off = db.pos(row);
			for(int col = 0; col < nCols; col++) {
				_retV[off + col] += rowV[col];
			}
		}
	}
}
