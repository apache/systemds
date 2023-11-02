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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.CommonThreadPool;

public final class CLALibStack {
	protected static final Log LOG = LogFactory.getLog(CLALibStack.class.getName());

	private CLALibStack() {
		// private constructor
	}

	/**
	 * Combine the map of index matrix blocks into a single MatrixBlock.
	 * 
	 * The intension is that the combining is able to resolve differences in the different MatrixBlocks allocation.
	 * 
	 * @param m The map of Index to MatrixBLocks
	 * @param d A map of the dictionaries contained in the compression scheme
	 * @param k The parallelization degree allowed for this operation
	 * @return The combined matrix.
	 */
	public static MatrixBlock combine(Map<MatrixIndexes, MatrixBlock> m, Map<Integer, List<IDictionary>> d, int k) {

		final MatrixIndexes lookup = new MatrixIndexes(1, 1);
		MatrixBlock b = m.get(lookup);
		final int blen = Math.max(b.getNumColumns(), b.getNumRows());

		// Dynamically find rlen, clen and blen;
		final long rows = findRLength(m, b);
		// TODO utilize the known size of m to extrapolate how many row blocks there are
		// and only use the last rowblock to calculate the total number of rows.
		final long cols = findCLength(m, b);

		return combine(m, d, lookup, (int) rows, (int) cols, blen, k);
	}

	/**
	 * Alternative read, when dimensions are known.
	 * 
	 * @param m    The sub matrix blocks
	 * @param d    The sub dictionaries
	 * @param rlen n rows
	 * @param clen n cols
	 * @param blen block size
	 * @param k    Parallelization degree
	 * @return The combined matrixBlock
	 */
	public static MatrixBlock combine(Map<MatrixIndexes, MatrixBlock> m, Map<Integer, List<IDictionary>> d, int rlen,
		int clen, int blen, int k) {
		final MatrixIndexes lookup = new MatrixIndexes();
		return combine(m, d, lookup, rlen, clen, blen, k);
	}

	private static long findRLength(Map<MatrixIndexes, MatrixBlock> m, MatrixBlock b) {
		final MatrixIndexes lookup = new MatrixIndexes(1, 1);
		long rows = 0;
		while((b = m.get(lookup)) != null) {
			rows += b.getNumRows();
			lookup.setIndexes(lookup.getRowIndex() + 1, 1);
		}
		return rows;
	}

	private static long findCLength(Map<MatrixIndexes, MatrixBlock> m, MatrixBlock b) {
		final MatrixIndexes lookup = new MatrixIndexes(1, 1);
		long cols = 0;
		while((b = m.get(lookup)) != null) {
			cols += b.getNumColumns();
			lookup.setIndexes(1, lookup.getColumnIndex() + 1);
		}
		return cols;
	}

	private static MatrixBlock combine(final Map<MatrixIndexes, MatrixBlock> m, Map<Integer, List<IDictionary>> d,
		final MatrixIndexes lookup, final int rlen, final int clen, final int blen, final int k) {
		try {
			return combineColumnGroups(m, d, lookup, rlen, clen, blen, k);
		}
		catch(Exception e) {
			LOG.warn("Failed to combine compressed blocks, fallback to decompression.", e);
			return combineViaDecompression(m, rlen, clen, blen, k);
		}
	}

	private static MatrixBlock combineViaDecompression(final Map<MatrixIndexes, MatrixBlock> m, final int rlen,
		final int clen, final int blen, int k) {
		final MatrixBlock out = new MatrixBlock(rlen, clen, false);
		out.allocateDenseBlock();
		for(Entry<MatrixIndexes, MatrixBlock> e : m.entrySet()) {
			final MatrixIndexes ix = e.getKey();
			final MatrixBlock block = e.getValue();
			if(block != null) {
				final int row_offset = (int) (ix.getRowIndex() - 1) * blen;
				final int col_offset = (int) (ix.getColumnIndex() - 1) * blen;
				block.putInto(out, row_offset, col_offset, false);
			}
		}
		out.setNonZeros(-1);
		out.examSparsity(true);
		return out;
	}

	// It is known all of the matrices are Compressed and they are non overlapping.
	private static MatrixBlock combineColumnGroups(final Map<MatrixIndexes, MatrixBlock> m,
		Map<Integer, List<IDictionary>> d, final MatrixIndexes lookup, final int rlen, final int clen, final int blen,
		final int k) {

		int nGroups = 0;
		for(int bc = 0; bc * blen < clen; bc++) {
			// iterate through the first row of blocks to see number of column groups.
			lookup.setIndexes(1, bc + 1);
			final CompressedMatrixBlock cmb = (CompressedMatrixBlock) m.get(lookup);
			final List<AColGroup> gs = cmb.getColGroups();
			nGroups += gs.size();
		}

		final int blocksInColumn = rlen / blen + (rlen % blen > 0 ? 1 : 0);
		final AColGroup[][] finalCols = new AColGroup[nGroups][blocksInColumn]; // temp array for combining

		for(int br = 0; br * blen < rlen; br++) {
			int cgid = 0;
			for(int bc = 0; bc * blen < clen; bc++) {
				lookup.setIndexes(br + 1, bc + 1);
				final CompressedMatrixBlock cmb = (CompressedMatrixBlock) m.get(lookup);
				if(cmb == null) {
					throw new RuntimeException("Invalid empty read: " + lookup + "  " + rlen + " " + clen + " " + blen);
				}
				final List<AColGroup> gs = cmb.getColGroups();
				if(cgid + gs.size() > nGroups)
					return combineViaDecompression(m, rlen, clen, blen, k);

				for(int i = 0; i < gs.size(); i++) {
					AColGroup g = gs.get(i);
					final AColGroup gc = bc > 0 ? g.shiftColIndices(bc * blen) : g;
					finalCols[cgid][br] = gc;
					cgid++;
				}
			}
			if(cgid != finalCols.length) {
				LOG.warn("Combining via decompression. The number of columngroups in each block is not identical");
				return combineViaDecompression(m, rlen, clen, blen, k);
			}
		}

		final ExecutorService pool = CommonThreadPool.get();
		try {

			List<AColGroup> finalGroups = pool.submit(() -> {
				return Arrays//
					.stream(finalCols)//
					.parallel()//
					.map(x -> {
						AColGroup r = AColGroup.appendN(x, blen, rlen);
						return r;
					}).collect(Collectors.toList());
			}).get();

			if(finalGroups.contains(null)) {
				LOG.warn("Combining via decompression. There was a column group that did not append ");
				return combineViaDecompression(m, rlen, clen, blen, k);
			}

			if(d != null) {
				finalGroups = CLALibSeparator.combine(finalGroups, d, blen);
			}

			return new CompressedMatrixBlock(rlen, clen, -1, false, finalGroups);
		}
		catch(InterruptedException | ExecutionException e) {
			throw new DMLRuntimeException("Failed to combine column groups", e);
		}
		finally {
			pool.shutdown();
		}
	}
}
