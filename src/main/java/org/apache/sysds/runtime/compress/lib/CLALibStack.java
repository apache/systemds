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
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
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
	 * @param k The parallelization degree allowed for this operation
	 * @return The combined matrix.
	 */
	public static MatrixBlock combine(Map<MatrixIndexes, MatrixBlock> m, int k) {
		final MatrixIndexes lookup = new MatrixIndexes(1, 1);
		MatrixBlock b = m.get(lookup);
		if(b == null)
			throw new DMLCompressionException("Invalid map to combine does not contain the top left map MatrixBlock");
		final int blen = Math.max(b.getNumColumns(), b.getNumRows());

		// Dynamically find rlen, clen and blen;
		final long rows = findRLength(m, b);
		// TODO utilize the known size of m to extrapolate how many row blocks there are
		// and only use the last rowblock to calculate the total number of rows.
		final long cols = findCLength(m, b);

		return combine(m, lookup, (int) rows, (int) cols, blen, k);
	}

	public static MatrixBlock combine(Map<MatrixIndexes, MatrixBlock> m, int rlen, int clen, int blen, int k) {
		final MatrixIndexes lookup = new MatrixIndexes();
		return combine(m, lookup, rlen, clen, blen, k);
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

	private static MatrixBlock combine(final Map<MatrixIndexes, MatrixBlock> m, final MatrixIndexes lookup,
		final int rlen, final int clen, final int blen, final int k) {

		if(rlen < blen) // Shortcut, in case file only contains one block in r length.
			return combineColumnGroups(m, lookup, rlen, clen, blen, k);

		final CompressionType[] colTypes = new CompressionType[clen];
		// Look through the first blocks in to the top.
		for(int bc = 0; bc * blen < clen; bc++) {
			lookup.setIndexes(1, bc + 1); // get first blocks
			final MatrixBlock b = m.get(lookup);
			if(!(b instanceof CompressedMatrixBlock)) {
				LOG.warn("Found uncompressed matrix in Map of matrices, this is not"
					+ " supported in combine therefore falling back to decompression");
				return combineViaDecompression(m, rlen, clen, blen, k);
			}
			final CompressedMatrixBlock cmb = (CompressedMatrixBlock) b;
			if(cmb.isOverlapping()) {
				LOG.warn("Not supporting overlapping combine yet falling back to decompression");
				return combineViaDecompression(m, rlen, clen, blen, k);
			}
			final List<AColGroup> gs = cmb.getColGroups();
			final int off = bc * blen;
			for(AColGroup g : gs) {
				try {
					final IIterate cols = g.getColIndices().iterator();
					final CompressionType t = g.getCompType();
					while(cols.hasNext())
						colTypes[cols.next() + off] = t;
				}
				catch(Exception e) {
					throw new DMLCompressionException("Failed combining: " + g.toString());
				}
			}
		}

		// Look through the Remaining blocks down in the rows.
		for(int br = 1; br * blen < rlen; br++) {
			for(int bc = 0; bc * blen < clen; bc++) {
				lookup.setIndexes(br + 1, bc + 1); // get first blocks
				final MatrixBlock b = m.get(lookup);
				if(!(b instanceof CompressedMatrixBlock)) {
					LOG.warn("Found uncompressed matrix in Map of matrices, this is not"
						+ " supported in combine therefore falling back to decompression");
					return combineViaDecompression(m, rlen, clen, blen, k);
				}
				final CompressedMatrixBlock cmb = (CompressedMatrixBlock) b;
				if(cmb.isOverlapping()) {
					LOG.warn("Not supporting overlapping combine yet falling back to decompression");
					return combineViaDecompression(m, rlen, clen, blen, k);
				}
				final List<AColGroup> gs = cmb.getColGroups();
				final int off = bc * blen;
				for(AColGroup g : gs) {
					final IIterate cols = g.getColIndices().iterator();
					final CompressionType t = g.getCompType();
					while(cols.hasNext()) {
						final int c = cols.next();
						if(colTypes[c + off] != t) {
							LOG.warn("Not supported different types of column groups to combine."
								+ "Falling back to decompression of all blocks");
							return combineViaDecompression(m, rlen, clen, blen, k);
						}
					}
				}
			}
		}

		return combineColumnGroups(m, lookup, rlen, clen, blen, k);
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
	private static MatrixBlock combineColumnGroups(final Map<MatrixIndexes, MatrixBlock> m, final MatrixIndexes lookup,
		final int rlen, final int clen, final int blen, int k) {

		final AColGroup[][] finalCols = new AColGroup[clen][]; // temp array for combining
		final int blocksInColumn = (rlen - 1) / blen + 1;


		// Add all the blocks into linear structure.
		for(int br = 0; br * blen < rlen; br++) {
			for(int bc = 0; bc * blen < clen; bc++) {
				lookup.setIndexes(br + 1, bc + 1);
				final CompressedMatrixBlock cmb = (CompressedMatrixBlock) m.get(lookup);
				for(AColGroup g : cmb.getColGroups()) {
					final AColGroup gc = bc > 0 ? g.shiftColIndices(bc * blen) : g;
					final int c = gc.getColIndices().get(0);
					if(br == 0)
						finalCols[c] = new AColGroup[blocksInColumn];
					else if(finalCols[c] == null) {
						LOG.warn("Combining via decompression. There was an column"
							+ " assigned not assigned in block 1 indicating spark compression");
						return combineViaDecompression(m, rlen, clen, blen, k);
					}
					finalCols[c][br] = gc;
					if(br != 0 && (finalCols[c][0] == null || !finalCols[c][br].getColIndices().equals(finalCols[c][0].getColIndices()))){
						LOG.warn("Combining via decompression. There was an column with different index");
						return combineViaDecompression(m, rlen, clen, blen, k);
					}

				}
			}
		}


		final ExecutorService pool = CommonThreadPool.get(Math.max(Math.min(clen / 500, k), 1));
		try {

			List<AColGroup> finalGroups = pool.submit(() -> {
				return Arrays//
					.stream(finalCols)//
					.filter(x -> x != null)// filter all columns that are contained in other groups.
					.parallel()//
					.map(x -> {
						return combineN(x);
					}).collect(Collectors.toList());
			}).get();
			if(finalGroups.contains(null)) {
				LOG.warn("Combining via decompression. There was a column group that did not append ");
				return combineViaDecompression(m, rlen, clen, blen, k);
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

	private static AColGroup combineN(AColGroup[] groups) {
		try {
			return AColGroup.appendN(groups);

		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed to combine groups:\n" + Arrays.toString(groups), e);
		}
	}
}
