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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Factory pattern for constructing ColGroups.
 */
public class ColGroupFactory {

	/**
	 * The actual compression method, that handles the logic of compressing multiple columns together. This method also
	 * have the responsibility of correcting any estimation errors previously made.
	 * 
	 * @param in           The input matrix, that could have been transposed if CompSettings was set to do that
	 * @param compRatios   The previously computed Compression ratings of individual col indexes.
	 * @param groups       The column groups to consider compressing together.
	 * @param compSettings The compression settings to construct the compression based on.
	 * @param k            The degree of parallelism used.
	 * @return A Resulting array of ColGroups, containing the compressed information from the input matrix block.
	 */
	public static ColGroup[] compressColGroups(MatrixBlock in, HashMap<Integer, Double> compRatios, List<int[]> groups,
		CompressionSettings compSettings, int k) {
		if(k <= 1) {
			return compressColGroups(in, compRatios, groups, compSettings);
		}
		else {
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<CompressTask> tasks = new ArrayList<>();
				for(int[] colIndexes : groups)
					tasks.add(new CompressTask(in, compRatios, colIndexes, compSettings));
				List<Future<ColGroup>> rtask = pool.invokeAll(tasks);
				ArrayList<ColGroup> ret = new ArrayList<>();
				for(Future<ColGroup> lrtask : rtask)
					ret.add(lrtask.get());
				pool.shutdown();
				return ret.toArray(new ColGroup[0]);
			}
			catch(InterruptedException | ExecutionException e) {
				// If there is an error in the parallel execution default to the non parallel implementation
				return compressColGroups(in, compRatios, groups, compSettings);
			}
		}
	}

	private static ColGroup[] compressColGroups(MatrixBlock in, HashMap<Integer, Double> compRatios, List<int[]> groups,
		CompressionSettings compSettings) {
		ColGroup[] ret = new ColGroup[groups.size()];
		for(int i = 0; i < groups.size(); i++)
			ret[i] = compressColGroup(in, compRatios, groups.get(i), compSettings);

		return ret;
	}

	private static class CompressedColumn implements Comparable<CompressedColumn> {
		final int colIx;
		final double compRatio;

		public CompressedColumn(int colIx, double compRatio) {
			this.colIx = colIx;
			this.compRatio = compRatio;
		}

		public static PriorityQueue<CompressedColumn> makePriorityQue(HashMap<Integer, Double> compRatios,
			int[] colIndexes) {
			PriorityQueue<CompressedColumn> compRatioPQ;

			// first modification
			compRatioPQ = new PriorityQueue<>();
			for(int i = 0; i < colIndexes.length; i++)
				compRatioPQ.add(new CompressedColumn(i, compRatios.get(colIndexes[i])));

			return compRatioPQ;
		}

		@Override
		public int compareTo(CompressedColumn o) {
			return (int) Math.signum(compRatio - o.compRatio);
		}
	}

	private static class CompressTask implements Callable<ColGroup> {
		private final MatrixBlock _in;
		private final HashMap<Integer, Double> _compRatios;
		private final int[] _colIndexes;
		private final CompressionSettings _compSettings;

		protected CompressTask(MatrixBlock in, HashMap<Integer, Double> compRatios, int[] colIndexes,
			CompressionSettings compSettings) {
			_in = in;
			_compRatios = compRatios;
			_colIndexes = colIndexes;
			_compSettings = compSettings;
		}

		@Override
		public ColGroup call() {
			return compressColGroup(_in, _compRatios, _colIndexes, _compSettings);
		}
	}

	private static ColGroup compressColGroup(MatrixBlock in, HashMap<Integer, Double> compRatios, int[] colIndexes,
		CompressionSettings compSettings) {

		int[] allGroupIndices = colIndexes.clone();

		CompressedSizeInfoColGroup sizeInfo;
		// The compression type is decided based on a full bitmap since it
		// will be reused for the actual compression step.
		ABitmap ubm = null;
		PriorityQueue<CompressedColumn> compRatioPQ = CompressedColumn.makePriorityQue(compRatios, colIndexes);

		// Switching to exact estimator here, when doing the actual compression.
		CompressedSizeEstimator estimator = new CompressedSizeEstimatorExact(in, compSettings);

		while(true) {

			// STEP 1.
			// Extract the entire input column list and observe compression ratio
			ubm = BitmapEncoder.extractBitmap(colIndexes, in, compSettings);
			sizeInfo = new CompressedSizeInfoColGroup(estimator.estimateCompressedColGroupSize(ubm),
				compSettings.validCompressions);

			// Throw error if for some reason the compression observed is 0.
			if(sizeInfo.getMinSize() == 0) {
				throw new DMLRuntimeException("Size info of compressed Col Group is 0");
			}

			// STEP 2.
			// Calculate the compression ratio compared to an uncompressed ColGroup type.
			double compRatio = sizeInfo.getCompressionSize(CompressionType.UNCOMPRESSED) / sizeInfo.getMinSize();

			// STEP 3.
			// Finish the search and close this compression if the group show good compression.

			// Seems a little early to stop here. Maybe reconsider how to decide when to stop.
			// Also when comparing to the case of 1.0 compression ratio, it could be that we chose to compress a group
			// worse than the individual columns.

			// Furthermore performance of a compressed representation that does not compress much, is decremental to
			// overall performance.
			if(compRatio > 1.0) {
				int rlen = compSettings.transposeInput ? in.getNumColumns() : in.getNumRows();
				return compress(colIndexes, rlen, ubm, sizeInfo.getBestCompressionType(), compSettings, in);
			}
			else {
				// STEP 4.
				// Try to remove the least compressible column from the columns to compress.
				// Then repeat from Step 1.

				allGroupIndices[compRatioPQ.poll().colIx] = -1;

				if(colIndexes.length - 1 == 0) {
					return null;
				}

				colIndexes = new int[colIndexes.length - 1];
				// copying the values that do not equal -1
				int ix = 0;
				for(int col : allGroupIndices)
					if(col != -1)
						colIndexes[ix++] = col;
			}
		}
	}

	/**
	 * Method for compressing an ColGroup.
	 * 
	 * @param colIndexes     The Column indexes to compress
	 * @param rlen           The number of rows in the columns
	 * @param ubm            The Bitmap containing all the data needed for the compression (unless Uncompressed
	 *                       ColGroup)
	 * @param compType       The CompressionType selected
	 * @param cs             The compression Settings used for the given compression
	 * @param rawMatrixBlock The copy of the original input (maybe transposed) MatrixBlock
	 * @return A Compressed ColGroup
	 */
	public static ColGroup compress(int[] colIndexes, int rlen, ABitmap ubm, CompressionType compType,
		CompressionSettings cs, MatrixBlock rawMatrixBlock) {
		switch(compType) {
			case DDC:
				if(ubm.getNumValues() < 256) {
					return new ColGroupDDC1(colIndexes, rlen, ubm, cs);
				}
				else {
					return new ColGroupDDC2(colIndexes, rlen, ubm, cs);
				}
			case RLE:
				return new ColGroupRLE(colIndexes, rlen, ubm, cs);
			case OLE:
				return new ColGroupOLE(colIndexes, rlen, ubm, cs);
			case UNCOMPRESSED:
				return new ColGroupUncompressed(colIndexes, rawMatrixBlock, cs);
			default:
				throw new DMLCompressionException("Not implemented ColGroup Type compressed in factory.");
		}
	}

	/**
	 * 
	 * Method for producing the final ColGroupList stored inside the CompressedMatrixBlock.
	 * 
	 * TODO Redesign this method such that it does not utilize the null pointers to decide on which ColGroups should be
	 * incompressable. This is done by changing both this method and compressColGroup inside this class.
	 * 
	 * @param numCols      The number of columns in input matrix
	 * @param colGroups    The colgroups made to assign
	 * @param rawBlock     The (maybe transposed) original MatrixBlock
	 * @param compSettings The Compressionsettings used.
	 * @return return the final ColGroupList.
	 */
	public static List<ColGroup> assignColumns(int numCols, ColGroup[] colGroups, MatrixBlock rawBlock,
		CompressionSettings compSettings) {

		List<ColGroup> _colGroups = new ArrayList<>();
		HashSet<Integer> remainingCols = seq(0, numCols - 1, 1);
		for(int j = 0; j < colGroups.length; j++) {
			if(colGroups[j] != null) {
				for(int col : colGroups[j].getColIndices())
					remainingCols.remove(col);
				_colGroups.add(colGroups[j]);
			}
		}

		if(!remainingCols.isEmpty()) {
			int[] list = remainingCols.stream().mapToInt(i -> i).toArray();
			ColGroupUncompressed ucgroup = new ColGroupUncompressed(list, rawBlock, compSettings);
			_colGroups.add(ucgroup);
		}
		return _colGroups;
	}

	private static HashSet<Integer> seq(int from, int to, int incr) {
		HashSet<Integer> ret = new HashSet<>();
		for(int i = from; i <= to; i += incr)
			ret.add(i);
		return ret;
	}
}
