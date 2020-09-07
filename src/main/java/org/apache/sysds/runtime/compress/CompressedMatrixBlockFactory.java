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

package org.apache.sysds.runtime.compress;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Factory pattern to construct a CompressedMatrixBlock.
 */
public class CompressedMatrixBlockFactory {

	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlockFactory.class.getName());
	private static final CompressionSettings defaultCompressionSettings = new CompressionSettingsBuilder().create();

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb) {
		// Default sequential execution of compression
		return compress(mb, 1, defaultCompressionSettings);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, CompressionSettings customSettings) {
		return compress(mb, 1, customSettings);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k) {
		return compress(mb, k, defaultCompressionSettings);
	}

	/**
	 * The main method for compressing the input matrix.
	 * 
	 * SAMPLE-BASED DECISIONS: Decisions such as testing if a column is amenable to bitmap compression or evaluating
	 * co-coding potentials are made based on a subset of the rows. For large data sets, sampling might take a
	 * significant amount of time. So, we generate only one sample and use it for the entire compression process.
	 * 
	 * Once the compression plan is selected based on sampling, the plan is verified and decisions are overwritten by
	 * full estimates.
	 * 
	 * @param mb           The matrix block to compress
	 * @param k            The number of threads used to execute the compression
	 * @param compSettings The Compression settings used
	 * @return A compressed matrix block.
	 */
	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k, CompressionSettings compSettings) {
		// Check for redundant compression
		if(mb instanceof CompressedMatrixBlock) {
			throw new DMLRuntimeException("Redundant compression, block already compressed.");
		}

		Timing time = new Timing(true);
		CompressionStatistics _stats = new CompressionStatistics();

		// Prepare basic meta data and deep copy / transpose input
		int numRows = mb.getNumRows();
		int numCols = mb.getNumColumns();
		boolean sparse = mb.isInSparseFormat();

		// Transpose the MatrixBlock if the TransposeInput flag is set.
		// This gives better cache consciousness, at a small upfront cost.
		MatrixBlock rawBlock = !compSettings.transposeInput ? new MatrixBlock(mb) : LibMatrixReorg
			.transpose(mb, new MatrixBlock(numCols, numRows, sparse), k);

		// Construct sample-based size estimator
		CompressedSizeEstimator sizeEstimator = CompressedSizeEstimatorFactory.getSizeEstimator(rawBlock, compSettings);

		// --------------------------------------------------
		// PHASE 1: Classify columns by compression type
		// Start by determining which columns are amenable to compression

		// Classify columns according to ratio (size uncompressed / size compressed),
		// where a column is compressible if ratio > 1.

		CompressedSizeInfo sizeInfos = sizeEstimator.computeCompressedSizeInfos(k);

		if(compSettings.investigateEstimate)
			_stats.estimatedSizeCols = sizeInfos.memoryEstimate();

		_stats.setNextTimePhase(time.stop());
		LOG.debug("Compression statistics:");
		LOG.debug("--compression phase 1: " + _stats.getLastTimePhase());

		if(sizeInfos.colsC.isEmpty()) {
			LOG.info("Abort block compression because all columns are incompressible.");
			return new ImmutablePair<>(new MatrixBlock().copyShallow(mb), _stats);
		}
		// --------------------------------------------------

		// --------------------------------------------------
		// PHASE 2: Grouping columns
		// Divide the columns into column groups.
		List<int[]> coCodeColGroups = PlanningCoCoder.findCoCodesByPartitioning(sizeEstimator, sizeInfos, numRows, k, compSettings);
		_stats.setNextTimePhase(time.stop());
		LOG.debug("--compression phase 2: " + _stats.getLastTimePhase());

		// TODO: Make second estimate of memory usage if the ColGroups are as above?
		// This should already be done inside the PlanningCoCoder, and therefore this information
		// should be returned there, and not estimated twice.
		// if(INVESTIGATE_ESTIMATES) {
		// _stats.estimatedSizeColGroups = memoryEstimateIfColsAre(coCodeColGroups);
		// }
		// --------------------------------------------------

		// --------------------------------------------------
		// PHASE 3: Compress and correct sample-based decisions
		ColGroup[] colGroups = ColGroupFactory
			.compressColGroups(rawBlock, sizeInfos.compRatios, coCodeColGroups, compSettings, k);

		// Make Compression happen!
		CompressedMatrixBlock res = new CompressedMatrixBlock(mb);
		List<ColGroup> colGroupList = ColGroupFactory.assignColumns(numCols, colGroups, rawBlock, compSettings);
		res.allocateColGroupList(colGroupList);
		_stats.setNextTimePhase(time.stop());
		if(LOG.isDebugEnabled()) {
			LOG.debug("--compression phase 3: " + _stats.getLastTimePhase());
		}
		// --------------------------------------------------

		// --------------------------------------------------
		// PHASE 4: Best-effort dictionary sharing for DDC1 single-col groups
		// Dictionary dict = (!(compSettings.validCompressions.contains(CompressionType.DDC)) ||
		// 	!(compSettings.allowSharedDDCDictionary)) ? null : createSharedDDC1Dictionary(colGroupList);
		// if(dict != null) {
		// 	applySharedDDC1Dictionary(colGroupList, dict);
		// 	res._sharedDDC1Dict = true;
		// }
		// _stats.setNextTimePhase(time.stop());
		if(LOG.isDebugEnabled()) {
			LOG.debug("--compression phase 4: " + _stats.getLastTimePhase());
		}
		// --------------------------------------------------

		// --------------------------------------------------
		// Phase 5: Cleanup
		// The remaining columns are stored uncompressed as one big column group
		_stats.size = res.estimateCompressedSizeInMemory();
		_stats.originalSize = mb.estimateSizeInMemory();
		_stats.ratio = _stats.originalSize / (double) _stats.size;

		if(_stats.ratio < 1) {
			LOG.info("Abort block compression because compression ratio is less than 1.");
			return new ImmutablePair<>(new MatrixBlock().copyShallow(mb), _stats);
		}

		// Final cleanup (discard uncompressed block)
		rawBlock.cleanupBlock(true, true);
		res.cleanupBlock(true, true);

		_stats.setNextTimePhase(time.stop());
		_stats.setColGroupsCounts(colGroupList);

		LOG.debug("--num col groups: " + colGroupList.size() + ", -- num input cols: " + numCols);
		LOG.debug("--compression phase 5: " + _stats.getLastTimePhase());
		LOG.debug("--col groups types " + _stats.getGroupsTypesString());
		LOG.debug("--col groups sizes " + _stats.getGroupsSizesString());
		LOG.debug("--compressed size: " + _stats.size);
		LOG.debug("--compression ratio: " + _stats.ratio);

		if( LOG.isTraceEnabled()){
			for (ColGroup colGroup : colGroupList) {
				LOG.trace("--colGroups colIndexes : " + Arrays.toString(colGroup.getColIndices()));
				LOG.trace("--colGroups type       : " + colGroup.getClass().getSimpleName());
				LOG.trace("--colGroups Values     : " + Arrays.toString(colGroup.getValues()));
			}
		}

		return new ImmutablePair<>(res, _stats);
		// --------------------------------------------------
	}

	/**
	 * Dictionary sharing between DDC ColGroups.
	 * 
	 * @param colGroups The List of all ColGroups.
	 * @return the shared value list for the DDC ColGroups.
	 */
	// private static Dictionary createSharedDDC1Dictionary(List<ColGroup> colGroups) {
	// 	// create joint dictionary
	// 	HashSet<Double> vals = new HashSet<>();
	// 	HashMap<Integer, Double> mins = new HashMap<>();
	// 	HashMap<Integer, Double> maxs = new HashMap<>();
	// 	int numDDC1 = 0;
	// 	for(final ColGroup grp : colGroups)
	// 		if(grp.getNumCols() == 1 && grp instanceof ColGroupDDC1) {
	// 			final ColGroupDDC1 grpDDC1 = (ColGroupDDC1) grp;
	// 			final double[] values = grpDDC1.getValues();
	// 			double min = Double.POSITIVE_INFINITY;
	// 			double max = Double.NEGATIVE_INFINITY;
	// 			for(int i = 0; i < values.length; i++) {
	// 				vals.add(values[i]);
	// 				min = Math.min(min, values[i]);
	// 				max = Math.max(max, values[i]);
	// 			}
	// 			mins.put(grpDDC1.getColIndex(0), min);
	// 			maxs.put(grpDDC1.getColIndex(0), max);
	// 			numDDC1++;
	// 		}

	// 	// abort shared dictionary creation if empty or too large
	// 	int maxSize = vals.contains(0d) ? 256 : 255;
	// 	if(numDDC1 < 2 || vals.size() > maxSize)
	// 		return null;

	// 	// build consolidated shared dictionary
	// 	double[] values = vals.stream().mapToDouble(Double::doubleValue).toArray();
	// 	int[] colIndexes = new int[numDDC1];
	// 	double[] extrema = new double[2 * numDDC1];
	// 	int pos = 0;
	// 	for(Entry<Integer, Double> e : mins.entrySet()) {
	// 		colIndexes[pos] = e.getKey();
	// 		extrema[2 * pos] = e.getValue();
	// 		extrema[2 * pos + 1] = maxs.get(e.getKey());
	// 		pos++;
	// 	}
	// 	return new DictionaryShared(values, colIndexes, extrema);
	// }

	// private static void applySharedDDC1Dictionary(List<ColGroup> colGroups, Dictionary dict) {
	// 	// create joint mapping table
	// 	HashMap<Double, Integer> map = new HashMap<>();
	// 	double[] values = dict.getValues();
	// 	for(int i = 0; i < values.length; i++)
	// 		map.put(values[i], i);

	// 	// recode data of all relevant DDC1 groups
	// 	for(ColGroup grp : colGroups)
	// 		if(grp.getNumCols() == 1 && grp instanceof ColGroupDDC1) {
	// 			ColGroupDDC1 grpDDC1 = (ColGroupDDC1) grp;
	// 			grpDDC1.recodeData(map);
	// 			grpDDC1.setDictionary(dict);
	// 		}
	// }
}
