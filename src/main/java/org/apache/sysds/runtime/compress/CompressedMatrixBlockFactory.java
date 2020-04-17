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

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC1;
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
	// local debug flag
	private static final boolean LOCAL_DEBUG = false;

	// DEBUG/TRACE for details
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;

	static {
		// for internal debugging only
		if(LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.compress").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}

	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlockFactory.class.getName());
	private static final CompressionSettings defaultCompressionSettings = new CompressionSettingsBuilder().create();

	public static MatrixBlock compress(MatrixBlock mb) {
		// Default sequential execution of compression
		return compress(mb, 1, defaultCompressionSettings);
	}

	public static MatrixBlock compress(MatrixBlock mb, CompressionSettings customSettings) {
		return compress(mb, 1, customSettings);
	}

	public static MatrixBlock compress(MatrixBlock mb, int k) {
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
	public static MatrixBlock compress(MatrixBlock mb, int k, CompressionSettings compSettings) {
		// Check for redundant compression
		if(mb instanceof CompressedMatrixBlock && ((CompressedMatrixBlock) mb).isCompressed()) {
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
			LOG.warn("Abort block compression because all columns are incompressible.");
			return new MatrixBlock().copyShallow(mb);
		}
		// --------------------------------------------------

		// --------------------------------------------------
		// PHASE 2: Grouping columns
		// Divide the columns into column groups.
		List<int[]> coCodeColGroups = PlanningCoCoder.findCocodesByPartitioning(sizeEstimator, sizeInfos, numRows, k);
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
		// TODO FIX DDC Sharing
		double[] dict = (!(compSettings.validCompressions.contains(CompressionType.DDC)) ||
			!(compSettings.allowSharedDDCDictionary)) ? null : createSharedDDC1Dictionary(colGroupList);
		if(dict != null) {
			applySharedDDC1Dictionary(colGroupList, dict);
			res._sharedDDC1Dict = true;
		}
		_stats.setNextTimePhase(time.stop());
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
			LOG.warn("Abort block compression because compression ratio is less than 1.");
			return new MatrixBlock().copyShallow(mb);
		}

		// Final cleanup (discard uncompressed block)
		rawBlock.cleanupBlock(true, true);
		res.cleanupBlock(true, true);

		_stats.setNextTimePhase(time.stop());
		_stats.setColGroupsCounts(colGroupList);

		LOG.info("--num col groups: " + colGroupList.size() + ", -- num input cols: " + numCols);
		LOG.debug("--compression phase 5: " + _stats.getLastTimePhase());
		LOG.debug("--col groups types " + _stats.getGroupsTypesString());
		LOG.debug("--col groups sizes " + _stats.getGroupsSizesString());
		LOG.debug("--compressed size: " + _stats.size);
		LOG.debug("--compression ratio: " + _stats.ratio);

		// Set the statistics object.
		// For better compression ratios this could be removed, since it is around 64 Bytes.
		res._stats = _stats;

		return res;
		// --------------------------------------------------
	}


	/**
	 * Dictionary sharing between DDC ColGroups.
	 * 
	 * FYI DOES NOT WORK FOR ALL CASES!
	 * @param colGroups The List of all ColGroups.
	 * @return the shared value list for the DDC ColGroups.
	 */
	private static double[] createSharedDDC1Dictionary(List<ColGroup> colGroups) {
		// create joint dictionary
		HashSet<Double> tmp = new HashSet<>();
		int numQual = 0;
		for(final ColGroup grp : colGroups)
			if(grp.getNumCols() == 1 && grp instanceof ColGroupDDC1) {
				final ColGroupDDC1 grpDDC1 = (ColGroupDDC1) grp;
				for(final double val : grpDDC1.getValues())
					tmp.add(val);
				numQual++;
			}

		// abort shared dictionary creation if empty or too large
		int maxSize = tmp.contains(0d) ? 256 : 255;
		if(tmp.isEmpty() || tmp.size() > maxSize || numQual < 2)
			return null;
		LOG.debug("Created shared directionary for " + numQual + " DDC1 single column groups.");

		// build consolidated dictionary
		return tmp.stream().mapToDouble(Double::doubleValue).toArray();
	}

	private static void applySharedDDC1Dictionary(List<ColGroup> colGroups, double[] dict) {
		// create joint mapping table
		HashMap<Double, Integer> map = new HashMap<>();
		for(int i = 0; i < dict.length; i++)
			map.put(dict[i], i);

		// recode data of all relevant DDC1 groups
		for(ColGroup grp : colGroups)
			if(grp.getNumCols() == 1 && grp instanceof ColGroupDDC1) {
				ColGroupDDC1 grpDDC1 = (ColGroupDDC1) grp;
				grpDDC1.recodeData(map);
				grpDDC1.setValues(dict);
			}
	}

}