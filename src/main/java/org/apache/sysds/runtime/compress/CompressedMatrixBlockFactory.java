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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.utils.DblArrayIntListHashMap;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.DMLCompressionStatistics;

/**
 * Factory pattern to construct a CompressedMatrixBlock.
 */
public class CompressedMatrixBlockFactory {

	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlockFactory.class.getName());

	private Timing time = new Timing(true);
	private CompressionStatistics _stats = new CompressionStatistics();
	private MatrixBlock mb;
	private MatrixBlock original;
	private int k;
	private CompressionSettings compSettings;
	private CompressedMatrixBlock res = null;
	private int phase = 0;

	private CompressedSizeInfo coCodeColGroups;

	private CompressedMatrixBlockFactory(MatrixBlock mb, int k, CompressionSettings compSettings) {
		this.mb = mb;
		this.k = k;
		this.compSettings = compSettings;
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb) {
		// Default sequential execution of compression
		return compress(mb, 1, new CompressionSettingsBuilder().create());
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb,
		CompressionSettings customSettings) {
		return compress(mb, 1, customSettings);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k) {
		return compress(mb, k, new CompressionSettingsBuilder().create());
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
	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k,
		CompressionSettings compSettings) {
		CompressedMatrixBlockFactory cmbf = new CompressedMatrixBlockFactory(mb, k, compSettings);
		return cmbf.compressMatrix();
	}

	public static CompressedMatrixBlock createConstant(int numRows, int numCols, double value) {
		CompressedMatrixBlock block = new CompressedMatrixBlock(numRows, numCols);
		ColGroupConst cg = ColGroupConst.genColGroupConst(numRows, numCols, value);
		block.allocateColGroup(cg);
		block.setNonZeros(value == 0.0 ? 0 : numRows * numCols);
		return block;
	}

	private Pair<MatrixBlock, CompressionStatistics> compressMatrix() {
		// Check for redundant compression
		if(mb instanceof CompressedMatrixBlock) {
			throw new DMLRuntimeException("Redundant compression, block already compressed.");
		}
		original = mb;
		mb = new MatrixBlock().copyShallow(mb);

		classifyPhase();
		if(coCodeColGroups == null)
			return abortCompression();
		transposePhase();
		compressPhase();
		cleanupPhase();
		if(res == null)
			return abortCompression();

		res.recomputeNonZeros();
		return new ImmutablePair<>(res, _stats);
	}

	private void classifyPhase() {
		CompressedSizeEstimator sizeEstimator = CompressedSizeEstimatorFactory.getSizeEstimator(mb, compSettings);
		CompressedSizeInfo sizeInfos = sizeEstimator.computeCompressedSizeInfos(k);

		if(compSettings.investigateEstimate)
			_stats.estimatedSizeCols = sizeInfos.memoryEstimate();

		logPhase();
		// LOG.error(sizeInfos);
		if(sizeInfos.isCompressible(original.getInMemorySize()))
			coCodePhase(sizeEstimator, sizeInfos, mb.getNumRows());
	}

	private void coCodePhase(CompressedSizeEstimator sizeEstimator, CompressedSizeInfo sizeInfos, int numRows) {
		coCodeColGroups = PlanningCoCoder.findCoCodesByPartitioning(sizeEstimator, sizeInfos, numRows, k, compSettings);
		logPhase();
	}

	private void transposePhase() {
		boolean sparse = mb.isInSparseFormat();
		transposeHeuristics();
		mb = compSettings.transposed ? LibMatrixReorg.transpose(mb,
			new MatrixBlock(mb.getNumColumns(), mb.getNumRows(), sparse),
			k) : new MatrixBlock(mb.getNumRows(), mb.getNumColumns(), sparse).copyShallow(mb);
		logPhase();
	}

	private void transposeHeuristics() {
		switch(compSettings.transposeInput) {
			case "true":
				compSettings.transposed = true;
				break;
			case "false":
				compSettings.transposed = false;
				break;
			default:
				if(original.isInSparseFormat()) {
					boolean isAboveRowNumbers = mb.getNumRows() > 500000;
					boolean isAboveThreadToColumnRatio = coCodeColGroups.getNumberColGroups() > mb.getNumColumns() / 2;
					compSettings.transposed = isAboveRowNumbers || isAboveThreadToColumnRatio;
				}
				else
					compSettings.transposed = false;
		}
	}

	private void compressPhase() {
		res = new CompressedMatrixBlock(original);
		AColGroup[] colGroups = ColGroupFactory.compressColGroups(mb, coCodeColGroups, compSettings, k);
		List<AColGroup> colGroupList = assignColumns(original.getNumColumns(), colGroups, mb, compSettings);
		res.allocateColGroupList(colGroupList);
		logPhase();
	}

	private void cleanupPhase() {

		res.cleanupBlock(true, true);
		mb.cleanupBlock(true, true);

		_stats.size = res.estimateCompressedSizeInMemory();
		_stats.originalSize = original.estimateSizeInMemory();
		_stats.denseSize = MatrixBlock.estimateSizeInMemory(original.getNumRows(), original.getNumColumns(), 1.0);
		_stats.ratio = _stats.originalSize / (double) _stats.size;

		if(_stats.ratio < 1) {
			LOG.info("--dense size:        " + _stats.denseSize);
			LOG.info("--original size:     " + _stats.originalSize);
			LOG.info("--compressed size:   " + _stats.size);
			LOG.info("--compression ratio: " + _stats.ratio );
			LOG.info("Abort block compression because compression ratio is less than 1.");
			res = null;
			return;
		}

		_stats.setColGroupsCounts(res.getColGroups());

		logPhase();

	}

	private Pair<MatrixBlock, CompressionStatistics> abortCompression() {
		LOG.warn("Compression aborted at phase: " + phase);
		return new ImmutablePair<>(original, _stats);
	}

	private void logPhase() {
		_stats.setNextTimePhase(time.stop());
		DMLCompressionStatistics.addCompressionTime(_stats.getLastTimePhase(), phase);
		if(LOG.isDebugEnabled()) {
			switch(phase) {
				case 0:
					LOG.debug("--compression phase " + phase + " Classify  : " + _stats.getLastTimePhase());
					break;
				case 1:
					LOG.debug("--compression phase " + phase + " Grouping  : " + _stats.getLastTimePhase());
					break;
				case 2:
					LOG.debug("--compression phase " + phase + " Transpose : " + _stats.getLastTimePhase());
					break;
				case 3:
					LOG.debug("--compression phase " + phase + " Compress  : " + _stats.getLastTimePhase());
					LOG.debug("--compression Hash collisions:" + DblArrayIntListHashMap.hashMissCount);
					DblArrayIntListHashMap.hashMissCount = 0;
					break;
				// case 4:
				// LOG.debug("--compression phase " + phase++ + " Share : " + _stats.getLastTimePhase());
				// break;
				case 4:
					LOG.debug("--num col groups: " + res.getColGroups().size());
					LOG.debug("--compression phase " + phase + " Cleanup   : " + _stats.getLastTimePhase());
					LOG.debug("--col groups types " + _stats.getGroupsTypesString());
					LOG.debug("--col groups sizes " + _stats.getGroupsSizesString());
					LOG.debug("--dense size:        " + _stats.denseSize);
					LOG.debug("--original size:     " + _stats.originalSize);
					LOG.debug("--compressed size:   " + _stats.size);
					LOG.debug("--compression ratio: " + _stats.ratio );
					int[] lengths = new int[res.getColGroups().size()];
					int i = 0;
					for(AColGroup colGroup : res.getColGroups()) {
						if(colGroup.getValues() != null)
							lengths[i++] = colGroup.getValues().length / colGroup.getColIndices().length;
					}
					LOG.debug("--compressed colGroup dictionary sizes: " + Arrays.toString(lengths));
					if(LOG.isTraceEnabled()) {
						for(AColGroup colGroup : res.getColGroups()) {
							LOG.trace("--colGroups colIndexes : " + Arrays.toString(colGroup.getColIndices()));
							LOG.trace("--colGroups type       : " + colGroup.getClass().getSimpleName());
						}
					}
				default:
			}
		}
		phase++;
	}

	private List<AColGroup> assignColumns(int numCols, AColGroup[] colGroups, MatrixBlock rawBlock,
		CompressionSettings compSettings) {

		// Find the columns that are not assigned yet, and assign them to uncompressed.
		List<AColGroup> _colGroups = new ArrayList<>();
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
			LOG.warn("UnCompressible Columns: " + Arrays.toString(list));
			ColGroupUncompressed ucGroup = new ColGroupUncompressed(list, original, compSettings.transposed);
			_colGroups.add(ucGroup);
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
