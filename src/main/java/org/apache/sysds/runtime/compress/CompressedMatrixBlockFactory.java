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
import java.util.concurrent.CompletableFuture;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupValue;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.compress.cost.MemoryCostEstimator;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.DMLCompressionStatistics;

/**
 * Factory pattern to compress a Matrix Block into a CompressedMatrixBlock.
 */
public class CompressedMatrixBlockFactory {

	private static final Log LOG = LogFactory.getLog(CompressedMatrixBlockFactory.class.getName());

	/** Timing object to measure the time of each phase in the compression */
	private final Timing time = new Timing(true);
	/** Compression statistics gathered throughout the compression */
	private final CompressionStatistics _stats = new CompressionStatistics();
	/** Parallelization degree */
	private final int k;
	/** Compression settings used for this compression */
	private final CompressionSettings compSettings;
	/** The main cost estimator used for the compression */
	private final ACostEstimate costEstimator;

	/** Time stamp of last phase */
	private double lastPhase;
	/** Pointer to the original matrix Block that is about to be compressed. */
	private MatrixBlock mb;
	/** The resulting compressed matrix */
	private CompressedMatrixBlock res;
	/** The current Phase ID */
	private int phase = 0;
	/** Object to extract statistics from columns to make decisions based on */
	private AComEst informationExtractor;
	/** Compression information gathered through the sampling, used for the actual compression decided */
	private CompressedSizeInfo compressionGroups;

	private CompressedMatrixBlockFactory(MatrixBlock mb, int k, CompressionSettingsBuilder compSettings,
		ACostEstimate costEstimator) {
		this(mb, k, compSettings.create(), costEstimator);
	}

	private CompressedMatrixBlockFactory(MatrixBlock mb, int k, CompressionSettings compSettings,
		ACostEstimate costEstimator) {
		this.mb = mb;
		this.k = k;
		this.compSettings = compSettings;
		this.costEstimator = costEstimator;
	}

	/**
	 * Default sequential compression with no parallelization
	 * 
	 * @param mb The matrixBlock to compress
	 * @return A Pair of a Matrix Block and Compression Statistics.
	 */
	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb) {
		return compress(mb, 1, new CompressionSettingsBuilder(), (WTreeRoot) null);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, WTreeRoot root) {
		return compress(mb, 1, new CompressionSettingsBuilder(), root);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, CostEstimatorBuilder csb) {
		return compress(mb, 1, new CompressionSettingsBuilder(), csb);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, InstructionTypeCounter ins) {
		if(ins == null)
			return compress(mb, 1, new CompressionSettingsBuilder());
		return compress(mb, 1, new CompressionSettingsBuilder(), new CostEstimatorBuilder(ins));
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb,
		CompressionSettingsBuilder customSettings) {
		return compress(mb, 1, customSettings, (WTreeRoot) null);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k) {
		return compress(mb, k, new CompressionSettingsBuilder(), (WTreeRoot) null);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k, WTreeRoot root) {
		return compress(mb, k, new CompressionSettingsBuilder(), root);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k, CostEstimatorBuilder csb) {
		return compress(mb, k, new CompressionSettingsBuilder(), csb);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k, InstructionTypeCounter ins) {
		if(ins == null)
			return compress(mb, 1, new CompressionSettingsBuilder());
		return compress(mb, k, new CompressionSettingsBuilder(), new CostEstimatorBuilder(ins));
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, ACostEstimate costEstimator) {
		return compress(mb, 1, new CompressionSettingsBuilder(), costEstimator);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k, ACostEstimate costEstimator) {
		return compress(mb, k, new CompressionSettingsBuilder(), costEstimator);
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k,
		CompressionSettingsBuilder compSettings) {
		return compress(mb, k, compSettings, (WTreeRoot) null);
	}

	public static void compressAsync(ExecutionContext ec, String varName) {
		compressAsync(ec, varName, null);
	}

	public static void compressAsync(ExecutionContext ec, String varName, InstructionTypeCounter ins) {
		CompletableFuture.runAsync(() -> {
			// method call or code to be asynch.
			CacheableData<?> data = ec.getCacheableData(varName);
			if(data instanceof MatrixObject) {
				MatrixObject mo = (MatrixObject) data;
				MatrixBlock mb = mo.acquireReadAndRelease();
				MatrixBlock mbc = CompressedMatrixBlockFactory.compress(mo.acquireReadAndRelease(), ins).getLeft();
				if(mbc instanceof CompressedMatrixBlock) {
					ExecutionContext.createCacheableData(mb);
					mo.acquireModify(mbc);
					mo.release();
				}
			}
		});
	}

	/**
	 * The main method for compressing the input matrix.
	 * 
	 * 
	 * @param mb           The matrix block to compress
	 * @param k            The number of threads used to execute the compression
	 * @param compSettings The Compression settings used
	 * @param root         The root instruction compressed, and used for calculating the computation cost of the
	 *                     compression
	 * @return A pair of an possibly compressed matrix block and compression statistics.
	 */
	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k,
		CompressionSettingsBuilder compSettings, WTreeRoot root) {
		CompressionSettings cs = compSettings.create();
		ACostEstimate ice;
		if(root == null)
			ice = CostEstimatorFactory.create(cs, null, mb.getNumRows(), mb.getNumColumns(), mb.getSparsity());
		else {
			CostEstimatorBuilder csb = new CostEstimatorBuilder(root);
			ice = CostEstimatorFactory.create(cs, csb, mb.getNumRows(), mb.getNumColumns(), mb.getSparsity());
		}
		CompressedMatrixBlockFactory cmbf = new CompressedMatrixBlockFactory(mb, k, cs, ice);
		return cmbf.compressMatrix();
	}

	/**
	 * The main method for compressing the input matrix.
	 * 
	 * @param mb           The matrix block to compress
	 * @param k            The number of threads used to execute the compression
	 * @param compSettings The Compression settings used
	 * @param csb          The cost estimation builder
	 * @return A pair of an possibly compressed matrix block and compression statistics.
	 */
	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k,
		CompressionSettingsBuilder compSettings, CostEstimatorBuilder csb) {
		CompressionSettings cs = compSettings.create();
		ACostEstimate ice = CostEstimatorFactory.create(cs, csb, mb.getNumRows(), mb.getNumColumns(), mb.getSparsity());

		CompressedMatrixBlockFactory cmbf = new CompressedMatrixBlockFactory(mb, k, cs, ice);
		return cmbf.compressMatrix();
	}

	public static Pair<MatrixBlock, CompressionStatistics> compress(MatrixBlock mb, int k,
		CompressionSettingsBuilder compSettings, ACostEstimate costEstimator) {
		CompressedMatrixBlockFactory cmbf = new CompressedMatrixBlockFactory(mb, k, compSettings, costEstimator);
		return cmbf.compressMatrix();
	}

	/**
	 * Generate a CompressedMatrixBlock Object that contains a single uncompressed matrix block column group. Note this
	 * could be an empty colgroup if the input is empty.
	 * 
	 * @param mb The matrix block to be contained in the uncompressed matrix block column,
	 * @return a CompressedMatrixBlock
	 */
	public static CompressedMatrixBlock genUncompressedCompressedMatrixBlock(MatrixBlock mb) {
		CompressedMatrixBlock ret = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
		AColGroup cg = ColGroupUncompressed.create(mb);
		ret.allocateColGroup(cg);
		ret.setNonZeros(mb.getNonZeros());
		return ret;
	}

	/**
	 * Method for constructing a compressed matrix out of an constant input.
	 * 
	 * Since the input is a constant value it is trivially compressable, therefore we skip the entire compression
	 * planning and directly return a compressed constant matrix
	 * 
	 * @param numRows The number of Rows in the matrix
	 * @param numCols The number of Columns in the matrix
	 * @param value   The value contained in the matrix
	 * @return The Compressed Constant matrix.
	 */
	public static CompressedMatrixBlock createConstant(int numRows, int numCols, double value) {
		CompressedMatrixBlock block = new CompressedMatrixBlock(numRows, numCols);
		AColGroup cg = ColGroupConst.create(numCols, value);
		block.allocateColGroup(cg);
		block.recomputeNonZeros();
		if(block.getNumRows() == 0 || block.getNumColumns() == 0) {
			throw new DMLCompressionException("Invalid size of allocated constant compressed matrix block");
		}
		return block;
	}

	private Pair<MatrixBlock, CompressionStatistics> compressMatrix() {

		if(mb instanceof CompressedMatrixBlock) // Redundant compression
			return returnSelf();
		else if(mb.isEmpty()) // empty input return empty compression
			return createEmpty();

		_stats.denseSize = MatrixBlock.estimateSizeInMemory(mb.getNumRows(), mb.getNumColumns(), 1.0);
		_stats.originalSize = mb.getInMemorySize();
		_stats.originalCost = costEstimator.getCost(mb);

		res = new CompressedMatrixBlock(mb); // copy metadata and allocate soft reference

		classifyPhase();
		if(compressionGroups == null)
			return abortCompression();

		// clear extra data from analysis
		compressionGroups.clearMaps();
		informationExtractor.clearNNZ();

		transposePhase();
		compressPhase();
		finalizePhase();

		if(res == null)
			return abortCompression();

		return new ImmutablePair<>(res, _stats);
	}

	private void classifyPhase() {
		// Create the extractor for column statistics
		informationExtractor = ComEstFactory.createEstimator(mb, compSettings, k);
		// Compute the individual columns cost information
		compressionGroups = informationExtractor.computeCompressedSizeInfos(k);

		if(LOG.isTraceEnabled()) {
			LOG.trace("Logging all individual columns estimated cost:");
			for(CompressedSizeInfoColGroup g : compressionGroups.getInfo())
				LOG.trace(String.format("Cost: %8.0f Size: %16d %15s", costEstimator.getCost(g), g.getMinSize(),
					Arrays.toString(g.getColumns())));
		}

		_stats.estimatedSizeCols = compressionGroups.memoryEstimate();
		_stats.estimatedCostCols = costEstimator.getCost(compressionGroups);

		logPhase();
		final int nCols = compSettings.transposed ? mb.getNumRows() : mb.getNumColumns();
		final double scale = (costEstimator instanceof ComputationCostEstimator) ? ((double) nCols) / 2 : 1;
		final double threshold = _stats.estimatedCostCols / scale;

		if(threshold < _stats.originalCost) {
			if(nCols > 1)
				coCodePhase();
			else // LOG a short cocode phase (since there is one column we don't cocode)
				logPhase();
		}
		else {
			// abort compression
			compressionGroups = null;
			if(LOG.isInfoEnabled()) {
				LOG.info("Aborting before co-code, because the compression looks bad");
				LOG.info("Threshold was set to : " + threshold + " but it was above original " + _stats.originalCost);
				LOG.info("Original size       : " + _stats.originalSize);
				LOG.info("single col size     : " + _stats.estimatedSizeCols);
				if(!(costEstimator instanceof MemoryCostEstimator)) {
					LOG.info("original cost      : " + _stats.originalCost);
					LOG.info("single col cost    : " + _stats.estimatedCostCols);
				}
			}
		}
	}

	private void coCodePhase() {

		compressionGroups = CoCoderFactory.findCoCodesByPartitioning(informationExtractor, compressionGroups, k,
			costEstimator, compSettings);

		_stats.estimatedSizeCoCoded = compressionGroups.memoryEstimate();
		_stats.estimatedCostCoCoded = costEstimator.getCost(compressionGroups);

		logPhase();
		// if cocode is estimated larger than uncompressed abort compression.
		if(_stats.estimatedCostCoCoded > _stats.originalCost) {
			// abort compression
			compressionGroups = null;
			if(LOG.isInfoEnabled()) {
				LOG.info("Aborting after co-code, because the compression looks bad");
				LOG.info("co-code size      : " + _stats.estimatedSizeCoCoded);
				LOG.info("original size     : " + _stats.originalSize);
				if(!(costEstimator instanceof MemoryCostEstimator)) {
					LOG.info("original cost    : " + _stats.originalCost);
					LOG.info("single col cost  : " + _stats.estimatedCostCols);
					LOG.info("co-code cost     : " + _stats.estimatedCostCoCoded);
				}
			}
		}
	}

	private void transposePhase() {
		final boolean haveMemory = Runtime.getRuntime().freeMemory() - (mb.estimateSizeInMemory() * 2) > 0;
		if(!compSettings.transposed && haveMemory) {
			transposeHeuristics();
			if(compSettings.transposed) {
				boolean sparse = mb.isInSparseFormat();
				mb = LibMatrixReorg.transpose(mb, new MatrixBlock(mb.getNumColumns(), mb.getNumRows(), sparse), k, true);
				mb.evalSparseFormatInMemory();
			}
		}

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
				if(mb.isInSparseFormat()) {
					boolean haveManyColumns = mb.getNumColumns() > 10000;
					boolean isNnzLowAndVerySparse = mb.getNonZeros() < 1000 && mb.getSparsity() < 0.4;
					boolean isAboveRowNumbers = mb.getNumRows() > 500000 && mb.getSparsity() < 0.4;
					boolean isAboveThreadToColumnRatio = compressionGroups.getNumberColGroups() > mb.getNumColumns() / 30;
					compSettings.transposed = haveManyColumns || isNnzLowAndVerySparse ||
						(isAboveRowNumbers && isAboveThreadToColumnRatio);
				}
				else
					compSettings.transposed = false;
		}
	}

	private void compressPhase() {
		List<AColGroup> c = ColGroupFactory.compressColGroups(mb, compressionGroups, compSettings, costEstimator, k);
		res.allocateColGroupList(c);
		_stats.compressedInitialSize = res.getInMemorySize();
		logPhase();
	}

	private void finalizePhase() {
		res.cleanupBlock(true, true);

		_stats.compressedSize = res.getInMemorySize();
		_stats.compressedCost = costEstimator.getCost(res.getColGroups(), res.getNumRows());

		final double ratio = _stats.getRatio();
		final double denseRatio = _stats.getDenseRatio();

		if(ratio < 1 && denseRatio < 100.0) {
			LOG.info("--dense size:        " + _stats.denseSize);
			LOG.info("--original size:     " + _stats.originalSize);
			LOG.info("--compressed size:   " + _stats.compressedSize);
			LOG.info("--compression ratio: " + ratio);
			LOG.info("Abort block compression because compression ratio is less than 1.");
			res = null;
			setNextTimePhase(time.stop());
			DMLCompressionStatistics.addCompressionTime(getLastTimePhase(), phase);
			return;
		}

		_stats.setColGroupsCounts(res.getColGroups());

		if(compSettings.isInSparkInstruction)
			res.clearSoftReferenceToDecompressed();

		final long oldNNZ = mb.getNonZeros();
		if(oldNNZ <= 0L)
			res.recomputeNonZeros();
		else
			res.setNonZeros(oldNNZ);

		logPhase();
	}

	private Pair<MatrixBlock, CompressionStatistics> abortCompression() {
		LOG.warn("Compression aborted at phase: " + phase);

		if(compSettings.transposed)
			LibMatrixReorg.transposeInPlace(mb, k);

		return new ImmutablePair<>(mb, _stats);
	}

	private void logPhase() {
		setNextTimePhase(time.stop());
		DMLCompressionStatistics.addCompressionTime(getLastTimePhase(), phase);
		if(LOG.isDebugEnabled()) {
			if(compSettings.isInSparkInstruction) {
				if(phase == 4)
					LOG.debug(_stats);
			}
			else {
				switch(phase) {
					case 0:
						LOG.debug("--Seed used for comp : " + compSettings.seed);
						LOG.debug("--compression phase " + phase + " Classify  : " + getLastTimePhase());
						LOG.debug("--Individual Columns Estimated Compression: " + _stats.estimatedSizeCols);
						break;
					case 1:
						LOG.debug("--compression phase " + phase + " Grouping  : " + getLastTimePhase());
						LOG.debug("Grouping using: " + compSettings.columnPartitioner);
						LOG.debug("Cost Calculated using: " + costEstimator);
						LOG.debug("--Cocoded Columns estimated Compression:" + _stats.estimatedSizeCoCoded);
						if(compressionGroups.getInfo().size() < 1000) {
							LOG.debug("--Cocoded Columns estimated nr distinct:" + compressionGroups.getEstimatedDistinct());
							LOG.debug("--Cocoded Columns nr columns           :" + compressionGroups.getNrColumnsString());
						}
						else {
							LOG.debug(
								"--CoCoded produce many columns but the first says:\n" + compressionGroups.getInfo().get(0));
						}
						break;
					case 2:
						LOG.debug("--compression phase " + phase + " Transpose : " + getLastTimePhase());
						LOG.debug("Did transpose: " + compSettings.transposed);
						break;
					case 3:
						LOG.debug("--compression phase " + phase + " Compress  : " + getLastTimePhase());
						LOG.debug("--compressed initial actual size:" + _stats.compressedInitialSize);
						break;
					case 4:
						LOG.debug("--num col groups:    " + res.getColGroups().size());
						LOG.debug("--compression phase  " + phase + " Cleanup   : " + getLastTimePhase());
						LOG.debug("--col groups types   " + _stats.getGroupsTypesString());
						LOG.debug("--col groups sizes   " + _stats.getGroupsSizesString());
						LOG.debug(String.format("--dense size:        %16d", _stats.denseSize));
						LOG.debug(String.format("--original size:     %16d", _stats.originalSize));
						LOG.debug(String.format("--compressed size:   %16d", _stats.compressedSize));
						LOG.debug(String.format("--compression ratio: %4.3f", _stats.getRatio()));
						LOG.debug(String.format("--Dense       ratio: %4.3f", _stats.getDenseRatio()));
						if(!(costEstimator instanceof MemoryCostEstimator)) {
							LOG.debug(String.format("--original cost:     %5.2E", _stats.originalCost));
							LOG.debug(String.format("--single col cost:   %5.2E", _stats.estimatedCostCols));
							LOG.debug(String.format("--cocode cost:       %5.2E", _stats.estimatedCostCoCoded));
							LOG.debug(String.format("--actual cost:       %5.2E", _stats.compressedCost));
							LOG.debug(
								String.format("--relative cost:     %1.4f", (_stats.compressedCost / _stats.originalCost)));
						}
						if(compressionGroups.getInfo().size() < 1000) {
							int[] lengths = new int[res.getColGroups().size()];
							int i = 0;
							for(AColGroup colGroup : res.getColGroups())
								lengths[i++] = colGroup.getNumValues();

							LOG.debug("--compressed colGroup dictionary sizes: " + Arrays.toString(lengths));
							LOG.debug(
								"--compressed colGroup nr columns      : " + constructNrColumnString(res.getColGroups()));
						}
						if(LOG.isTraceEnabled()) {
							for(AColGroup colGroup : res.getColGroups()) {
								if(colGroup.estimateInMemorySize() < 1000)
									LOG.trace(colGroup);
								else {
									LOG.trace("--colGroups type       : " + colGroup.getClass().getSimpleName() + " size: "
										+ colGroup.estimateInMemorySize()
										+ ((colGroup instanceof AColGroupValue) ? "  numValues :"
											+ ((AColGroupValue) colGroup).getNumValues() : "")
										+ "  colIndexes : " + Arrays.toString(colGroup.getColIndices()));
								}
							}
						}
					default:
				}
			}
		}
		phase++;
	}

	private void setNextTimePhase(double time) {
		lastPhase = time;
	}

	private double getLastTimePhase() {
		return lastPhase;
	}

	private Pair<MatrixBlock, CompressionStatistics> createEmpty() {
		LOG.info("Empty input to compress, returning a compressed Matrix block with empty column group");
		CompressedMatrixBlock ret = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
		ColGroupEmpty cg = ColGroupEmpty.create(mb.getNumColumns());
		ret.allocateColGroup(cg);
		ret.setNonZeros(0);
		return new ImmutablePair<>(ret, null);
	}

	private Pair<MatrixBlock, CompressionStatistics> returnSelf() {
		LOG.info("MatrixBlock already compressed or is Empty");
		return new ImmutablePair<>(mb, null);
	}

	private static String constructNrColumnString(List<AColGroup> cg) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		sb.append(cg.get(0).getNumCols());
		for(int id = 1; id < cg.size(); id++)
			sb.append(", " + cg.get(id).getNumCols());
		sb.append("]");
		return sb.toString();
	}
}
