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

package org.apache.sysds.runtime.transform.encode;

import static org.apache.sysds.utils.MemoryEstimates.intArrayCost;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.estim.ComEstSample;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;
import org.apache.sysds.runtime.util.DependencyWrapperTask;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.stats.TransformStatistics;

public class MultiColumnEncoder implements Encoder {

	protected static final Log LOG = LogFactory.getLog(MultiColumnEncoder.class.getName());
	// If true build and apply separately by placing a synchronization barrier
	public static boolean MULTI_THREADED_STAGES = ConfigurationManager.isStagedParallelTransform();

	// Only affects if  MULTI_THREADED_STAGES is true
	// if true apply tasks for each column will complete
	// before the next will start.
	public static boolean APPLY_ENCODER_SEPARATE_STAGES = false; 

	private List<ColumnEncoderComposite> _columnEncoders;
	// These encoders are deprecated and will be phased out soon.
	private EncoderMVImpute _legacyMVImpute = null;
	private EncoderOmit _legacyOmit = null;
	private int _colOffset = 0; // offset for federated Workers who are using subrange encoders
	private FrameBlock _meta = null;
	private boolean _partitionDone = false;

	public MultiColumnEncoder(List<ColumnEncoderComposite> columnEncoders) {
		_columnEncoders = columnEncoders;
	}

	public MultiColumnEncoder(MultiColumnEncoder menc) {
		// This constructor creates a shallow copy for all encoders except for bag_of_words encoders
		List<ColumnEncoderComposite> colEncs = menc._columnEncoders;
		_columnEncoders= new ArrayList<>();
		for (ColumnEncoderComposite cColEnc : colEncs) {
			List<ColumnEncoder> newEncs = new ArrayList<>();
			ColumnEncoderComposite cColEncCopy = new ColumnEncoderComposite(newEncs, cColEnc._colID);
			_columnEncoders.add(cColEncCopy);
			for (ColumnEncoder enc : cColEnc.getEncoders()) {
				newEncs.add(enc instanceof ColumnEncoderBagOfWords ? new ColumnEncoderBagOfWords((ColumnEncoderBagOfWords) enc) : enc);
			}
		}
	}

	public MultiColumnEncoder() {
		_columnEncoders = new ArrayList<>();
	}

	public MatrixBlock encode(CacheBlock<?> in) {
		return encode(in, 1);
	}

	public MatrixBlock encode(CacheBlock<?> in, int k) {
		return encode(in, k, false);
	}

	public MatrixBlock encode(CacheBlock<?> in, boolean compressedOut) {
		return encode(in, 1, compressedOut);
	}

	public MatrixBlock encode(CacheBlock<?> in, int k, boolean compressedOut){
		try {
			if(isCompressedTransformEncode(in, compressedOut))
				return CompressedEncode.encode(this, (FrameBlock ) in, k);

			deriveNumRowPartitions(in, k);
			if(k > 1 && !MULTI_THREADED_STAGES && !hasLegacyEncoder()) {
				MatrixBlock out = new MatrixBlock();
				DependencyThreadPool pool = new DependencyThreadPool(k);
				LOG.debug("Encoding with full DAG on " + k + " Threads");
				try {
					List<DependencyTask<?>> tasks = getEncodeTasks(in, out, pool);
					pool.submitAllAndWait(tasks);
				}
				finally{
					pool.shutdown();
				}
				outputMatrixPostProcessing(out, k);
				outputLogging(out);
				return out;
			}
			else {
				LOG.debug("Encoding with staged approach on: " + k + " Threads");
				long t0 = System.nanoTime();
				build(in, k);
				long t1 = System.nanoTime();
				LOG.debug("Elapsed time for build phase: "+ ((double) t1 - t0) / 1000000 + " ms");
				if(_legacyMVImpute != null) {
					// These operations are redundant for every encoder excluding the legacyMVImpute, the workaround to
					// fix it for this encoder would be very dirty. This will only have a performance impact if there
					// is a lot of recoding in combination with the legacyMVImpute.
					// But since it is legacy this should be fine
					_meta = getMetaData(new FrameBlock(in.getNumColumns(), Types.ValueType.STRING));
					initMetaData(_meta);
				}
				// apply meta data
				t0 = System.nanoTime();
				MatrixBlock out = apply(in, k);
				t1 = System.nanoTime();
				LOG.debug("Elapsed time for apply phase: "+ ((double) t1 - t0) / 1000000 + " ms");

				outputLogging(out);
				return out;
			}
		}
		catch(Exception ex) {
			String st = this.toString();
			st = st.substring(0, Math.min(st.length(), 1000));
			throw new DMLRuntimeException("Failed transform-encode frame with encoder:\n" + st, ex);
		}
	}

	private void outputLogging(MatrixBlock out) {
		if(LOG.isDebugEnabled()) {
			LOG.debug("Transform Encode output mem size: " + out.getInMemorySize());
			LOG.debug(String.format("Transform Encode output rows     : %10d", out.getNumRows()));
			LOG.debug(String.format("Transform Encode output cols     : %10d", out.getNumColumns()));
			LOG.debug(String.format("Transform Encode output sparsity : %10.5f", out.getSparsity()));
			LOG.debug(String.format("Transform Encode output nnz      : %10d", out.getNonZeros()));
		}
	}

	protected List<ColumnEncoderComposite> getEncoders() {
		return _columnEncoders;
	}

	/* TASK DETAILS:
	 * InitOutputMatrixTask:        Allocate output matrix
	 * AllocMetaTask:               Allocate metadata frame
	 * BuildTask:                   Build an encoder
	 * ColumnCompositeUpdateDCTask: Update domain size of a DC encoder based on #distincts, #bins, K
	 * ColumnMetaDataTask:          Fill up metadata of an encoder
	 * ApplyTasksWrapperTask:       Wrapper task for an Apply task
	 * UpdateOutputColTask:         Set starting offsets of the DC columns
	 */
	private List<DependencyTask<?>> getEncodeTasks(CacheBlock<?> in, MatrixBlock out, DependencyThreadPool pool) {
		List<DependencyTask<?>> tasks = new ArrayList<>();
		List<DependencyTask<?>> applyTAgg = null;
		Map<Integer[], Integer[]> depMap = new HashMap<>();
		boolean hasDC = !getColumnEncoders(ColumnEncoderDummycode.class).isEmpty();
		boolean hasBOW = !getColumnEncoders(ColumnEncoderBagOfWords.class).isEmpty();
		boolean applyOffsetDep = false;
		boolean independentUpdateDC = false;
		_meta = new FrameBlock(in.getNumColumns(), ValueType.STRING);
		// Create the output and metadata allocation tasks
		tasks.add(DependencyThreadPool.createDependencyTask(new InitOutputMatrixTask(this, in, out)));
		tasks.add(DependencyThreadPool.createDependencyTask(new AllocMetaTask(this, _meta)));

		for(ColumnEncoderComposite e : _columnEncoders) {
			// Create the build tasks
			List<DependencyTask<?>> buildTasks = e.getBuildTasks(in);
			tasks.addAll(buildTasks);
			boolean compositeHasDC = e.hasEncoder(ColumnEncoderDummycode.class);
			boolean compositeHasBOW = e.hasEncoder(ColumnEncoderBagOfWords.class);
			if(!buildTasks.isEmpty()) {
				// Check if any Build independent UpdateDC task (Bin+DC, FH+DC)
				if (compositeHasDC
					&& buildTasks.size() > 1  //filter out FH
					&& !buildTasks.get(buildTasks.size()-2).hasDependency(buildTasks.get(buildTasks.size()-1)))
						independentUpdateDC = true;
				
				// Independent UpdateDC task
				if (independentUpdateDC) {
					// Apply Task depends on task prior to UpdateDC (Build/MergePartialBuild)
					depMap.put(new Integer[] {tasks.size(), tasks.size() + 1},     //ApplyTask
						new Integer[] {tasks.size() - 2, tasks.size() - 1});       //BuildTask
					// getMetaDataTask depends on task prior to UpdateDC 
					depMap.put(new Integer[] {tasks.size() + 1, tasks.size() + 2}, //MetaDataTask
						new Integer[] {tasks.size() - 2, tasks.size() - 1});       //BuildTask
				}
				else { 
					// Apply Task depends on the last task (Build/MergePartial/UpdateDC)
					depMap.put(new Integer[] {tasks.size(), tasks.size() + 1},     //ApplyTask
						new Integer[] {tasks.size() - 1, tasks.size()});           //Build/UpdateDC
					// getMetaDataTask depends on build completion
					depMap.put(new Integer[] {tasks.size() + 1, tasks.size() + 2}, //MetaDataTask
						new Integer[] {tasks.size() - 1, tasks.size()});           //Build/UpdateDC
				}
				// AllocMetaTask never depends on the UpdateDC task
				if (compositeHasDC && buildTasks.size() > 1)
					depMap.put(new Integer[] {1, 2},                               //AllocMetaTask (2nd task)
						new Integer[] {tasks.size() - 2, tasks.size()-1});         //BuildTask
				else
					depMap.put(new Integer[] {1, 2},                               //AllocMetaTask (2nd task)
						new Integer[] {tasks.size() - 1, tasks.size()});           //BuildTask
			}

			// getMetaDataTask depends on AllocMeta task
			depMap.put(new Integer[] {tasks.size() + 1, tasks.size() + 2},     //MetaDataTask
				new Integer[] {1, 2});                                         //AllocMetaTask (2nd task)

			// Apply Task depends on InitOutputMatrixTask (output allocation)
			depMap.put(new Integer[] {tasks.size(), tasks.size() + 1},         //ApplyTask
					new Integer[] {0, 1});                                     //Allocation task (1st task)
			ApplyTasksWrapperTask applyTaskWrapper = new ApplyTasksWrapperTask(e, in, out, pool);

			if(compositeHasDC || compositeHasBOW) {
				// Allocation depends on build if DC or BOW is in the list.
				// Note, DC is the only encoder that changes dimensionality
				depMap.put(new Integer[]{0, 1},                               //Allocation task (1st task)
						new Integer[]{tasks.size() - 1, tasks.size()});       //BuildTask
			}
			if(compositeHasDC || compositeHasBOW){
				// UpdateOutputColTask, that sets the starting offsets of the DC columns,
				// depends on the Build completion tasks
				depMap.put(new Integer[] {-2, -1},                             //UpdateOutputColTask (last task) 
						new Integer[] {tasks.size() - 1, tasks.size()});       //BuildTask
				buildTasks.forEach(t -> t.setPriority(5));
				applyOffsetDep = true;
			}

			if((hasDC || hasBOW) && applyOffsetDep) {
				// Apply tasks depend on UpdateOutputColTask
				depMap.put(new Integer[] {tasks.size(), tasks.size() + 1},     //ApplyTask 
						new Integer[] {-2, -1});                               //UpdateOutputColTask (last task)

				applyTAgg = applyTAgg == null ? new ArrayList<>() : applyTAgg;
				applyTAgg.add(applyTaskWrapper);
			}
			else {
				applyTaskWrapper.setOffset(0);
			}
			// Create the ApplyTask (wrapper)
			tasks.add(applyTaskWrapper);
			// Create the getMetadata task
			tasks.add(DependencyThreadPool.createDependencyTask(new ColumnMetaDataTask<ColumnEncoder>(e, _meta)));
		}
		if(hasDC || hasBOW)
			// Create the last task, UpdateOutputColTask
			tasks.add(DependencyThreadPool.createDependencyTask(new UpdateOutputColTask(this, applyTAgg)));

		List<List<? extends Callable<?>>> deps = new ArrayList<>(Collections.nCopies(tasks.size(), null));
		DependencyThreadPool.createDependencyList(tasks, depMap, deps);
		return DependencyThreadPool.createDependencyTasks(tasks, deps);
	}

	public void build(CacheBlock<?> in) {
		build(in, 1);
	}

	public void build(CacheBlock<?> in, int k) {
		build(in, k, null);
	}

	public void build(CacheBlock<?> in, int k, Map<Integer, double[]> equiHeightBinMaxs) {
		if(hasLegacyEncoder() && !(in instanceof FrameBlock))
			throw new DMLRuntimeException("LegacyEncoders do not support non FrameBlock Inputs");
		if(!_partitionDone) //happens if this method is directly called
			deriveNumRowPartitions(in, k);
		if(k > 1) {
			buildMT(in, k);
		}
		else {
			for(ColumnEncoderComposite columnEncoder : _columnEncoders) {
				columnEncoder.build(in, equiHeightBinMaxs);
				columnEncoder.updateAllDCEncoders();
			}
		}
		if(hasLegacyEncoder())
			legacyBuild((FrameBlock) in);
	}

	private List<DependencyTask<?>> getBuildTasks(CacheBlock<?> in) {
		List<DependencyTask<?>> tasks = new ArrayList<>();
		for(ColumnEncoderComposite columnEncoder : _columnEncoders) {
			tasks.addAll(columnEncoder.getBuildTasks(in));
		}
		return tasks;
	}

	private void buildMT(CacheBlock<?> in, int k) {
		DependencyThreadPool pool = new DependencyThreadPool(k);
		try {
			pool.submitAllAndWait(getBuildTasks(in));
		}
		catch(ExecutionException | InterruptedException e) {
			throw new RuntimeException(e);
		}
		finally{
			pool.shutdown();
		}
	}

	public void legacyBuild(FrameBlock in) {
		if(_legacyOmit != null)
			_legacyOmit.build(in);
		if(_legacyMVImpute != null)
			_legacyMVImpute.build(in);
	}


	public MatrixBlock apply(CacheBlock<?> in) {
		return apply(in, 1);
	}

	public MatrixBlock apply(CacheBlock<?> in, int k) {
		// domain sizes are not updated if called from transformapply
		EncoderMeta encm = getEncMeta(_columnEncoders, true, k, in);
		updateAllDCEncoders();
		int numCols = getNumOutCols();
		long estNNz = (long) in.getNumRows() * (encm.hasWE || encm.hasUDF ? numCols : (in.getNumColumns() - encm.numBOWEnc) + encm.nnzBOW);
		// FIXME: estimate nnz for multiple encoders including dummycode
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(in.getNumRows(), numCols, estNNz) && !encm.hasUDF;
		MatrixBlock out = new MatrixBlock(in.getNumRows(), numCols, sparse, estNNz);
		return apply(in, out, 0, k, encm, estNNz);
	}

	public void updateAllDCEncoders(){
		for(ColumnEncoderComposite columnEncoder : _columnEncoders)
			columnEncoder.updateAllDCEncoders();
	}

	public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol) {
		// unused method, only exists currently because of the interface
		throw new DMLRuntimeException("MultiColumnEncoder apply without Encoder Characteristics should not be called directly");
	}

	public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol, int k, EncoderMeta encm, long nnz) {
		// There should be a encoder for every column
		if(hasLegacyEncoder() && !(in instanceof FrameBlock))
			throw new DMLRuntimeException("LegacyEncoders do not support non FrameBlock Inputs");
		int numEncoders = getEncoders().size();
		// getFromAll(ColumnEncoderComposite.class, ColumnEncoder::getColID).size();
		if(in.getNumColumns() != numEncoders)
			throw new DMLRuntimeException("Not every column in has a CompositeEncoder. Please make sure every column "
				+ "has a encoder or slice the input accordingly: num encoders:  " + getEncoders()  + " vs columns " + in.getNumColumns());
		// TODO smart checks
		// Block allocation for MT access
		if(in.getNumRows() == 0)
			throw new DMLRuntimeException("Invalid input with wrong number or rows");

		ArrayList<int[]> nnzOffsets = outputMatrixPreProcessing(out, in, encm, nnz, k);
		if(k > 1) {
			if(!_partitionDone) //happens if this method is directly called
				deriveNumRowPartitions(in, k);
			applyMT(in, out, outputCol, k, nnzOffsets);
		}
		else {
			int offset = outputCol, i = 0;
			int[] nnzOffset = null;
			for(ColumnEncoderComposite columnEncoder : _columnEncoders) {
				columnEncoder.sparseRowPointerOffset = nnzOffset;
				columnEncoder.apply(in, out, columnEncoder._colID - 1 + offset);
				offset = getOutputColOffset(offset, columnEncoder);
				nnzOffset = nnzOffsets != null ? nnzOffsets.get(i++) : null;
			}
		}
		// Recomputing NNZ since we access the block directly
		// TODO set NNZ explicit count them in the encoders
		outputMatrixPostProcessing(out, k);
		if(_legacyOmit != null)
			out = _legacyOmit.apply((FrameBlock) in, out);
		if(_legacyMVImpute != null)
			out = _legacyMVImpute.apply((FrameBlock) in, out);

		return out;
	}

	private List<DependencyTask<?>> getApplyTasks(CacheBlock<?> in, MatrixBlock out, int outputCol, ArrayList<int[]> nnzOffsets) {
		List<DependencyTask<?>> tasks = new ArrayList<>();
		int offset = outputCol;
		int i = 0;
		int[] currentNnzOffsets = null;
		for(ColumnEncoderComposite e : _columnEncoders) {
			tasks.addAll(e.getApplyTasks(in, out, e._colID - 1 + offset, currentNnzOffsets));
			currentNnzOffsets = nnzOffsets != null ? nnzOffsets.get(i++) : null;
			offset = getOutputColOffset(offset, e);
		}
		return tasks;
	}

	private int getOutputColOffset(int offset, ColumnEncoderComposite e) {
		if(e.hasEncoder(ColumnEncoderDummycode.class))
			offset += e.getEncoder(ColumnEncoderDummycode.class)._domainSize - 1;
		if(e.hasEncoder(ColumnEncoderWordEmbedding.class))
			offset += e.getEncoder(ColumnEncoderWordEmbedding.class).getDomainSize() - 1;
		if(e.hasEncoder(ColumnEncoderBagOfWords.class))
			offset += e.getEncoder(ColumnEncoderBagOfWords.class).getDomainSize() - 1;
		return offset;
	}

	private void applyMT(CacheBlock<?> in, MatrixBlock out, int outputCol, int k, ArrayList<int[]> nnzOffsets) {
		DependencyThreadPool pool = new DependencyThreadPool(k);
		try {
			if(APPLY_ENCODER_SEPARATE_STAGES) {
				int offset = outputCol;
				int i = 0;
				int[] currentNnzOffsets = null;
				for (ColumnEncoderComposite e : _columnEncoders) {
					pool.submitAllAndWait(e.getApplyTasks(in, out, e._colID - 1 + offset, currentNnzOffsets));
					offset = getOutputColOffset(offset, e);
					currentNnzOffsets = nnzOffsets != null ? nnzOffsets.get(i) : null;
					i++;
				}
			} else
				pool.submitAllAndWait(getApplyTasks(in, out, outputCol, nnzOffsets));
		}
		catch(ExecutionException | InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
		}
	}

	private void deriveNumRowPartitions(CacheBlock<?> in, int k) {
		int[] numBlocks = new int[2];
		if (k == 1) { //single-threaded
			numBlocks[0] = 1;
			numBlocks[1] = 1;
			_columnEncoders.forEach(e -> e.setNumPartitions(1, 1));
			_partitionDone = true;
			return;
		}
		// Read from global flags. These are set by the unit tests
		if (ColumnEncoder.BUILD_ROW_BLOCKS_PER_COLUMN > 0)
			numBlocks[0] = ColumnEncoder.BUILD_ROW_BLOCKS_PER_COLUMN;
		if (ColumnEncoder.APPLY_ROW_BLOCKS_PER_COLUMN > 0)
			numBlocks[1] = ColumnEncoder.APPLY_ROW_BLOCKS_PER_COLUMN;

		// Read from the config file if set. These overwrite the derived values.
		if (numBlocks[0] == 0 && ConfigurationManager.getParallelBuildBlocks() > 0)
			numBlocks[0] = ConfigurationManager.getParallelBuildBlocks();
		if (numBlocks[1] == 0 && ConfigurationManager.getParallelApplyBlocks() > 0)
			numBlocks[1] = ConfigurationManager.getParallelApplyBlocks();

		// Else, derive the optimum number of partitions
		int nRow = in.getNumRows();
		int nThread = OptimizerUtils.getTransformNumThreads(); //VCores
		int minNumRows = 16000; //min rows per partition
		List<ColumnEncoderComposite> recodeEncoders = new ArrayList<>();
		List<ColumnEncoderComposite> bowEncoders = new ArrayList<>();
		// Count #Builds and #Applies (= #Col)
		int nBuild = 0;
		for (ColumnEncoderComposite e : _columnEncoders)
			if (e.hasBuild()) {
				nBuild++;
				if (e.hasEncoder(ColumnEncoderRecode.class))
					recodeEncoders.add(e);
				if (e.hasEncoder(ColumnEncoderBagOfWords.class))
					bowEncoders.add(e);
			}
		int nApply = in.getNumColumns();
		// #BuildBlocks = (2 * #PhysicalCores)/#build
		if (numBlocks[0] == 0 && nBuild > 0 && nBuild < nThread)
			numBlocks[0] = Math.round(((float)nThread)/nBuild);
		// #ApplyBlocks = (4 * #PhysicalCores)/#apply
		if (numBlocks[1] == 0 && nApply > 0 && nApply < nThread*2)
			numBlocks[1] = Math.round(((float)nThread*2)/nApply);

		int bowNumBuildBlks = numBlocks[0];
		int bowNumApplyBlks = numBlocks[1];

		// Reduce #blocks if #rows per partition is too small
		// while (numBlocks[0] > 1 && nRow/numBlocks[0] < minNumRows)
		// 	 numBlocks[0]--;
		// while (numBlocks[1] > 1 && nRow/numBlocks[1] < minNumRows)
		//	 numBlocks[1]--;
		// the two while loop should be equal to following code:
		int optimalPartitions = Math.max(1, nRow / minNumRows);
		numBlocks[0] = Math.min(numBlocks[0], optimalPartitions);
		numBlocks[1] = Math.min(numBlocks[1], optimalPartitions);
		int rcdNumBuildBlks = numBlocks[0];

		// Use a smaller minNumRows for BOW encoders because of a larger computational overhead per row
		optimalPartitions = Math.max(1, nRow / (minNumRows / 16));
		bowNumBuildBlks = Math.min(bowNumBuildBlks, optimalPartitions);
		bowNumApplyBlks = Math.min(bowNumApplyBlks, optimalPartitions);


		// RC: Reduce #build blocks for all encoders if all don't fit in memory
		if (numBlocks[0] > 1 && !recodeEncoders.isEmpty() && bowEncoders.isEmpty()) {
			rcdNumBuildBlks = getNumBuildBlksMemorySafe(in, recodeEncoders, rcdNumBuildBlks, false);
		}
		// BOW: Reduce #build blocks for all encoders if all don't fit in memory
		else if (bowNumBuildBlks > 1 && recodeEncoders.isEmpty() && !bowEncoders.isEmpty()) {
			bowNumBuildBlks = getNumBuildBlksMemorySafe(in, bowEncoders, bowNumBuildBlks, true);
		}
		// RC + BOW: check if all encoders fit into memory
		else if (bowNumBuildBlks > 1 || rcdNumBuildBlks > 1) {
			// Estimate map sizes, fused with other encoders (bag_of_words)
			List<List<ColumnEncoderComposite>> encoders = new ArrayList<>();
			encoders.add(recodeEncoders);
			encoders.add(bowEncoders);
			int[] bldBlks = new int[]{rcdNumBuildBlks, bowNumBuildBlks};
			getNumBuildBlksMixedEncMemorySafe(in, encoders, bldBlks);
			rcdNumBuildBlks = bldBlks[0];
			bowNumBuildBlks = bldBlks[1];
		}
		// TODO: If still don't fit, serialize the column encoders

		// Set to 1 if not set by the above logics
		for (int i=0; i<2; i++)
			if (numBlocks[i] == 0)
				numBlocks[i] = 1; //default 1

		_partitionDone = true;
		// Materialize the partition counts in the encoders
		_columnEncoders.forEach(e -> e.setNumPartitions(numBlocks[0], numBlocks[1]));
		if (rcdNumBuildBlks > 0 && rcdNumBuildBlks != numBlocks[0]) {
			int rcdNumBlocks = rcdNumBuildBlks;
			recodeEncoders.forEach(e -> e.setNumPartitions(rcdNumBlocks, numBlocks[1]));
		}
		if (bowNumBuildBlks > 0) {
			final int bowNumBlocks = bowNumBuildBlks;
			final int bowApplyBlks = bowNumApplyBlks;
			bowEncoders.forEach(e -> e.setNumPartitions(bowNumBlocks, bowApplyBlks));
		}
		//System.out.println("Block count = ["+numBlocks[0]+", "+numBlocks[1]+"], Recode block count = "+rcdNumBuildBlks);
	}

	private int getNumBuildBlksMemorySafe(CacheBlock<?> in, List<ColumnEncoderComposite> encoders, int numBldBlks, boolean hasBOW) {
		estimateMapSize(in, encoders);
		// Memory budget for maps = 70% of heap - sizeof(input)
		long memBudget = (long) (OptimizerUtils.getLocalMemBudget() - in.getInMemorySize());
		if(hasBOW){
			// integer arrays: nnzPerRow for each bow encoder
			memBudget -= encoders.size()*(long) intArrayCost(in.getNumRows());
		}
		// Worst case scenario: all partial maps contain all distinct values (if < #rows)
		long totMemOverhead = getTotalMemOverhead(in, numBldBlks, encoders);

		// Reduce recode build blocks count till they fit in the memory budget
		while (numBldBlks > 1 && totMemOverhead > memBudget) {
			numBldBlks--;
			totMemOverhead = getTotalMemOverhead(in, numBldBlks, encoders);
			// TODO: Reduce only the ones with large maps
		}
		return numBldBlks;
	}

	private void getNumBuildBlksMixedEncMemorySafe(CacheBlock<?> in, List<List<ColumnEncoderComposite>> encs, int[] blks) {
		// Memory budget for maps = 70% of heap - sizeof(input)
		long memBudget = (long) (OptimizerUtils.getLocalMemBudget() - in.getInMemorySize());
		// integer arrays: nnzPerRow for each bow encoder
		memBudget -= encs.get(1).size()*((long) intArrayCost(in.getNumRows()));

		int numOfEncTypes = encs.size();
		long[] totMemOverhead = new long[numOfEncTypes];
		for (int i = 0; i < numOfEncTypes; i++) {
			estimateMapSize(in, encs.get(i));
			// Worst case scenario: all partial maps contain all distinct values (if < #rows)
			totMemOverhead[i] = getTotalMemOverhead(in, blks[i], encs.get(i));
		}

		int next = blks[1] > 1 ? 1 : 0;
		// round-robin reducing
		int skipped = 0;
		while (skipped != numOfEncTypes && Arrays.stream(totMemOverhead).sum() > memBudget) {
			if(blks[next] > 1){
				blks[next]--;
				totMemOverhead[next] = getTotalMemOverhead(in, blks[next], encs.get(next));
				next = (next + 1) % numOfEncTypes;
				skipped = 0;
			} else
				skipped++;
		}
		// TODO: Reduce the large encoder types, similar to getNumBuildBlksMemorySafe
	}

// not used rn, commented because of missing code coverage
//	private long estimateSparseOutputSize(List<ColumnEncoderComposite> bowEncs, int nApply, int nRows){
//		// #rows x (#col - #bowEncs + bow-avg-nnz)
//		double avgNnzPerRow = 0.0;
//		for (ColumnEncoderComposite enc : bowEncs){
//			ColumnEncoderBagOfWords bow = enc.getEncoder(ColumnEncoderBagOfWords.class);
//			avgNnzPerRow += bow.avgNnzPerRow;
//		}
//		long nnzBow = (long) (avgNnzPerRow*nRows);
//		long nnzOther = (long) nRows *(nApply - bowEncs.size());
//		long nnz = nnzBow + nnzOther;
//		return estimateSizeInMemory(nRows, nnz);
//	}

	private void estimateMapSize(CacheBlock<?> in, List<ColumnEncoderComposite> encList) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Collect sample row indices
		int k = OptimizerUtils.getTransformNumThreads();
		int[] sampleInds = getSampleIndices(in, (int) (0.1 * in.getNumRows()), (int) System.nanoTime(), 1);

		// Concurrent (column-wise) recode map size estimation
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			pool.submit(() -> {
				encList.stream().parallel().forEach(e -> {
					e.computeMapSizeEstimate(in, sampleInds);
				});
			}).get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}

		if(DMLScript.STATISTICS) {
			LOG.debug("Elapsed time for encoder map size estimation: " + ((double) System.nanoTime() - t0) / 1000000 + " ms");
			TransformStatistics.incMapSizeEstimationTime(System.nanoTime() - t0);
		}
	}

	private static int[] getSampleIndices(CacheBlock<?> in, int sampleSize, int seed, int k){
		return ComEstSample.getSortedSample(in.getNumRows(), sampleSize, seed, k);
	}

	// Estimate total memory overhead of the partial recode maps of all recoders
	private long getTotalMemOverhead(CacheBlock<?> in, int nBuildpart, List<ColumnEncoderComposite> encoders) {
		long totMemOverhead = 0;
		if (nBuildpart == 1) {
			// Sum the estimated map sizes
			totMemOverhead = encoders.stream().mapToLong(ColumnEncoderComposite::getEstMetaSize).sum();
			return totMemOverhead;
		}
		// Estimate map size of each partition and sum
		for (ColumnEncoderComposite enc : encoders) {
			int partSize = in.getNumRows()/nBuildpart;
			int partNumDist = Math.min(partSize, enc.getEstNumDistincts()); //#distincts not more than #rows
			if(enc.getAvgEntrySize() == 0)
				throw new DMLRuntimeException("Error while estimating entry size of encoder map");
			long allMapsSize = partNumDist * enc.getAvgEntrySize() * nBuildpart; //worst-case scenario
			totMemOverhead += allMapsSize;
		}
		return totMemOverhead;
	}

	private static ArrayList<int[]> outputMatrixPreProcessing(MatrixBlock output, CacheBlock<?> input, EncoderMeta encm, long nnz, int k) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(nnz < 0)
			nnz = (long) output.getNumRows() * input.getNumColumns();
		ArrayList<int[]> bowNnzRowOffsets = null;
		if(output.isInSparseFormat()) {
			if (MatrixBlock.DEFAULT_SPARSEBLOCK != SparseBlock.Type.CSR
					&& MatrixBlock.DEFAULT_SPARSEBLOCK != SparseBlock.Type.MCSR)
				throw new RuntimeException("Transformapply is only supported for MCSR and CSR output matrix");
			//boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;
			boolean mcsr = false; //force CSR for transformencode
			if (mcsr) {
				output.allocateBlock();
				SparseBlock block = output.getSparseBlock();
				if (encm.hasDC && OptimizerUtils.getTransformNumThreads()>1) {
					// DC forces a single threaded allocation after the build phase and
					// before the apply starts. Below code parallelizes sparse allocation.
					IntStream.range(0, output.getNumRows())
					.parallel().forEach(r -> {
						block.allocate(r, input.getNumColumns());
						((SparseRowVector)block.get(r)).setSize(input.getNumColumns());
					});
				}
				else {
					for(int r = 0; r < output.getNumRows(); r++) {
						// allocate all sparse rows so MT sync can be done.
						// should be rare that rows have only 0
						block.allocate(r, input.getNumColumns());
						// Setting the size here makes it possible to run all sparse apply tasks without any sync
						// could become problematic if the input is very sparse since we allocate the same size as the input
						// should be fine in theory ;)
						((SparseRowVector)block.get(r)).setSize(input.getNumColumns());
					}
				}
			}
			else { //csr
				// Manually fill the row pointers based on nnzs/row (= #cols in the input)
				// Not using the set() methods to 1) avoid binary search and shifting, 
				// 2) reduce thread contentions on the arrays
				int nnzInt = (int) nnz;
				int[] rptr = new int[output.getNumRows()+1];
				// easy case: no bow encoders
				// nnz per row = #encoders = #inputCols
				if(encm.numBOWEnc <= 0 )
					for (int i = 0; i < rptr.length - 1; i++)
						rptr[i + 1] = rptr[i] + input.getNumColumns();
				else {
					if( encm.nnzPerRowBOW != null) {
						// #nzPerRow has been already computed and aggregated for all bow encoders
						int static_offset = input.getNumColumns() - encm.numBOWEnc;
						// - #bow since the nnz are already counted
						for (int i = 0; i < rptr.length - 1; i++) {
							int nnzPerRow = static_offset + encm.nnzPerRowBOW[i];
							rptr[i + 1] = rptr[i] + nnzPerRow;
						}
					} else {
						// case for transform_apply where the #nnz ofr bow is unknown yet, since we have no build phase,
						// we have to compute the nnz now, we parallelize for now over the #bowEncoders and #rows
						// for the aggregation we parallelize just over the number of rows
						bowNnzRowOffsets = getNnzPerRowFromBOWEncoders(input, encm, k);
						// the last array contains the complete aggregation
						int static_offset = input.getNumColumns() - 1;
						// we just subtract -1 since we already subtracted -1 for every bow encoder except the first
						int[] aggOffsets = bowNnzRowOffsets.get(bowNnzRowOffsets.size() - 1);
						for (int i = 0; i < rptr.length-1; i++) {
							rptr[i+1] = rptr[i] + static_offset + aggOffsets[i];
						}
						nnzInt = rptr[rptr.length-1];
					}
				}
				SparseBlockCSR csrblock = new SparseBlockCSR(rptr, new int[nnzInt],  new double[nnzInt], nnzInt) ;
				output.setSparseBlock(csrblock);

			}
		}
		else {
			// Allocate dense block and set nnz to total #entries
			output.allocateDenseBlock(true, encm.hasWE);
			if(encm.hasWE){
				DenseBlockFP64DEDUP dedup = ((DenseBlockFP64DEDUP) output.getDenseBlock());
				dedup.setDistinct(encm.distinctWE);
				dedup.setEmbeddingSize(encm.sizeWE);
			}
			//output.setAllNonZeros();
		}

		if(DMLScript.STATISTICS) {
			LOG.debug("Elapsed time for allocation: "+ ((double) System.nanoTime() - t0) / 1000000 + " ms");
			TransformStatistics.incOutMatrixPreProcessingTime(System.nanoTime()-t0);
		}
		return bowNnzRowOffsets;
	}

	private static ArrayList<int[]> getNnzPerRowFromBOWEncoders(CacheBlock<?> input, EncoderMeta encm, int k) {
		ArrayList<int[]> bowNnzRowOffsets;
		int min_block_size = 1000;
		int num_blocks = input.getNumRows() / min_block_size;
		// 1 <= num_blks1 <= k / #enc
		int num_blks1= Math.min( (k + encm.numBOWEnc - 1)/ encm.numBOWEnc, Math.max(num_blocks, 1));
		int blk_len1 = (input.getNumRows() + num_blks1 - 1) / num_blks1;
		// 1 <= num_blks2 <= k
		int num_blks2= Math.min(k, Math.max(num_blocks, 1));
		int blk_len2 = (input.getNumRows() + num_blks2 - 1) / num_blks1;

		ExecutorService pool = CommonThreadPool.get(k);
		ArrayList<int[]> bowNnzRowOffsetsFinal = new ArrayList<>();
		try {
			encm.bowEncoders.forEach(e -> e._nnzPerRow = new int[input.getNumRows()]);
			ArrayList<Future<?>> list = new ArrayList<>();
			for (int i = 0; i < num_blks1; i++) {
				int start = i * blk_len1;
				int end = Math.min((i + 1) * blk_len1, input.getNumRows());
				list.add(pool.submit(() -> encm.bowEncoders.stream().parallel()
					.forEach(e -> e.computeNnzPerRow(input, start, end))));
			}
			for(Future<?> f : list)
				f.get();
			list.clear();
			int[] previous = null;
			for(ColumnEncoderComposite enc : encm.encs){
				if(enc.hasEncoder(ColumnEncoderBagOfWords.class)){
					previous = previous == null ? 
						enc.getEncoder(ColumnEncoderBagOfWords.class)._nnzPerRow :
						new int[input.getNumRows()];
				}
				bowNnzRowOffsetsFinal.add(previous);
			}
			for (int i = 0; i < num_blks2; i++) {
				int start = i * blk_len1;
				list.add(pool.submit(() -> aggregateNnzPerRow(start, blk_len2, 
					input.getNumRows(), encm.encs, bowNnzRowOffsetsFinal)));
			}
			for(Future<?> f : list)
				f.get();
			bowNnzRowOffsets = bowNnzRowOffsetsFinal;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
		return bowNnzRowOffsets;
	}

	private static void aggregateNnzPerRow(int start, int blk_len, int numRows, List<ColumnEncoderComposite> encs, ArrayList<int[]> bowNnzRowOffsets) {
		int end = Math.min(start + blk_len, numRows);
		int pos = 0;
		int[] aggRowOffsets = null;
		for(ColumnEncoderComposite enc : encs){
			int[] currentOffsets = bowNnzRowOffsets.get(pos);
			if (enc.hasEncoder(ColumnEncoderBagOfWords.class)) {
				ColumnEncoderBagOfWords bow = enc.getEncoder(ColumnEncoderBagOfWords.class);
				if(aggRowOffsets == null)
					aggRowOffsets = currentOffsets;
				else
					for (int i = start; i < end; i++)
						currentOffsets[i] = aggRowOffsets[i] + bow._nnzPerRow[i] - 1;
			}
			pos++;
		}
	}

	private void outputMatrixPostProcessing(MatrixBlock output, int k){
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(output.isInSparseFormat() && containsZeroOut()){
			if (k == 1) 
				outputMatrixPostProcessingSingleThread(output);
			else 
				outputMatrixPostProcessingParallel(output, k);	
		}
		output.recomputeNonZeros(k);
		
		if(DMLScript.STATISTICS)
			TransformStatistics.incOutMatrixPostProcessingTime(System.nanoTime()-t0);
	}

	private void outputMatrixPostProcessingSingleThread(MatrixBlock output){
		final SparseBlock sb = output.getSparseBlock();
		if(sb instanceof SparseBlockMCSR) {
			IntStream.range(0, output.getNumRows()).forEach(row -> {
				sb.compact(row);
			});
		}
		else {
			((SparseBlockCSR) sb).compact();
		}
	}

	private boolean containsZeroOut() {
		for(ColumnEncoder e : _columnEncoders)
			if(e.containsZeroOut())
				return true;
		return false;
	}

	private void outputMatrixPostProcessingParallel(MatrixBlock output, int k) {
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			final SparseBlock sb = output.getSparseBlock();
			if(sb instanceof SparseBlockMCSR) {
				pool.submit(() -> {
					IntStream.range(0, output.getNumRows()).parallel().forEach(row -> {
						sb.compact(row);
					});
				}).get();
			}
			else {
				((SparseBlockCSR) sb).compact();
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			pool.shutdown();
		}
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			columnEncoder.allocateMetaData(meta);
		}
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		return getMetaData(meta, 1);
	}

	public FrameBlock getMetaData(FrameBlock meta, int k) {
		long t0 = System.nanoTime();
		if(_meta != null)
			return _meta;
		if(meta == null)
			meta = new FrameBlock(_columnEncoders.size(), ValueType.STRING);
		this.allocateMetaData(meta);
		if (k > 1) {
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<ColumnMetaDataTask<? extends ColumnEncoder>> tasks = new ArrayList<>();
				for(ColumnEncoder columnEncoder : _columnEncoders)
					tasks.add(new ColumnMetaDataTask<>(columnEncoder, meta));
				List<Future<Object>> taskret = pool.invokeAll(tasks);
				for (Future<Object> task : taskret)
					task.get();
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}
		else {
			for(ColumnEncoder columnEncoder : _columnEncoders)
				columnEncoder.getMetaData(meta);
		}

		//_columnEncoders.stream().parallel().forEach(columnEncoder ->
		//		columnEncoder.getMetaData(meta));
		if(_legacyOmit != null)
			_legacyOmit.getMetaData(meta);
		if(_legacyMVImpute != null)
			_legacyMVImpute.getMetaData(meta);
		LOG.debug("Time spent getting metadata "+((double) System.nanoTime() - t0) / 1000000 + " ms");
		return meta;
	}

	@Override
	public void initMetaData(FrameBlock meta) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.initMetaData(meta);
		if(_legacyOmit != null)
			_legacyOmit.initMetaData(meta);
		if(_legacyMVImpute != null)
			_legacyMVImpute.initMetaData(meta);
	}

	//pass down init to composite encoders
	public void initEmbeddings(MatrixBlock embeddings) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.initEmbeddings(embeddings);
	}

	@Override
	public void prepareBuildPartial() {
		for(Encoder encoder : _columnEncoders)
			encoder.prepareBuildPartial();
	}

	@Override
	public void buildPartial(FrameBlock in) {
		for(Encoder encoder : _columnEncoders)
			encoder.buildPartial(in);
	}

	/**
	 * Obtain the column mapping of encoded frames based on the passed meta data frame.
	 *
	 * @param meta meta data frame block
	 * @return matrix with column mapping (one row per attribute)
	 */
	public MatrixBlock getColMapping(FrameBlock meta) {
		MatrixBlock out = new MatrixBlock(meta.getNumColumns(), 3, false);
		List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class);

		for(int i = 0, ni = 0; i < out.getNumRows(); i++) {
			final int colID = i + 1; // 1-based
			int nColID = ni + 1;
			List<ColumnEncoderDummycode> encoder = dc.stream().filter(e -> e.getColID() == colID)
				.collect(Collectors.toList());
			assert encoder.size() <= 1;
			if(encoder.size() == 1) {
				ni += meta.getColumnMetadata(i).getNumDistinct();
			}
			else {
				ni++;
			}
			out.set(i, 0, colID);
			out.set(i, 1, nColID);
			out.set(i, 2, ni);
		}
		return out;
	}

	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims, int offset) {
		_columnEncoders.forEach(encoder -> encoder.updateIndexRanges(beginDims, endDims, offset));
		if(_legacyOmit != null)
			_legacyOmit.updateIndexRanges(beginDims, endDims);
		if(_legacyMVImpute != null)
			_legacyMVImpute.updateIndexRanges(beginDims, endDims);
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeBoolean(_legacyMVImpute != null);
		if(_legacyMVImpute != null)
			_legacyMVImpute.writeExternal(out);
		out.writeBoolean(_legacyOmit != null);
		if(_legacyOmit != null)
			_legacyOmit.writeExternal(out);

		out.writeInt(_colOffset);
		out.writeInt(_columnEncoders.size());
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			out.writeInt(columnEncoder._colID);
			columnEncoder.writeExternal(out);
		}
		out.writeBoolean(_meta != null);
		if(_meta != null)
			_meta.write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		if(in.readBoolean()) {
			_legacyMVImpute = new EncoderMVImpute();
			_legacyMVImpute.readExternal(in);
		}
		if(in.readBoolean()) {
			_legacyOmit = new EncoderOmit();
			_legacyOmit.readExternal(in);
		}

		_colOffset = in.readInt();
		int encodersSize = in.readInt();
		_columnEncoders = new ArrayList<>();
		for(int i = 0; i < encodersSize; i++) {
			int colID = in.readInt();
			ColumnEncoderComposite columnEncoder = new ColumnEncoderComposite();
			columnEncoder.readExternal(in);
			columnEncoder.setColID(colID);
			_columnEncoders.add(columnEncoder);
		}
		if(in.readBoolean()) {
			FrameBlock meta = new FrameBlock();
			meta.readFields(in);
			_meta = meta;
		}
	}

	public <T extends ColumnEncoder> List<T> getColumnEncoders(Class<T> type) {
		// TODO cache results for faster access
		List<T> ret = new ArrayList<>();
		for(ColumnEncoder encoder : _columnEncoders) {
			if(encoder.getClass().equals(ColumnEncoderComposite.class) && type != ColumnEncoderComposite.class) {
				encoder = ((ColumnEncoderComposite) encoder).getEncoder(type);
			}
			if(encoder != null && encoder.getClass().equals(type)) {
				ret.add(type.cast(encoder));
			}
		}
		return ret;
	}

	public <T extends ColumnEncoder> T getColumnEncoder(int colID, Class<T> type) {
		for(T encoder : getColumnEncoders(type)) {
			if(encoder._colID == colID) {
				return encoder;
			}
		}
		return null;
	}

	public <T extends ColumnEncoder, E> List<E> getFromAll(Class<T> type, Function<? super T, ? extends E> mapper) {
		return getColumnEncoders(type).stream().map(mapper).collect(Collectors.toList());
	}

	public <T extends ColumnEncoder> int[] getFromAllIntArray(Class<T> type,
		Function<? super T, ? extends Integer> mapper) {
		return getFromAll(type, mapper).stream().mapToInt(i -> i).toArray();
	}

	public <T extends ColumnEncoder> double[] getFromAllDoubleArray(Class<T> type,
		Function<? super T, ? extends Double> mapper) {
		return getFromAll(type, mapper).stream().mapToDouble(i -> i).toArray();
	}

	public List<ColumnEncoderComposite> getColumnEncoders() {
		return _columnEncoders;
	}

	public List<ColumnEncoderComposite> getCompositeEncodersForID(int colID) {
		return _columnEncoders.stream().filter(encoder -> encoder._colID == colID).collect(Collectors.toList());
	}

	public List<Class<? extends ColumnEncoder>> getEncoderTypes(int colID) {
		HashSet<Class<? extends ColumnEncoder>> set = new HashSet<>();
		for(ColumnEncoderComposite encoderComp : _columnEncoders) {
			if(encoderComp._colID != colID && colID != -1)
				continue;
			for(ColumnEncoder encoder : encoderComp.getEncoders()) {
				set.add(encoder.getClass());
			}
		}
		return new ArrayList<>(set);
	}

	public List<Class<? extends ColumnEncoder>> getEncoderTypes() {
		return getEncoderTypes(-1);
	}

	public int getNumOutCols() {
		int sum = 0;
		for(int i = 0; i < _columnEncoders.size(); i++)
			sum += _columnEncoders.get(i).getDomainSize();
		return sum;
	}

	public int getNumExtraCols(IndexRange ixRange) {
		List<ColumnEncoderDummycode> dc = getColumnEncoders(ColumnEncoderDummycode.class).stream()
			.filter(dce -> ixRange.inColRange(dce._colID)).collect(Collectors.toList());
		if(dc.isEmpty()) {
			return 0;
		}
		return dc.stream().map(ColumnEncoderDummycode::getDomainSize).mapToInt(i -> i).sum() - dc.size();
	}

	public <T extends ColumnEncoder> boolean containsEncoderForID(int colID, Class<T> type) {
		return getColumnEncoders(type).stream().anyMatch(encoder -> encoder.getColID() == colID);
	}

	public <T extends ColumnEncoder, E> void applyToAll(Class<T> type, Consumer<? super T> function) {
		getColumnEncoders(type).forEach(function);
	}

	public <T extends ColumnEncoder, E> void applyToAll(Consumer<? super ColumnEncoderComposite> function) {
		getColumnEncoders().forEach(function);
	}

	public MultiColumnEncoder subRangeEncoder(IndexRange ixRange) {
		List<ColumnEncoderComposite> encoders = new ArrayList<>();
		for(long i = ixRange.colStart; i < ixRange.colEnd; i++) {
			encoders.addAll(getCompositeEncodersForID((int) i));
		}
		MultiColumnEncoder subRangeEncoder = new MultiColumnEncoder(encoders);
		subRangeEncoder._colOffset = (int) -ixRange.colStart + 1;
		if(_legacyOmit != null)
			subRangeEncoder.addReplaceLegacyEncoder(_legacyOmit.subRangeEncoder(ixRange));
		if(_legacyMVImpute != null)
			subRangeEncoder.addReplaceLegacyEncoder(_legacyMVImpute.subRangeEncoder(ixRange));
		return subRangeEncoder;
	}

	public <T extends ColumnEncoder> MultiColumnEncoder subRangeEncoder(IndexRange ixRange, Class<T> type) {
		List<T> encoders = new ArrayList<>();
		for(long i = ixRange.colStart; i < ixRange.colEnd; i++) {
			encoders.add(getColumnEncoder((int) i, type));
		}
		if(type.equals(ColumnEncoderComposite.class))
			return new MultiColumnEncoder(
				encoders.stream().map(e -> ((ColumnEncoderComposite) e)).collect(Collectors.toList()));
		else
			return new MultiColumnEncoder(
				encoders.stream().map(ColumnEncoderComposite::new).collect(Collectors.toList()));
	}

	public void mergeReplace(MultiColumnEncoder multiEncoder) {
		for(ColumnEncoderComposite otherEncoder : multiEncoder._columnEncoders) {
			ColumnEncoderComposite encoder = getColumnEncoder(otherEncoder._colID, otherEncoder.getClass());
			if(encoder != null) {
				_columnEncoders.remove(encoder);
			}
			_columnEncoders.add(otherEncoder);
		}
	}

	public void mergeAt(Encoder other, int columnOffset, int row) {
		if(other instanceof MultiColumnEncoder) {
			for(ColumnEncoder encoder : ((MultiColumnEncoder) other)._columnEncoders) {
				addEncoder(encoder, columnOffset);
			}
			// +1 since legacy function uses 1-based
			legacyMergeAt((MultiColumnEncoder) other, row, columnOffset + 1);
		}
		else {
			addEncoder((ColumnEncoder) other, columnOffset);
		}
	}

	private void legacyMergeAt(MultiColumnEncoder other, int row, int col) {
		if(other._legacyOmit != null)
			other._legacyOmit.shiftCols(col - 1);
		if(other._legacyOmit != null) {
			if(_legacyOmit == null)
				_legacyOmit = new EncoderOmit();
			_legacyOmit.mergeAt(other._legacyOmit, row, col);
		}

		if(other._legacyMVImpute != null)
			other._legacyMVImpute.shiftCols(col - 1);
		if(_legacyMVImpute != null && other._legacyMVImpute != null)
			_legacyMVImpute.mergeAt(other._legacyMVImpute, row, col);
		else if(_legacyMVImpute == null)
			_legacyMVImpute = other._legacyMVImpute;

	}

	private void addEncoder(ColumnEncoder encoder, int columnOffset) {
		// Check if same encoder exists
		int colId = encoder._colID + columnOffset;
		ColumnEncoder presentEncoder = getColumnEncoder(colId, encoder.getClass());
		if(presentEncoder != null) {
			encoder.shiftCol(columnOffset);
			presentEncoder.mergeAt(encoder);
		}
		else {
			// Check if CompositeEncoder for this colID exists
			ColumnEncoderComposite presentComposite = getColumnEncoder(colId, ColumnEncoderComposite.class);
			if(presentComposite != null) {
				// if here encoder can never be a CompositeEncoder
				encoder.shiftCol(columnOffset);
				presentComposite.mergeAt(encoder);
			}
			else {
				encoder.shiftCol(columnOffset);
				if(encoder instanceof ColumnEncoderComposite) {
					_columnEncoders.add((ColumnEncoderComposite) encoder);
				}
				else {
					_columnEncoders.add(new ColumnEncoderComposite(encoder));
				}
			}
		}
	}

	public <T extends LegacyEncoder> void addReplaceLegacyEncoder(T encoder) {
		if(encoder.getClass() == EncoderMVImpute.class) {
			_legacyMVImpute = (EncoderMVImpute) encoder;
		}
		else if(encoder.getClass().equals(EncoderOmit.class)) {
			_legacyOmit = (EncoderOmit) encoder;
		}
		else {
			throw new DMLRuntimeException("Tried to add non legacy Encoder");
		}
	}

	public <T extends LegacyEncoder> boolean hasLegacyEncoder() {
		return hasLegacyEncoder(EncoderMVImpute.class) || hasLegacyEncoder(EncoderOmit.class);
	}

	public boolean isCompressedTransformEncode(CacheBlock<?> in, boolean enabled){
		return (enabled || ConfigurationManager.getDMLConfig().getBooleanValue(DMLConfig.COMPRESSED_TRANSFORMENCODE)) &&
			in instanceof FrameBlock && _colOffset == 0;
	}

	public <T extends LegacyEncoder> boolean hasLegacyEncoder(Class<T> type) {
		if(type.equals(EncoderMVImpute.class))
			return _legacyMVImpute != null;
		if(type.equals(EncoderOmit.class))
			return _legacyOmit != null;
		assert false;
		return false;
	}

	public <T extends LegacyEncoder> T getLegacyEncoder(Class<T> type) {
		if(type.equals(EncoderMVImpute.class))
			return type.cast(_legacyMVImpute);
		if(type.equals(EncoderOmit.class))
			return type.cast(_legacyOmit);
		assert false;
		return null;
	}

	/*
	 * This function applies the _columOffset to all encoders. Used in federated env.
	 */
	public void applyColumnOffset() {
		applyToAll(e -> e.shiftCol(_colOffset));
		if(_legacyOmit != null)
			_legacyOmit.shiftCols(_colOffset);
		if(_legacyMVImpute != null)
			_legacyMVImpute.shiftCols(_colOffset);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\nIs Legacy: ");
		sb.append(_legacyMVImpute);
		sb.append("\nEncoders:\n");

		for(int i = 0; i < _columnEncoders.size(); i++) {
			sb.append(_columnEncoders.get(i));
			sb.append("\n");
		}

		return sb.toString();
	}

	private static class EncoderMeta {
		// contains information about the encoders and their relevant data characteristics
		public final boolean hasUDF;
		public final boolean hasDC;
		public final boolean hasWE;
		public final int distinctWE;
		public final int sizeWE;
		public final long nnzBOW;
		public final int numBOWEnc;
		public final int[] nnzPerRowBOW;
		public final ArrayList<ColumnEncoderBagOfWords> bowEncoders;
		public final List<ColumnEncoderComposite> encs;

		public EncoderMeta(boolean hasUDF, boolean hasDC, boolean hasWE, int distinctWE, int sizeWE, long nnzBOW,
						   int numBOWEncoder, int[] nnzPerRowBOW, ArrayList<ColumnEncoderBagOfWords> bows,
						   List<ColumnEncoderComposite> encoders) {
			this.hasUDF = hasUDF;
			this.hasDC = hasDC;
			this.hasWE = hasWE;
			this.distinctWE = distinctWE;
			this.sizeWE = sizeWE;
			this.nnzBOW = nnzBOW;
			this.numBOWEnc = numBOWEncoder;
			this.nnzPerRowBOW = nnzPerRowBOW;
			this.bowEncoders = bows;
			this.encs = encoders;
		}
	}

	private static EncoderMeta getEncMeta(List<ColumnEncoderComposite> encoders, boolean noBuild, int k, CacheBlock<?> in) {
		boolean hasUDF = false, hasDC = false, hasWE = false;
		int distinctWE = 0;
		int sizeWE = 0;
		long nnzBOW = 0;
		int numBOWEncoder = 0;
		int[] nnzPerRowBOW = null;
		ArrayList<ColumnEncoderBagOfWords> bows = new ArrayList<>();
		for (ColumnEncoderComposite enc : encoders){
			if(enc.hasEncoder(ColumnEncoderUDF.class))
				hasUDF = true;
			else if (enc.hasEncoder(ColumnEncoderDummycode.class))
				hasDC = true;
			else if(enc.hasEncoder(ColumnEncoderBagOfWords.class)){
				ColumnEncoderBagOfWords bowEnc = enc.getEncoder(ColumnEncoderBagOfWords.class);
				numBOWEncoder++;
				nnzBOW += bowEnc._nnz;
				if(noBuild){
					// estimate nnz by sampling
					bows.add(bowEnc);
				} else if(nnzPerRowBOW != null)
					for (int i = 0; i < bowEnc._nnzPerRow.length; i++) {
						nnzPerRowBOW[i] += bowEnc._nnzPerRow[i];
					}
				else {
					nnzPerRowBOW = bowEnc._nnzPerRow.clone();
				}
			}
			else if(enc.hasEncoder(ColumnEncoderWordEmbedding.class)){
				hasWE = true;
				distinctWE = enc.getEncoder(ColumnEncoderWordEmbedding.class).getNrDistinctEmbeddings();
				sizeWE = enc.getDomainSize();
			}
		}
		if(!bows.isEmpty()){
			int[] sampleInds = getSampleIndices(in, in.getNumRows() > 1000 ? (int) (0.1 * in.getNumRows()) : in.getNumRows(), (int) System.nanoTime(), 1);
			// Concurrent (column-wise) bag of words nnz estimation per row, we estimate the number of nnz because the
			// exact number is only needed for sparse outputs not for dense, if sparse, we recount the nnz for all rows later
			// Note: the sampling might be problematic since we used for the sparsity estimation -> which impacts performance
			// if we go for the non-ideal output format
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				Double result = pool.submit(() -> bows.stream().parallel()
							.mapToDouble(e -> e.computeNnzEstimate(in, sampleInds))
							.sum()).get();
				nnzBOW = (long) Math.ceil(result);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}
        return new EncoderMeta(hasUDF, hasDC, hasWE, distinctWE, sizeWE, nnzBOW, numBOWEncoder, nnzPerRowBOW, bows, encoders);
	}

	/*
	 * Currently, not in use will be integrated in the future
	 */
	@SuppressWarnings("unused")
	private static class MultiColumnLegacyBuildTask implements Callable<Object> {

		private final MultiColumnEncoder _encoder;
		private final FrameBlock _input;

		protected MultiColumnLegacyBuildTask(MultiColumnEncoder encoder, FrameBlock input) {
			_encoder = encoder;
			_input = input;
		}

		@Override
		public Void call() throws Exception {
			_encoder.legacyBuild(_input);
			return null;
		}
	}

	@SuppressWarnings("unused")
	private static class MultiColumnLegacyMVImputeMetaPrepareTask implements Callable<Object> {

		private final MultiColumnEncoder _encoder;
		private final FrameBlock _input;

		protected MultiColumnLegacyMVImputeMetaPrepareTask(MultiColumnEncoder encoder, FrameBlock input) {
			_encoder = encoder;
			_input = input;
		}

		@Override
		public Void call() throws Exception {
			_encoder._meta = _encoder.getMetaData(new FrameBlock(_input.getNumColumns(), Types.ValueType.STRING));
			_encoder.initMetaData(_encoder._meta);
			return null;
		}
	}

	private static class InitOutputMatrixTask implements Callable<Object> {
		private final MultiColumnEncoder _encoder;
		private final CacheBlock<?> _input;
		private final MatrixBlock _output;

		private InitOutputMatrixTask(MultiColumnEncoder encoder, CacheBlock<?> input, MatrixBlock output) {
			_encoder = encoder;
			_input = input;
			_output = output;
		}

		@Override
		public Object call() {
			EncoderMeta encm = getEncMeta(_encoder.getEncoders(), false, -1, _input);
			int numCols = _encoder.getNumOutCols();
			long estNNz = (long) _input.getNumRows() * (encm.hasUDF ? numCols : _input.getNumColumns() - encm.numBOWEnc) + encm.nnzBOW;
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(_input.getNumRows(), numCols, estNNz) && !encm.hasUDF;
			_output.reset(_input.getNumRows(), numCols, sparse, estNNz);
			outputMatrixPreProcessing(_output, _input, encm, estNNz, 1);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName();
		}

	}

	private static class ApplyTasksWrapperTask extends DependencyWrapperTask<Object> {
		private final ColumnEncoder _encoder;
		private final MatrixBlock _out;
		private final CacheBlock<?> _in;
		/** Offset because of dummmy coding such that the column id is correct. */
		private int _offset = -1;
		private int[] _sparseRowPointerOffsets = null;

		private ApplyTasksWrapperTask(ColumnEncoder encoder, CacheBlock<?> in, 
				MatrixBlock out, DependencyThreadPool pool) {
			super(pool);
			_encoder = encoder;
			_out = out;
			_in = in;
		}

		@Override
		public List<DependencyTask<?>> getWrappedTasks() {
			return _encoder.getApplyTasks(_in, _out, _encoder._colID - 1 + _offset, _sparseRowPointerOffsets);
		}

		@Override
		public Object call() throws Exception {
			// Is called only when building of encoder is done, Output Matrix is allocated
			// and _outputCol has been updated!
			if(_offset == -1)
				throw new DMLRuntimeException(
					"OutputCol for apply task wrapper has not been updated!, Most likely some concurrency issues\n " + this);
			return super.call();
		}

		public void setOffset(int offset) {
			_offset = offset;
		}

		public void setSparseRowPointerOffsets(int[] offsets) {
			_sparseRowPointerOffsets = offsets;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}
	}

	/*
	 * Task responsible for updating the output column of the apply tasks after the building of the DC recoders. So the
	 * offsets in the output are correct.
	 */
	private static class UpdateOutputColTask implements Callable<Object> {
		private final MultiColumnEncoder _encoder;
		private final List<DependencyTask<?>> _applyTasksWrappers;

		private UpdateOutputColTask(MultiColumnEncoder encoder, List<DependencyTask<?>> applyTasksWrappers) {
			_encoder = encoder;
			_applyTasksWrappers = applyTasksWrappers;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName();
		}

		@Override
		public Object call() throws Exception {
			// updates the outputCol offset and sets the nnz offsets, which are created by bow encoders, in each encoder
			int currentCol = -1;
			int currentOffset = 0;
			int[] sparseRowPointerOffsets = null;
			for(DependencyTask<?> dtask : _applyTasksWrappers) {
				((ApplyTasksWrapperTask) dtask).setOffset(currentOffset);
				if(sparseRowPointerOffsets != null)
					((ApplyTasksWrapperTask) dtask).setSparseRowPointerOffsets(sparseRowPointerOffsets);
				int nonOffsetCol = ((ApplyTasksWrapperTask) dtask)._encoder._colID - 1;
				if(nonOffsetCol > currentCol) {
					currentCol = nonOffsetCol;
					ColumnEncoderComposite enc = _encoder._columnEncoders.get(nonOffsetCol);
					if(enc.hasEncoder(ColumnEncoderDummycode.class))
						currentOffset += enc.getEncoder(ColumnEncoderDummycode.class)._domainSize - 1;
					else if (enc.hasEncoder(ColumnEncoderBagOfWords.class)) {
						ColumnEncoderBagOfWords bow = enc.getEncoder(ColumnEncoderBagOfWords.class);
						currentOffset += bow.getDomainSize() - 1;
						if(sparseRowPointerOffsets == null)
							sparseRowPointerOffsets = bow._nnzPerRow;
						else{
							sparseRowPointerOffsets = sparseRowPointerOffsets.clone();
							// TODO: experiment if it makes sense to parallize here (for frames with many rows)
							for (int r = 0; r < sparseRowPointerOffsets.length; r++) {
								sparseRowPointerOffsets[r] += bow._nnzPerRow[r] - 1;
							}
						}
					}
				}
			}
			return null;
		}
	}

	private static class AllocMetaTask implements Callable<Object> {
		private final MultiColumnEncoder _encoder;
		private final FrameBlock _meta;
		
		private AllocMetaTask (MultiColumnEncoder encoder, FrameBlock meta) {
			_encoder = encoder;
			_meta = meta;
		}

		@Override
		public Object call() throws Exception {
			_encoder.allocateMetaData(_meta);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName();
		}
	}
	
	private static class ColumnMetaDataTask<T extends ColumnEncoder> implements Callable<Object> {
		private final T _colEncoder;
		private final FrameBlock _out;

		protected ColumnMetaDataTask(T encoder, FrameBlock out) {
			_colEncoder = encoder;
			_out = out;
		}

		@Override
		public Object call() throws Exception {
			_colEncoder.getMetaData(_out);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _colEncoder._colID + ">";
		}
	}
}
