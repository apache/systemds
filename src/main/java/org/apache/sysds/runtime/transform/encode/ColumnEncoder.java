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

import static org.apache.sysds.runtime.transform.encode.EncoderFactory.getEncoderType;
import static org.apache.sysds.runtime.util.UtilFunctions.getBlockSizes;
import static org.apache.sysds.runtime.util.UtilFunctions.getEndIndex;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;
import org.apache.sysds.utils.stats.TransformStatistics;

/**
 * Base class for all transform encoders providing both a row and block interface for decoding frames to matrices.
 *
 */
public abstract class ColumnEncoder implements Encoder, Comparable<ColumnEncoder> {
	protected static final Log LOG = LogFactory.getLog(ColumnEncoder.class.getName());
	public static int APPLY_ROW_BLOCKS_PER_COLUMN = -1;
	public static int BUILD_ROW_BLOCKS_PER_COLUMN = -1;
	private static final long serialVersionUID = 2299156350718979064L;
	protected int _colID;
	protected ArrayList<Integer> _sparseRowsWZeros = null;
	protected int[] sparseRowPointerOffset = null;	// offsets created by bag of words encoders (multiple nnz)
	// protected ArrayList<Integer> _sparseRowsWZeros = null;

	protected boolean containsZeroOut = false;
	protected long _estMetaSize = 0;
	protected int _estNumDistincts = 0;
	protected int _nBuildPartitions = 0;
	protected int _nApplyPartitions = 0;
	protected long _avgEntrySize = 0;

	//Override in ColumnEncoderWordEmbedding
	public void initEmbeddings(MatrixBlock embeddings){
		return;
	}

	protected enum TransformType{
		BIN, RECODE, DUMMYCODE, FEATURE_HASH, PASS_THROUGH, UDF, WORD_EMBEDDING, BAG_OF_WORDS, N_A
	}

	protected ColumnEncoder(int colID) {
		_colID = colID;
	}

	/**
	 * Apply Functions are only used in Single Threaded or Multi-Threaded Dense context.
	 * That's why there is no regard for MT sparse!
	 *
	 * @param in Input Block
	 * @param out Output Matrix
	 * @param outputCol The output column for the given column
	 * @return same as out
	 *
	 */
	public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol){
		return apply(in, out, outputCol, 0, -1);
	}

	public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		if(out.isInSparseFormat())
			applySparse(in, out, outputCol, rowStart, blk);
		else
			applyDense(in, out, outputCol, rowStart, blk);

		if (DMLScript.STATISTICS){
			long t = System.nanoTime()-t0;
			switch (this.getTransformType()){
				case RECODE:
					TransformStatistics.incRecodeApplyTime(t);
					break;
				case BIN:
					TransformStatistics.incBinningApplyTime(t);
					break;
				case DUMMYCODE:
					TransformStatistics.incDummyCodeApplyTime(t);
					break;
				case WORD_EMBEDDING:
					TransformStatistics.incWordEmbeddingApplyTime(t);
					break;
				case BAG_OF_WORDS:
					TransformStatistics.incBagOfWordsApplyTime(t);
					break;
				case FEATURE_HASH:
					TransformStatistics.incFeatureHashingApplyTime(t);
					break;
				case PASS_THROUGH:
					TransformStatistics.incPassThroughApplyTime(t);
					break;
				default:
					break;
			}
		}
		return out;
	}

	protected abstract double getCode(CacheBlock<?>in, int row);

	/**
	 * Get the coded values for a given range from start to end.
	 * 
	 * @param in       The CacheBlock to extract the values from
	 * @param startInd The start Index
	 * @param rowEnd   The end index
	 * @param tmp 	   double tmp array to put the result into if valid.
	 * @return The encoded double values.
	 */
	protected abstract double[] getCodeCol(CacheBlock<?> in, int startInd, int rowEnd, double[] tmp);

	protected void applySparse(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
		boolean mcsr = out.getSparseBlock() instanceof SparseBlockMCSR;
		int index = _colID - 1;
		// Apply loop tiling to exploit CPU caches
		int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
		double[] codes = getCodeCol(in, rowStart, rowEnd, null);
		int B = 32; //tile size
		for(int i = rowStart; i < rowEnd; i+=B) {
			int lim = Math.min(i+B, rowEnd);
			for (int ii=i; ii<lim; ii++) {
				int indexWithOffset = sparseRowPointerOffset != null ? sparseRowPointerOffset[ii] - 1 + index : index;
				if (mcsr) {
					SparseRowVector row = (SparseRowVector) out.getSparseBlock().get(ii);
					row.values()[index] = codes[ii-rowStart];
					row.indexes()[index] = outputCol;
				}
				else { //csr
					// Manually fill the column-indexes and values array
					SparseBlockCSR csrblock = (SparseBlockCSR)out.getSparseBlock();
					int rptr[] = csrblock.rowPointers();
					csrblock.indexes()[rptr[ii]+indexWithOffset] = outputCol;
					csrblock.values()[rptr[ii]+indexWithOffset] = codes[ii-rowStart];
				}
			}
		}
	}

	protected void applyDense(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk){
		// Apply loop tiling to exploit CPU caches
		final int rowEnd = getEndIndex(in.getNumRows(), rowStart, blk);
		final int smallTile = 64;
		final double[] tmp = new double[smallTile]; 
		final DenseBlock outB = out.getDenseBlock();
		if( outB.isContiguous(rowStart, blk)) 
			for(int i = rowStart; i < rowEnd; i += smallTile) 
				applyDenseTileContiguous(in, outB, outputCol, i, Math.min(i + smallTile, rowEnd),tmp);	
		else 
			for(int i = rowStart; i < rowEnd; i += smallTile) 
				applyDenseTileGeneric(in, outB, outputCol, i, Math.min(i + smallTile, rowEnd),tmp);
		
	}

	private void applyDenseTileContiguous(CacheBlock<?> in, DenseBlock out, int outputCol, int s, int e, double[] tmp) {
		// these are codes for this block offset by rowStart
		final double[] codes = getCodeCol(in, s, e, tmp);
		final double[] vals = out.values(s);
		int off = out.pos(s) + outputCol;
		final int nCol = out.getDim(1);
		for(int i = 0; i < e - s; i++, off += nCol){
			vals[off] = codes[i];
		}
	}

	private void applyDenseTileGeneric(CacheBlock<?> in, DenseBlock out, int outputCol, int s, int e, double[] tmp) {
		// these are codes for this block offset by rowStart
		final double[] codes = getCodeCol(in, s, e, tmp);
		for(int i = s; i < e; i++) // rows
			out.set(i, outputCol, codes[i - s]);
	}

	protected abstract TransformType getTransformType();
	/**
	 * Indicates if this encoder is applicable, i.e, if there is a column to encode.
	 *
	 * @return true if a colID is set
	 */
	public boolean isApplicable() {
		return _colID != -1;
	}

	/**
	 * Indicates if this encoder is applicable for the given column ID, i.e., if it is subject to this transformation.
	 *
	 * @param colID column ID
	 * @return true if encoder is applicable for given column
	 */
	public boolean isApplicable(int colID) {
		return colID == _colID;
	}

	/**
	 * Allocates internal data structures for partial build.
	 */
	public void prepareBuildPartial() {
		// do nothing
	}

	public int getDomainSize(){
		return 1;
	}


	/**
	 * Partial build of internal data structures (e.g., in distributed spark operations).
	 *
	 * @param in input frame block
	 */
	public void buildPartial(FrameBlock in) {
		// do nothing
	}

	public void build(CacheBlock<?> in, double[] equiHeightMaxs) {
		// do nothing
	}

	public void build(CacheBlock<?> in, Map<Integer, double[]> equiHeightMaxs) {
		// do nothing
	}


	/**
	 * Merges another encoder, of a compatible type, in after a certain position. Resizes as necessary.
	 * <code>ColumnEncoders</code> are compatible with themselves and <code>EncoderComposite</code> is compatible with
	 * every other <code>ColumnEncoders</code>. <code>MultiColumnEncoders</code> are compatible with every encoder
	 *
	 * @param other the encoder that should be merged in
	 */
	public void mergeAt(ColumnEncoder other) {
		throw new DMLRuntimeException(
			this.getClass().getSimpleName() + " does not support merging with " + other.getClass().getSimpleName());
	}

	/**
	 * Update index-ranges to after encoding. Note that only Dummycoding changes the ranges.
	 *
	 * @param beginDims begin dimensions of range
	 * @param endDims   end dimensions of range
	 */
	public void updateIndexRanges(long[] beginDims, long[] endDims, int colOffset) {
		// do nothing - default
	}

	/**
	 * Obtain the column mapping of encoded frames based on the passed meta data frame.
	 *
	 * @param meta meta data frame block
	 * @return matrix with column mapping (one row per attribute)
	 */
	public MatrixBlock getColMapping(FrameBlock meta) {
		// default: do nothing
		return null;
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd serialization.
	 *
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput os) throws IOException {
		os.writeInt(_colID);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd deserialization.
	 *
	 * @param in object input
	 * @throws IOException if IOException occur
	 */
	@Override
	public void readExternal(ObjectInput in) throws IOException {
		_colID = in.readInt();
	}

	public int getColID() {
		return _colID;
	}

	public void setColID(int colID) {
		_colID = colID;
	}

	public void shiftCol(int columnOffset) {
		_colID += columnOffset;
	}

	public void setEstMetaSize(long estSize) {
		_estMetaSize = estSize;
	}

	public long getEstMetaSize() {
		return _estMetaSize;
	}

	public void setEstNumDistincts(int numDistincts) {
		_estNumDistincts = numDistincts;
	}

	public int getEstNumDistincts() {
		return _estNumDistincts;
	}

	public void computeMapSizeEstimate(CacheBlock<?> in, int[] sampleIndices) {
		throw new DMLRuntimeException(this + " does not need map size estimation");
	}


	@Override
	public int compareTo(ColumnEncoder o) {
		return Integer.compare(getEncoderType(this), getEncoderType(o));
	}

	/*
	 * Returns a Dependency Task List such that if executed the encoder is built. Last Task in the list shall only
	 * complete if all previous tasks are done. This is so that we can use the last task as a dependency for the whole
	 * build, reducing unnecessary dependencies.
	 */
	public List<DependencyTask<?>> getBuildTasks(CacheBlock<?> in) {
		List<Callable<Object>> tasks = new ArrayList<>();
		List<List<? extends Callable<?>>> dep = null;
		int nRows = in.getNumRows();
		int[] blockSizes = getBlockSizes(nRows, _nBuildPartitions);
		if(blockSizes.length == 1) {
			tasks.add(getBuildTask(in));
		}
		else {
			if(this instanceof ColumnEncoderBagOfWords)
				((ColumnEncoderBagOfWords) this).initNnzPartials(in.getNumRows(), blockSizes.length);
			HashMap<Integer, Object> ret = new HashMap<>();
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++)
				tasks.add(getPartialBuildTask(in, startRow, blockSizes[i], ret, i));
			tasks.add(getPartialMergeBuildTask(ret));
			dep = new ArrayList<>(Collections.nCopies(tasks.size() - 1, null));
			dep.add(tasks.subList(0, tasks.size() - 1));
		}
		return DependencyThreadPool.createDependencyTasks(tasks, dep);
	}

	public Callable<Object> getBuildTask(CacheBlock<?> in) {
		throw new DMLRuntimeException("Trying to get the Build task of an Encoder which does not require building");
	}

	public Callable<Object> getPartialBuildTask(CacheBlock<?> in, int startRow, 
			int blockSize, HashMap<Integer, Object> ret, int p) {
		throw new DMLRuntimeException(
			"Trying to get the PartialBuild task of an Encoder which does not support  partial building");
	}

	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		throw new DMLRuntimeException(
			"Trying to get the BuildMergeTask task of an Encoder which does not support partial building");
	}


	public List<DependencyTask<?>> getApplyTasks(CacheBlock<?> in, MatrixBlock out, int outputCol, int[] sparseRowPointerOffsets) {
		List<Callable<Object>> tasks = new ArrayList<>();
		List<List<? extends Callable<?>>> dep = null;
		//for now single threaded apply for bag of words
		int[] blockSizes = getBlockSizes(in.getNumRows(), _nApplyPartitions);
		this.sparseRowPointerOffset = out.isInSparseFormat() ? sparseRowPointerOffsets : null;
		for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
			if(out.isInSparseFormat())
				tasks.add(getSparseTask(in, out, outputCol, startRow, blockSizes[i]));
			else
				tasks.add(getDenseTask(in, out, outputCol, startRow, blockSizes[i]));
		}
		if(tasks.size() > 1) {
			dep = new ArrayList<>(Collections.nCopies(tasks.size(), null));
			tasks.add(() -> null);  // Empty task as barrier
			dep.add(tasks.subList(0, tasks.size()-1));
		}

		return DependencyThreadPool.createDependencyTasks(tasks, dep);
	}

	protected ColumnApplyTask<? extends ColumnEncoder>
			getSparseTask(CacheBlock<?> in, MatrixBlock out, int outputCol, int startRow, int blk){
		return new ColumnApplyTask<>(this, in, out, outputCol, startRow, blk);
	}

	protected ColumnApplyTask<? extends ColumnEncoder>
			getDenseTask(CacheBlock<?> in, MatrixBlock out, int outputCol, int startRow, int blk){
		return new ColumnApplyTask<>(this, in, out, outputCol, startRow, blk);
	}

	protected void addSparseRowsWZeros(List<Integer> sparseRowsWZeros){
		synchronized (this){
			if(_sparseRowsWZeros == null)
				_sparseRowsWZeros = new ArrayList<>();
			_sparseRowsWZeros.addAll(sparseRowsWZeros);
		}
	
	}

	protected boolean containsZeroOut(){
		return containsZeroOut;
	}

	protected void setBuildRowBlocksPerColumn(int nPart) {
		_nBuildPartitions = nPart;
	}

	protected void setApplyRowBlocksPerColumn(int nPart) {
		_nApplyPartitions = nPart;
	}

	public enum EncoderType {
		Recode, FeatureHash, PassThrough, Bin, Dummycode, Omit, MVImpute, Composite, WordEmbedding, BagOfWords, UDF
	}

	/*
	 * This is the base Task for each column apply. If no custom "getApplyTasks" is implemented in an Encoder this task
	 * will be used.
	 */
	protected static class ColumnApplyTask<T extends ColumnEncoder> implements Callable<Object> {

		protected final T _encoder;
		protected final CacheBlock<?> _input;
		protected final MatrixBlock _out;
		protected final int _outputCol;
		protected final int _startRow;
		protected final int _blk;

		protected ColumnApplyTask(T encoder, CacheBlock<?> input, MatrixBlock out, int outputCol){
			this(encoder, input, out, outputCol, 0, -1);
		}

		protected ColumnApplyTask(T encoder, CacheBlock<?> input, MatrixBlock out, int outputCol, int startRow, int blk) {
			_encoder = encoder;
			_input = input;
			_out = out;
			_outputCol = outputCol;
			_startRow = startRow;
			_blk = blk;
		}

		@Override
		public Object call() throws Exception {
			assert _outputCol >= 0;
			_encoder.apply(_input, _out, _outputCol, _startRow, _blk);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<Encoder: " + _encoder.getClass().getSimpleName() + "; ColId: "
				+ _encoder._colID + ">";
		}

	}
}
