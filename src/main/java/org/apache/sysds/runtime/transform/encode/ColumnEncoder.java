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

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

/**
 * Base class for all transform encoders providing both a row and block interface for decoding frames to matrices.
 *
 */
public abstract class ColumnEncoder implements Externalizable, Encoder, Comparable<ColumnEncoder> {
	protected static final Log LOG = LogFactory.getLog(ColumnEncoder.class.getName());
	protected static final int APPLY_ROW_BLOCKS_PER_COLUMN = 1;
	public static int BUILD_ROW_BLOCKS_PER_COLUMN = 1;
	private static final long serialVersionUID = 2299156350718979064L;
	protected int _colID;

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
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol){
		return apply(in, out, outputCol, 0, -1);
	}

	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol){
		return apply(in, out, outputCol, 0, -1);
	}

	public abstract MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk);

	public abstract MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk);

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

	/**
	 * Partial build of internal data structures (e.g., in distributed spark operations).
	 *
	 * @param in input frame block
	 */
	public void buildPartial(FrameBlock in) {
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

	@Override
	public int compareTo(ColumnEncoder o) {
		return Integer.compare(getEncoderType(this), getEncoderType(o));
	}

	/*
	 * Returns a Dependency Task List such that if executed the encoder is built. Last Task in the list shall only
	 * complete if all previous tasks are done. This is so that we can use the last task as a dependency for the whole
	 * build, reducing unnecessary dependencies.
	 */
	public List<DependencyTask<?>> getBuildTasks(FrameBlock in) {
		List<Callable<Object>> tasks = new ArrayList<>();
		List<List<? extends Callable<?>>> dep = null;
		int nRows = in.getNumRows();
		int[] blockSizes = getBlockSizes(nRows, getNumBuildRowPartitions());
		if(blockSizes.length == 1) {
			tasks.add(getBuildTask(in));
		}
		else {
			HashMap<Integer, Object> ret = new HashMap<>();
			for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++)
				tasks.add(getPartialBuildTask(in, startRow, blockSizes[i], ret));
			tasks.add(getPartialMergeBuildTask(ret));
			dep = new ArrayList<>(Collections.nCopies(tasks.size() - 1, null));
			dep.add(tasks.subList(0, tasks.size() - 1));
		}
		return DependencyThreadPool.createDependencyTasks(tasks, dep);
	}

	public Callable<Object> getBuildTask(FrameBlock in) {
		throw new DMLRuntimeException("Trying to get the Build task of an Encoder which does not require building");
	}

	public Callable<Object> getPartialBuildTask(FrameBlock in, int startRow, int blockSize,
		HashMap<Integer, Object> ret) {
		throw new DMLRuntimeException(
			"Trying to get the PartialBuild task of an Encoder which does not support  partial building");
	}

	public Callable<Object> getPartialMergeBuildTask(HashMap<Integer, ?> ret) {
		throw new DMLRuntimeException(
			"Trying to get the BuildMergeTask task of an Encoder which does not support partial building");
	}

	public List<DependencyTask<?>> getApplyTasks(FrameBlock in, MatrixBlock out, int outputCol) {
		return getApplyTasks(in, null, out, outputCol);
	}

	public List<DependencyTask<?>> getApplyTasks(MatrixBlock in, MatrixBlock out, int outputCol) {
		return getApplyTasks(null, in, out, outputCol);
	}

	private List<DependencyTask<?>> getApplyTasks(FrameBlock inF, MatrixBlock inM, MatrixBlock out, int outputCol){
		List<Callable<Object>> tasks = new ArrayList<>();
		List<List<? extends Callable<?>>> dep = null;
		if ((inF != null && inM != null) || (inF == null && inM == null))
			throw new DMLRuntimeException("getApplyTasks needs to be called with either FrameBlock input " +
					"or MatrixBlock input");
		int nRows = inF == null ? inM.getNumRows() : inF.getNumRows();
		int[] blockSizes = getBlockSizes(nRows, getNumApplyRowPartitions());
		for(int startRow = 0, i = 0; i < blockSizes.length; startRow+=blockSizes[i], i++){
			if(inF != null)
				if(out.isInSparseFormat())
					tasks.add(getSparseTask(inF, out, outputCol, startRow, blockSizes[i]));
				else
					tasks.add(new ColumnApplyTask<>(this, inF, out, outputCol, startRow, blockSizes[i]));
			else
			if(out.isInSparseFormat())
				tasks.add(getSparseTask(inM, out, outputCol, startRow, blockSizes[i]));
			else
				tasks.add(new ColumnApplyTask<>(this, inM, out, outputCol, startRow, blockSizes[i]));
		}
		if(tasks.size() > 1){
			dep = new ArrayList<>(Collections.nCopies(tasks.size(), null));
			tasks.add(() -> null);  // Empty task as barrier
			dep.add(tasks.subList(0, tasks.size()-1));
		}

		return DependencyThreadPool.createDependencyTasks(tasks, dep);
	}

	protected abstract ColumnApplyTask<? extends ColumnEncoder> 
			getSparseTask(FrameBlock in, MatrixBlock out, int outputCol, int startRow, int blk);

	protected abstract ColumnApplyTask<? extends ColumnEncoder> 
			getSparseTask(MatrixBlock in, MatrixBlock out, int outputCol, int startRow, int blk);

	protected int getNumApplyRowPartitions(){
		return APPLY_ROW_BLOCKS_PER_COLUMN;
	}

	protected int getNumBuildRowPartitions(){
		return BUILD_ROW_BLOCKS_PER_COLUMN;
	}

	public enum EncoderType {
		Recode, FeatureHash, PassThrough, Bin, Dummycode, Omit, MVImpute, Composite
	}

	/*
	 * This is the base Task for each column apply. If no custom "getApplyTasks" is implemented in an Encoder this task
	 * will be used.
	 */
	protected static class ColumnApplyTask<T extends ColumnEncoder> implements Callable<Object> {

		protected final T _encoder;
		protected final FrameBlock _inputF;
		protected final MatrixBlock _inputM;
		protected final MatrixBlock _out;
		protected final int _outputCol;
		protected final int _startRow;
		protected final int _blk;

		protected ColumnApplyTask(T encoder, FrameBlock input, MatrixBlock out, int outputCol){
			this(encoder, input, out, outputCol, 0, -1);
		}

		protected ColumnApplyTask(T encoder, MatrixBlock input, MatrixBlock out, int outputCol){
			this(encoder, input, out, outputCol, 0, -1);
		}

		protected ColumnApplyTask(T encoder, FrameBlock input, MatrixBlock out, int outputCol, int startRow, int blk) {
			this(encoder, input, null, out, outputCol, startRow, blk);
		}

		protected ColumnApplyTask(T encoder, MatrixBlock input, MatrixBlock out, int outputCol, int startRow, int blk) {
			this(encoder, null, input, out, outputCol, startRow, blk);
		}
		private  ColumnApplyTask(T encoder, FrameBlock inputF, MatrixBlock inputM, MatrixBlock out, int outputCol,
								 int startRow, int blk){
			_encoder = encoder;
			_inputM = inputM;
			_inputF = inputF;
			_out = out;
			_outputCol = outputCol;
			_startRow = startRow;
			_blk = blk;
		}

		@Override
		public Object call() throws Exception {
			assert _outputCol >= 0;
			if(_out.isInSparseFormat()){
				// this is an issue since most sparse Tasks modify the sparse structure so normal get and set calls are
				// not possible.
				throw new DMLRuntimeException("ColumnApplyTask called although output is in sparse format.");
			}
			if(_inputF == null)
				_encoder.apply(_inputM, _out, _outputCol, _startRow, _blk);
			else
				_encoder.apply(_inputF, _out, _outputCol, _startRow, _blk);
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<Encoder: " + _encoder.getClass().getSimpleName() + "; ColId: "
				+ _encoder._colID + ">";
		}

	}
}
