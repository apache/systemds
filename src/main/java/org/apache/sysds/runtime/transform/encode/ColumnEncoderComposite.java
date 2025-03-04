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

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DependencyTask;
import org.apache.sysds.runtime.util.DependencyThreadPool;

/**
 * Simple composite encoder that applies a list of encoders in specified order. By implementing the default encoder API
 * it can be used as a drop-in replacement for any other encoder.
 * 
 */
// TODO assert each type of encoder can only be present once
public class ColumnEncoderComposite extends ColumnEncoder {
	private static final long serialVersionUID = -8473768154646831882L;

	private List<ColumnEncoder> _columnEncoders = null;
	private FrameBlock _meta = null;
	private long avgEntrySize = 0L;

	// map to keep track of which encoder has how many build tasks
	//private Map<ColumnEncoder, Integer> _partialBuildTaskMap;

	public ColumnEncoderComposite() {
		super(-1);
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders, FrameBlock meta) {
		super(-1);
		if(!(!columnEncoders.isEmpty() &&
			columnEncoders.stream().allMatch((encoder -> encoder._colID == columnEncoders.get(0)._colID))))
			throw new DMLRuntimeException("Tried to create Composite Encoder with no encoders or mismatching columnIDs");
		_colID = columnEncoders.get(0)._colID;
		_meta = meta;
		_columnEncoders = columnEncoders;
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders) {
		this(columnEncoders, null);
	}
	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders, int colID) {
		super(colID);
		_columnEncoders = columnEncoders;
		_meta = null;
	}

	public ColumnEncoderComposite(ColumnEncoder columnEncoder) {
		super(columnEncoder._colID);
		_columnEncoders = new ArrayList<>();
		_columnEncoders.add(columnEncoder);
	}

	public List<ColumnEncoder> getEncoders() {
		return _columnEncoders;
	}

	public <T extends ColumnEncoder> T getEncoder(Class<T> type) {
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			if(columnEncoder.getClass().equals(type))
				return type.cast(columnEncoder);
		}
		return null;
	}

	public boolean isEncoder(int colID, Class<?> type) {
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			if(columnEncoder.getClass().equals(type) && columnEncoder._colID == colID)
				return true;
		}
		return false;
	}

	@Override
	public void build(CacheBlock<?> in) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.build(in);
	}

	@Override
	public void build(CacheBlock<?> in, Map<Integer, double[]> equiHeightMaxs) {
		if(equiHeightMaxs == null)
			build(in);
		else{
			for(ColumnEncoder columnEncoder : _columnEncoders)
				if(columnEncoder instanceof ColumnEncoderBin && ((ColumnEncoderBin) columnEncoder).getBinMethod() == ColumnEncoderBin.BinMethod.EQUI_HEIGHT) {
					columnEncoder.build(in, equiHeightMaxs.get(columnEncoder.getColID()));
				} else {
					columnEncoder.build(in);
				}
		}
	}

	@Override
	public List<DependencyTask<?>> getApplyTasks(CacheBlock<?> in, MatrixBlock out, int outputCol, int[] sparseRowPointerOffsets) {
		List<DependencyTask<?>> tasks = new ArrayList<>();
		List<Integer> sizes = new ArrayList<>();
		for(int i = 0; i < _columnEncoders.size(); i++) {
			List<DependencyTask<?>> t;
			if(i == 0) {
				// 1. encoder writes data into MatrixBlock Column all others use this column for further encoding
				t = _columnEncoders.get(i).getApplyTasks(in, out, outputCol, sparseRowPointerOffsets);
			}
			else {
				t = _columnEncoders.get(i).getApplyTasks(out, out, outputCol, sparseRowPointerOffsets);
			}
			if(t == null)
				continue;
			sizes.add(t.size());
			tasks.addAll(t);
		}

		List<List<? extends Callable<?>>> dep = new ArrayList<>(Collections.nCopies(tasks.size(), null));

		for(int c = 0, i = sizes.get(c); i < tasks.size(); c++, i += sizes.get(c)) {
			for(int k = i; k < i + sizes.get(c + 1); k++) {
				dep.set(k, tasks.subList(i - 1, i));
			}
		}

		tasks = DependencyThreadPool.createDependencyTasks(tasks, dep);
		return tasks;
	}

	@Override
	protected ColumnApplyTask<? extends ColumnEncoder> 
		getSparseTask(CacheBlock<?> in, MatrixBlock out, int outputCol, int startRow, int blk) {
		throw new NotImplementedException();
	}

	@Override
	public List<DependencyTask<?>> getBuildTasks(CacheBlock<?> in) {
		List<DependencyTask<?>> tasks = new ArrayList<>();
		Map<Integer[], Integer[]> depMap = null;
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			List<DependencyTask<?>> t = columnEncoder.getBuildTasks(in);
			if(t == null)
				continue;
			// Linear execution between encoders so they can't be built in parallel
			if(!tasks.isEmpty()) {
				// TODO: is that still needed? currently there is no CompositeEncoder with 2 encoders with build phase
				// avoid unnecessary map initialization
				depMap = (depMap == null) ? new HashMap<>() : depMap;
				// This workaround is needed since sublist is only valid for effective final lists,
				// otherwise the view breaks
				depMap.put(new Integer[] {tasks.size(), tasks.size() + t.size()},
					new Integer[] {tasks.size() - 1, tasks.size()});
			}
			tasks.addAll(t);
		}

		List<List<? extends Callable<?>>> dep = new ArrayList<>(Collections.nCopies(tasks.size(), null));
		DependencyThreadPool.createDependencyList(tasks, depMap, dep);
		// If DC is required, add an UpdateDC task to update the domainsize as the last task
		// Only for RC build, UpdateDC must depends on the Build task, other can be independent.
		if(hasEncoder(ColumnEncoderDummycode.class)) {
			tasks.add(DependencyThreadPool.createDependencyTask(new ColumnCompositeUpdateDCTask(this)));
			if (_columnEncoders.get(0) instanceof ColumnEncoderRecode) {
				dep.add(tasks.subList(tasks.size() - 2, tasks.size() - 1));
				return DependencyThreadPool.createDependencyTasks(tasks, dep);
			}
		}
		return DependencyThreadPool.createDependencyTasks(tasks, null);
	}

	@Override
	public void prepareBuildPartial() {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.prepareBuildPartial();
	}

	@Override
	public void buildPartial(FrameBlock in) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.buildPartial(in);
	}

	@Override
	public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		try {
			for(int i = 0; i < _columnEncoders.size(); i++) {
				// set sparseRowPointerOffset in the encoder
				_columnEncoders.get(i).sparseRowPointerOffset = this.sparseRowPointerOffset;
				if(i == 0) {
					// 1. encoder writes data into MatrixBlock Column all others use this column for further encoding
					_columnEncoders.get(i).apply(in, out, outputCol, rowStart, blk);
				}
				else {
					_columnEncoders.get(i).apply(out, out, outputCol, rowStart, blk);
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Failed to transform-apply frame with \n" + this, ex);
		}
		return out;
	}

	@Override
	protected double getCode(CacheBlock<?> in, int row) {
		throw new DMLRuntimeException("CompositeEncoder does not have a Code");
	}

	@Override
	protected double[] getCodeCol(CacheBlock<?> in, int startInd, int endInd, double[] tmp) {
		throw new DMLRuntimeException("CompositeEncoder does not have a Code");
	}

	@Override
	protected TransformType getTransformType() {
		return TransformType.N_A;
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		ColumnEncoderComposite that = (ColumnEncoderComposite) o;
		return _columnEncoders.equals(that._columnEncoders) && Objects.equals(_meta, that._meta);
	}

	@Override
	public int hashCode() {
		return Objects.hash(_columnEncoders, _meta);
	}

	@Override
	public void mergeAt(ColumnEncoder other) {
		if(other instanceof ColumnEncoderComposite) {
			ColumnEncoderComposite otherComposite = (ColumnEncoderComposite) other;
			assert otherComposite._colID == _colID;
			// TODO maybe assert that the _encoders never have the same type of encoder twice or more
			for(ColumnEncoder otherEnc : otherComposite.getEncoders()) {
				addEncoder(otherEnc);
			}
		}
		else
			addEncoder(other);

		updateAllDCEncoders();
	}

	public void updateAllDCEncoders() {
		// update dummycode encoder domain sizes based on distinctness information from other encoders
		ColumnEncoderDummycode dc = getEncoder(ColumnEncoderDummycode.class);
		if(dc != null)
			dc.updateDomainSizes(_columnEncoders);
		ColumnEncoderUDF udf = getEncoder(ColumnEncoderUDF.class);
		if (udf != null && dc != null)
			udf.updateDomainSizes(_columnEncoders);
	}

	public void addEncoder(ColumnEncoder other) {
		ColumnEncoder encoder = getEncoder(other.getClass());
		assert _colID == other._colID;
		if(encoder != null)
			encoder.mergeAt(other);
		else {
			_columnEncoders.add(other);
			_columnEncoders.sort(null);
		}
	}

	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims, int colOffset) {
		for(ColumnEncoder enc : _columnEncoders) {
			enc.updateIndexRanges(beginDims, endDims, colOffset);
		}
	}

	@Override
	public void allocateMetaData(FrameBlock meta) {
		if(_meta != null)
			return;
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.allocateMetaData(meta);
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		if(_meta != null)
			return _meta;
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.getMetaData(out);
		return out;
	}

	@Override
	public void initMetaData(FrameBlock out) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.initMetaData(out);
	}

	//pass down init to actual encoders, only ColumnEncoderWordEmbedding has actually implemented the init method
	public void initEmbeddings(MatrixBlock embeddings){
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.initEmbeddings(embeddings);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompositeEncoder(").append(_columnEncoders.size()).append("):\n");
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			sb.append("-- ");
			sb.append(columnEncoder);
			sb.append("\n");
		}
		return sb.toString();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(_columnEncoders.size());
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			out.writeInt(columnEncoder._colID);
			out.writeByte(EncoderFactory.getEncoderType(columnEncoder));
			columnEncoder.writeExternal(out);
		}
		out.writeBoolean(_meta != null);
		if(_meta != null)
			_meta.write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		int encodersSize = in.readInt();
		_columnEncoders = new ArrayList<>();
		for(int i = 0; i < encodersSize; i++) {
			int colID = in.readInt();
			ColumnEncoder columnEncoder = EncoderFactory.createInstance(in.readByte());
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

	public <T extends ColumnEncoder> boolean hasEncoder(Class<T> type) {
		return _columnEncoders.stream().anyMatch(encoder -> encoder.getClass().equals(type));
	}

	public <T extends ColumnEncoder> boolean hasBuild() {
		for (ColumnEncoder e : _columnEncoders)
			if (e instanceof ColumnEncoderRecode
				|| e instanceof ColumnEncoderDummycode
				|| e instanceof ColumnEncoderBin
				|| e instanceof ColumnEncoderBagOfWords)
				return true;
		return false;
	}

	public void computeMapSizeEstimate(CacheBlock<?> in, int[] sampleIndices) {
		int estNumDist = 0;
		for (ColumnEncoder e : _columnEncoders){
			if (e.getClass().equals(ColumnEncoderRecode.class) || e.getClass().equals(ColumnEncoderBagOfWords.class)) {
				e.computeMapSizeEstimate(in, sampleIndices);
				estNumDist = e.getEstNumDistincts();
				this.avgEntrySize = e._avgEntrySize;
			}
		}
		long totEstSize = _columnEncoders.stream().mapToLong(ColumnEncoder::getEstMetaSize).sum();
		setEstMetaSize(totEstSize);
		setEstNumDistincts(estNumDist);

	}

	public long getAvgEntrySize(){
		return this.avgEntrySize;
	}

	public void setNumPartitions(int nBuild, int nApply) {
			_columnEncoders.forEach(e -> {
				e.setBuildRowBlocksPerColumn(nBuild);
				if (e.getClass().equals(ColumnEncoderUDF.class))
					e.setApplyRowBlocksPerColumn(1);
				else
					e.setApplyRowBlocksPerColumn(nApply);
			});
	}

	@Override
	public void shiftCol(int columnOffset) {
		super.shiftCol(columnOffset);
		_columnEncoders.forEach(e -> e.shiftCol(columnOffset));
	}

	protected boolean containsZeroOut(){
		for(int i = 0; i < _columnEncoders.size(); i++)
			if(_columnEncoders.get(i).containsZeroOut())
				return true;
		return false;
	}

	@Override
	public int getDomainSize() {
		return _columnEncoders.stream()//
			.map(ColumnEncoder::getDomainSize).reduce((a,x) -> Integer.max(a,x)).get();
	}


	public boolean isRecodeToDummy(){
		return _columnEncoders.size() == 2 //
			&& _columnEncoders.get(0) instanceof ColumnEncoderRecode //
			&& _columnEncoders.get(1) instanceof ColumnEncoderDummycode;
	}

	public boolean isRecode(){
		return _columnEncoders.size() == 1 //
		&& _columnEncoders.get(0) instanceof ColumnEncoderRecode;
	}

	public boolean isPassThrough(){
		return _columnEncoders.size() == 1 //
			&& _columnEncoders.get(0) instanceof ColumnEncoderPassThrough;
	}

	public boolean isBin(){
		return _columnEncoders.size() == 1//
			&& _columnEncoders.get(0) instanceof ColumnEncoderBin;
	}

	public boolean isBinToDummy(){
		return _columnEncoders.size() == 2//
			&& _columnEncoders.get(0) instanceof ColumnEncoderBin//
			&& _columnEncoders.get(1) instanceof ColumnEncoderDummycode;
	}

	public boolean isHash() {
		return _columnEncoders.size() == 1//
			&& _columnEncoders.get(0) instanceof ColumnEncoderFeatureHash;//
	}

	public boolean isHashToDummy() {
		return _columnEncoders.size() == 2//
			&& _columnEncoders.get(0) instanceof ColumnEncoderFeatureHash//
			&& _columnEncoders.get(1) instanceof ColumnEncoderDummycode;
	}

	private static class ColumnCompositeUpdateDCTask implements Callable<Object> {

		private final ColumnEncoderComposite _encoder;

		protected ColumnCompositeUpdateDCTask(ColumnEncoderComposite encoder) {
			_encoder = encoder;
		}

		@Override
		public Void call() throws Exception {
			_encoder.updateAllDCEncoders();
			return null;
		}

		@Override
		public String toString() {
			return getClass().getSimpleName() + "<ColId: " + _encoder._colID + ">";
		}

	}

}
