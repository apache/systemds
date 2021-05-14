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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

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

	// map to keep track of which encoder has how many build tasks
	private Map<ColumnEncoder, Integer> _partialBuildTaskMap;

	public ColumnEncoderComposite() {
		super(-1);
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders, FrameBlock meta) {
		super(-1);
		if(!(columnEncoders.size() > 0 &&
			columnEncoders.stream().allMatch((encoder -> encoder._colID == columnEncoders.get(0)._colID))))
			throw new DMLRuntimeException("Tried to create Composite Encoder with no encoders or mismatching columIDs");
		_colID = columnEncoders.get(0)._colID;
		_meta = meta;
		_columnEncoders = columnEncoders;
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders) {
		this(columnEncoders, null);
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
	public void build(FrameBlock in) {
		for(ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.build(in);
	}

	@Override
	public List<Callable<Object>> getPartialBuildTasks(FrameBlock in, int blockSize) {
		List<Callable<Object>> tasks = new ArrayList<>();
		_partialBuildTaskMap = new HashMap<>();
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			List<Callable<Object>> _tasks = columnEncoder.getPartialBuildTasks(in, blockSize);
			if(_tasks != null)
				tasks.addAll(_tasks);
			_partialBuildTaskMap.put(columnEncoder, _tasks != null ? _tasks.size() : 0);
		}
		return tasks.size() == 0 ? null : tasks;
	}

	@Override
	public void mergeBuildPartial(List<Future<Object>> futurePartials, int start, int end)
		throws ExecutionException, InterruptedException {
		int endLocal;
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			endLocal = start + _partialBuildTaskMap.get(columnEncoder);
			columnEncoder.mergeBuildPartial(futurePartials, start, endLocal);
			start = endLocal;
			if(start >= end)
				break;
		}
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
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol) {
		return apply(in, out, outputCol, 0, -1);
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		try {
			for(int i = 0; i < _columnEncoders.size(); i++) {
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
			LOG.error("Failed to transform-apply frame with \n" + this);
			throw ex;
		}
		return out;
	}

	@Override
	public MatrixBlock apply(MatrixBlock in, MatrixBlock out, int outputCol, int rowStart, int blk) {
		try {
			for(int i = 0; i < _columnEncoders.size(); i++) {
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
			LOG.error("Failed to transform-apply matrix with \n" + this);
			throw ex;
		}
		return in;
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
		else {
			addEncoder(other);
		}
		updateAllDCEncoders();
	}

	public void updateAllDCEncoders(){
		// update dummycode encoder domain sizes based on distinctness information from other encoders
		ColumnEncoderDummycode dc = getEncoder(ColumnEncoderDummycode.class);
		if(dc != null)
			dc.updateDomainSizes(_columnEncoders);
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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompositeEncoder(").append(_columnEncoders.size()).append("):\n");
		for(ColumnEncoder columnEncoder : _columnEncoders) {
			sb.append("-- ");
			sb.append(columnEncoder.getClass().getSimpleName());
			sb.append(": ");
			sb.append(columnEncoder._colID);
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

	@Override
	public void shiftCol(int columnOffset) {
		super.shiftCol(columnOffset);
		_columnEncoders.forEach(e -> e.shiftCol(columnOffset));
	}
}
