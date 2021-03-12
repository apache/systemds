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
import java.util.List;
import java.util.Objects;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Simple composite encoder that applies a list of encoders 
 * in specified order. By implementing the default encoder API
 * it can be used as a drop-in replacement for any other encoder. 
 * 
 */
// TODO assert each type of encoder can only be present once
public class ColumnEncoderComposite extends ColumnEncoder
{
	private static final long serialVersionUID = -8473768154646831882L;
	
	private List<ColumnEncoder> _columnEncoders = null;
	private FrameBlock _meta = null;

	public ColumnEncoderComposite() {
		super( -1);
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders, FrameBlock meta) {
		super(-1);
		if(!(columnEncoders.size() > 0 && columnEncoders.stream().allMatch((encoder -> encoder._colID == columnEncoders.get(0)._colID))))
			throw new DMLRuntimeException("Tried to create Composite Encoder with no encoders or mismatching columIDs");
		_colID = columnEncoders.get(0)._colID;
		_meta = meta;
		_columnEncoders = columnEncoders;
	}

	public ColumnEncoderComposite(List<ColumnEncoder> columnEncoders){
		this(columnEncoders, null);
	}

	public ColumnEncoderComposite(ColumnEncoder columnEncoder){
		super(columnEncoder._colID);
		_columnEncoders = new ArrayList<>();
		_columnEncoders.add(columnEncoder);
	}

	public List<ColumnEncoder> getEncoders() {
		return _columnEncoders;
	}
	
	public ColumnEncoder getEncoder(Class<?> type) {
		for( ColumnEncoder columnEncoder : _columnEncoders) {
			if( columnEncoder.getClass().equals(type) )
				return columnEncoder;
		}
		return null;
	}
	
	public boolean isEncoder(int colID, Class<?> type) {
		for( ColumnEncoder columnEncoder : _columnEncoders) {
			if( columnEncoder.getClass().equals(type) && columnEncoder._colID == colID)
				return true;
		}
		return false;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		try {
			//build meta data first (for all encoders)
			for( ColumnEncoder columnEncoder : _columnEncoders)
				columnEncoder.build(in);
			
			//propagate meta data 
			_meta = new FrameBlock(in.getNumColumns(), ValueType.STRING);
			for( ColumnEncoder columnEncoder : _columnEncoders)
				_meta = columnEncoder.getMetaData(_meta);
			for( ColumnEncoder columnEncoder : _columnEncoders)
				columnEncoder.initMetaData(_meta);
			
			//apply meta data
			for( ColumnEncoder columnEncoder : _columnEncoders)
				out = columnEncoder.apply(in, out);
		}
		catch(Exception ex) {
			LOG.error("Failed transform-encode frame with \n" + this);
			throw ex;
		}
		
		return out;
	}

	@Override
	public void build(FrameBlock in) {
		for( ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.build(in);
	}
	
	@Override
	public void prepareBuildPartial() {
		for( ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.prepareBuildPartial();
	}
	
	@Override
	public void buildPartial(FrameBlock in) {
		for( ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.buildPartial(in);
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		try {
			for( ColumnEncoder columnEncoder : _columnEncoders)
				out = columnEncoder.apply(in, out);
		}
		catch(Exception ex) {
			LOG.error("Failed to transform-apply frame with \n" + this);
			throw ex;
		}
		return out;
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		ColumnEncoderComposite that = (ColumnEncoderComposite) o;
		return _columnEncoders.equals(that._columnEncoders)
			&& Objects.equals(_meta, that._meta);
	}

	@Override
	public int hashCode() {
		return Objects.hash(_columnEncoders, _meta);
	}

	@Override
	public void mergeAt(ColumnEncoder other, int row) {
		if (other instanceof ColumnEncoderComposite) {
			ColumnEncoderComposite otherComposite = (ColumnEncoderComposite) other;
			assert otherComposite._colID == _colID;
			// TODO maybe assert that the _encoders never have the same type of encoder twice or more
			for (ColumnEncoder otherEnc : otherComposite.getEncoders()) {
				boolean mergedIn = false;
				for (ColumnEncoder columnEncoder : _columnEncoders) {
					if (columnEncoder.getClass() == otherEnc.getClass()) {
						columnEncoder.mergeAt(otherEnc, row);
						mergedIn = true;
						break;
					}
				}
				if(!mergedIn) {
					throw new DMLRuntimeException("Tried to merge in encoder of class that is not present in "
						+ "EncoderComposite: " + otherEnc.getClass().getSimpleName());
				}
			}
			// update dummycode encoder domain sizes based on distinctness information from other encoders
			for (ColumnEncoder columnEncoder : _columnEncoders) {
				if (columnEncoder instanceof ColumnEncoderDummycode) {
					((ColumnEncoderDummycode) columnEncoder).updateDomainSizes(_columnEncoders);
					return;
				}
			}
			return;
		}
		for (ColumnEncoder columnEncoder : _columnEncoders) {
			if (columnEncoder.getClass() == other.getClass()) {
				columnEncoder.mergeAt(other, row);
				// update dummycode encoder domain sizes based on distinctness information from other encoders
				for (ColumnEncoder encDummy : _columnEncoders) {
					if (encDummy instanceof ColumnEncoderDummycode) {
						((ColumnEncoderDummycode) encDummy).updateDomainSizes(_columnEncoders);
						return;
					}
				}
				return;
			}
		}
		super.mergeAt(other, row);
	}
	
	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		for(ColumnEncoder enc : _columnEncoders) {
			enc.updateIndexRanges(beginDims, endDims);
		}
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		if( _meta != null )
			return _meta;
		for( ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.getMetaData(out);
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock out) {
		for( ColumnEncoder columnEncoder : _columnEncoders)
			columnEncoder.initMetaData(out);
	}
	
	@Override
	public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
		//determine if dummycode encoder exists
		ColumnEncoderDummycode dummy = null;
		for( ColumnEncoder columnEncoder : _columnEncoders)
			if( columnEncoder instanceof ColumnEncoderDummycode)
				dummy = (ColumnEncoderDummycode) columnEncoder;
		//computed shifted start positions
		if( dummy != null ) {
			//delete to dummycode encoder
			out = dummy.getColMapping(meta, out);
		}
		//use simple 1-1 mapping
		else {
			for(int i=0; i<out.getNumRows(); i++) {
				out.quickSetValue(i, 0, i+1);
				out.quickSetValue(i, 1, i+1);
				out.quickSetValue(i, 2, i+1);
			}
		}
		
		return out;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("CompositeEncoder("+ _columnEncoders.size()+"):\n");
		for( ColumnEncoder columnEncoder : _columnEncoders) {
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
		if (in.readBoolean()) {
			FrameBlock meta = new FrameBlock();
			meta.readFields(in);
			_meta = meta;
		}
	}

	@Override
	public void shiftOutCol(int shift){
		for(ColumnEncoder encoder: _columnEncoders){
			encoder.shiftOutCol(shift);
		}
		_colID += shift;
	}
}
