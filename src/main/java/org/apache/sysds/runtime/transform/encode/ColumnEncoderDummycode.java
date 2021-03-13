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
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ColumnEncoderDummycode extends ColumnEncoder
{
	private static final long serialVersionUID = 5832130477659116489L;

	public int _domainSize = -1;  // length = #of dummycoded columns
	protected long _clen = 0;

	public ColumnEncoderDummycode() {
		super(-1);
	}

	public ColumnEncoderDummycode(int colID, long clen) {
		super(colID);
		_clen = clen;
	}

	public ColumnEncoderDummycode(int colID, int domainSize, long clen) {
		super(colID);
		_domainSize = domainSize;
		_clen = clen;
	}

	
	@Override
	public MatrixBlock encode(FrameBlock in) {
		return apply(in);
	}

	public MatrixBlock encode(MatrixBlock in) {
		return apply(in);
	}

	@Override
	public void build(FrameBlock in) {
		//do nothing
	}


	@Override
	public MatrixBlock apply(FrameBlock in){
		if(!in.getSchema()[_colID-1].isNumeric())
			throw new DMLRuntimeException("DummyCoder input with non numeric value");
		MatrixBlock in_ = new MatrixBlock(in.getNumRows(), 1, false);
		for(int i = 0; i < in.getNumRows(); i++){
			in_.quickSetValue(i, 0, UtilFunctions.objectToDouble(in.getSchema()[i], in.get(i, _colID-1)));
		}
		return apply(in_);
	}

	public MatrixBlock apply(MatrixBlock in) {
		assert in.getNumColumns() == 1;
		//allocate output in dense or sparse representation
		final boolean sparse = MatrixBlock.evalSparseFormatInMemory(
			in.getNumRows(), _domainSize, in.getNonZeros());
		MatrixBlock ret = new MatrixBlock(in.getNumRows(), _domainSize, sparse);
		
		//append dummy coded or unchanged values to output
		final int clen = in.getNumColumns();
		for( int i=0; i<in.getNumRows(); i++ ) {
			double val = in.quickGetValue(i, 0);
			ret.appendValue(i, (int)val-1, 1);
		}
		return ret;
	}


	@Override
	public void mergeAt(ColumnEncoder other, int row) {
		if(other instanceof ColumnEncoderDummycode) {
			assert other._colID == _colID;
			// temporary, will be updated later
			_domainSize = 0;
			return;
		}
		super.mergeAt(other, row);
	}
	
	@Override
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		long[] initialBegin = Arrays.copyOf(beginDims, beginDims.length);
		long[] initialEnd = Arrays.copyOf(endDims, endDims.length);
		// 1-based vs 0-based
		if(_colID < initialBegin[1] + 1) {
			// new columns inserted left of the columns of this partial (federated) block
			beginDims[1] += _domainSize - 1;
			endDims[1] += _domainSize - 1;
		}
		else if(_colID < initialEnd[1] + 1) {
			// new columns inserted in this (federated) block
			endDims[1] += _domainSize - 1;
		}
	}
	
	public void updateDomainSizes(List<ColumnEncoder> columnEncoders) {
		if(_colID == -1)
			return;
		for (ColumnEncoder columnEncoder : columnEncoders) {
			int distinct = -1;
			if (columnEncoder instanceof ColumnEncoderRecode) {
				ColumnEncoderRecode columnEncoderRecode = (ColumnEncoderRecode) columnEncoder;
				distinct = columnEncoderRecode.numDistinctValues();
			}
			else if (columnEncoder instanceof ColumnEncoderBin) {
				distinct = ((ColumnEncoderBin) columnEncoder)._numBin;
			}
			
			if (distinct != -1) {
					_domainSize = distinct;
			}
		}
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		return meta;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//initialize domain sizes and output num columns
		_domainSize = -1;
		_domainSize= (int)meta.getColumnMetadata()[_colID-1].getNumDistinct();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeLong(_clen);
		out.writeInt(_domainSize);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_clen = in.readLong();
		_domainSize = in.readInt();
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		ColumnEncoderDummycode that = (ColumnEncoderDummycode) o;
		return _colID == that._colID
			&& (_domainSize == that._domainSize);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(_colID);
		result = 31 * result + Objects.hashCode(_domainSize);
		return result;
	}

	public int getDomainSize() {
		return _domainSize;
	}
}
