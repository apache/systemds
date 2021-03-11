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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderDummycode extends Encoder 
{
	private static final long serialVersionUID = 5832130477659116489L;

	public int _domainSize = -1;  // length = #of dummycoded columns
	private long _dummycodedLength = 0; // #of columns after dummycoded
	private long _clen = 0;

	/*
	public EncoderDummycode(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException {
		super(null, clen);

		if(parsedSpec.containsKey(TfMethod.DUMMYCODE.toString())) {
			int[] collist = TfMetaUtils
				.parseJsonIDList(parsedSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol);
			initColList(collist);
		}
	}

	 */
	public EncoderDummycode() {
		super(-1);
	}
	
	public EncoderDummycode(int colID, int domainSize, long dummycodedLength, long clen) {
		super(colID);
		_domainSize = domainSize;
		_dummycodedLength = dummycodedLength;
		_clen = clen;
	}

	public int getNumCols() {
		return (int)_dummycodedLength;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		return apply(in, out);
	}

	@Override
	public void build(FrameBlock in) {
		//do nothing
	}

	// TODO Not optimal now since ret is allocated for each colum that is dummycoded.
	// TODO Optimisation across dummycoders!!! -> Matrix allocation only once
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		//allocate output in dense or sparse representation
		final boolean sparse = MatrixBlock.evalSparseFormatInMemory(
			out.getNumRows(), getNumCols(), out.getNonZeros());
		MatrixBlock ret = new MatrixBlock(out.getNumRows(), getNumCols(), sparse);
		
		//append dummy coded or unchanged values to output
		final int clen = out.getNumColumns();
		for( int i=0; i<out.getNumRows(); i++ ) {
			for(int colID=1, idx=0, ncolID=1; colID <= clen; colID++) {
				double val = out.quickGetValue(i, colID-1);
				if(colID == _colID) {
					ret.appendValue(i, ncolID-1+(int)val-1, 1);
					ncolID += _domainSize;
					idx ++;
				}
				else {
					double ptval = out.quickGetValue(i, colID-1);
					ret.appendValue(i, ncolID-1, ptval);
					ncolID ++;
				}
			}
		}
		return ret;
	}


	@Override
	public void mergeAt(Encoder other, int row) {
		if(other instanceof EncoderDummycode) {
			assert other._colID == _colID;
			// temporary, will be updated later
			_domainSize = 0;
			_dummycodedLength = _clen;
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
	
	public void updateDomainSizes(List<Encoder> encoders) {
		if(_colID == -1)
			return;
		_dummycodedLength = _clen;
		for (Encoder encoder : encoders) {
			int distinct = -1;
			if (encoder instanceof EncoderRecode) {
				EncoderRecode encoderRecode = (EncoderRecode) encoder;
				distinct = encoderRecode.numDistinctValues();
			}
			else if (encoder instanceof EncoderBin) {
				distinct = ((EncoderBin) encoder)._numBin;
			}
			
			if (distinct != -1) {
					_domainSize = distinct;
					_dummycodedLength += _domainSize - 1;
			}
		}
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//initialize domain sizes and output num columns
		_domainSize = -1;
		_dummycodedLength = _clen;
		_domainSize= (int)meta.getColumnMetadata()[_colID-1].getNumDistinct();
		_dummycodedLength += _domainSize-1;
	}
	
	@Override
	public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
		final int clen = out.getNumRows();
		for(int colID=1, idx=0, ncolID=1; colID <= clen; colID++) {
			int start = ncolID;
			if( colID == _colID ) {
				ncolID += meta.getColumnMetadata(colID-1).getNumDistinct();
				idx ++;
			}
			else {
				ncolID ++;
			}
			out.quickSetValue(colID-1, 0, colID);
			out.quickSetValue(colID-1, 1, start);
			out.quickSetValue(colID-1, 2, ncolID-1);
		}
		
		return out;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		super.writeExternal(out);
		out.writeLong(_dummycodedLength);
		out.writeInt(_domainSize);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		super.readExternal(in);
		_dummycodedLength = in.readLong();
		_domainSize = in.readInt();
	}

	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o == null || getClass() != o.getClass())
			return false;
		EncoderDummycode that = (EncoderDummycode) o;
		return _dummycodedLength == that._dummycodedLength
			&& (_domainSize == that._domainSize);
	}

	@Override
	public int hashCode() {
		int result = Objects.hash(_dummycodedLength);
		result = 31 * result + Objects.hashCode(_domainSize);
		return result;
	}
}
