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

package org.apache.sysml.runtime.transform.encode;

import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderDummycode extends Encoder 
{
	private static final long serialVersionUID = 5832130477659116489L;

	private int[] _domainSizes = null;  // length = #of dummycoded columns
	private long _dummycodedLength = 0; // #of columns after dummycoded

	public EncoderDummycode(JSONObject parsedSpec, String[] colnames, int clen) throws JSONException {
		super(null, clen);
		
		if ( parsedSpec.containsKey(TfUtils.TXMETHOD_DUMMYCODE) ) {
			int[] collist = TfMetaUtils.parseJsonIDList(parsedSpec, colnames, TfUtils.TXMETHOD_DUMMYCODE);
			initColList(collist);
		}
	}
	
	@Override
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
				if( idx < _colList.length && colID==_colList[idx] ) {
					ret.appendValue(i, ncolID-1+(int)val-1, 1);
					ncolID += _domainSizes[idx];
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
	public FrameBlock getMetaData(FrameBlock out) {
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		//initialize domain sizes and output num columns
		_domainSizes = new int[_colList.length];
		_dummycodedLength = _clen;
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			_domainSizes[j] = (int)meta.getColumnMetadata()[colID-1].getNumDistinct();
			_dummycodedLength += _domainSizes[j]-1;
		}
	}
	
	@Override
	public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
		final int clen = out.getNumRows();
		for(int colID=1, idx=0, ncolID=1; colID <= clen; colID++) {
			int start = ncolID;
			if( idx < _colList.length && colID==_colList[idx] ) {
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
}
