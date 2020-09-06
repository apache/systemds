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

import org.apache.sysds.runtime.util.IndexRange;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Class used for feature hashing transformation of frames. 
 */
public class EncoderFeatureHash extends Encoder
{
	private static final long serialVersionUID = 7435806042138687342L;
	private long _K;

	public EncoderFeatureHash(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException {
		super(null, clen);
		_colList = TfMetaUtils.parseJsonIDList(parsedSpec, colnames, TfMethod.HASH.toString(), minCol, maxCol);
		_K = getK(parsedSpec);
	}
	
	public EncoderFeatureHash(int[] colList, int clen, long K) {
		super(colList, clen);
		_K = K;
	}
	
	public EncoderFeatureHash() {
		super(new int[0], 0);
		_K = 0;
	}
	
	/**
	 * Get K value used for calculation during feature hashing from parsed specifications.
	 * @param parsedSpec parsed specifications
	 * @return K value
	 * @throws JSONException
	 */
	private static long getK(JSONObject parsedSpec) throws JSONException {
		//TODO generalize to different k per feature
		return parsedSpec.getLong("K");
	}
	
	private long getCode(String key) {
		return key.hashCode() % _K;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		if( !isApplicable() )
			return out;
		
		//apply only
		apply(in, out);
		
		return out;
	}

	@Override
	public void build(FrameBlock in) {
		//do nothing (no meta data other than K)
	}

	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		//apply feature hashing column wise
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			for( int i=0; i<in.getNumRows(); i++ ) {
				Object okey = in.get(i, colID-1);
				String key = (okey!=null) ? okey.toString() : null;
				long code = getCode(key);
				out.quickSetValue(i, colID-1,
					(code >= 0) ? code : Double.NaN);
			}
		}
		return out;
	}
	
	@Override
	public Encoder subRangeEncoder(IndexRange ixRange) {
		int[] colList = subRangeColList(ixRange);
		if(colList.length == 0)
			// empty encoder -> sub range encoder does not exist
			return null;
		return new EncoderFeatureHash(colList, (int) ixRange.colSpan(), _K);
	}
	
	@Override
	public void mergeAt(Encoder other, int row, int col) {
		if(other instanceof EncoderFeatureHash) {
			mergeColumnInfo(other, col);
			if (((EncoderFeatureHash) other)._K != 0 && _K == 0)
				_K = ((EncoderFeatureHash) other)._K;
			return;
		}
		super.mergeAt(other, row, col);
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		if( !isApplicable() )
			return meta;
		
		meta.ensureAllocatedColumns(1);
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			meta.set(0, colID-1, String.valueOf(_K));
		}
		
		return meta;
	}

	@Override
	public void initMetaData( FrameBlock meta ) {
		if( meta == null || meta.getNumRows()<=0 )
			return;
		
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			_K = UtilFunctions.parseToLong(meta.get(0, colID-1).toString());
		}
	}
}
