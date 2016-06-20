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

package org.apache.sysml.runtime.transform.decode;

import java.util.HashMap;
import java.util.List;

import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * Simple atomic decoder for recoded columns. This decoder builds internally
 * inverted recode maps from the given frame meta data. 
 *  
 */
public class DecoderRecode extends Decoder
{
	private static final long serialVersionUID = -3784249774608228805L;

	private HashMap<Long, Object>[] _rcMaps = null;
	private boolean _onOut = false;
	
	protected DecoderRecode(List<ValueType> schema, boolean onOut, int[] rcCols) {
		super(schema, rcCols);
		_onOut = onOut;
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		if( _onOut ) { //recode on output (after dummy)
			for( int i=0; i<in.getNumRows(); i++ ) {
				for( int j=0; j<_colList.length; j++ ) {
					int colID = _colList[j];
					double val = UtilFunctions.objectToDouble(
							out.getSchema().get(colID-1), out.get(i, colID-1));
					long key = UtilFunctions.toLong(val);
					out.set(i, colID-1, _rcMaps[j].get(key));
				}
			}
		}
		else { //recode on input (no dummy)
			out.ensureAllocatedColumns(in.getNumRows());
			for( int i=0; i<in.getNumRows(); i++ ) {
				for( int j=0; j<_colList.length; j++ ) {
					double val = in.quickGetValue(i, _colList[j]-1);
					long key = UtilFunctions.toLong(val);
					out.set(i, _colList[j]-1, _rcMaps[j].get(key));
				}
			}
		}
		return out;
	}

	@Override
	@SuppressWarnings("unchecked")
	public void initMetaData(FrameBlock meta) {
		//initialize recode maps according to schema
		_rcMaps = new HashMap[_colList.length];
		for( int j=0; j<_colList.length; j++ ) {
			HashMap<Long, Object> map = new HashMap<Long, Object>();
			for( int i=0; i<meta.getNumRows(); i++ ) {
				if( meta.get(i, _colList[j]-1)==null )
					break; //reached end of recode map
				String[] tmp = meta.get(i, _colList[j]-1).toString().split(Lop.DATATYPE_PREFIX);				
				Object obj = UtilFunctions.stringToObject(_schema.get(_colList[j]-1), tmp[0]);
				map.put(Long.parseLong(tmp[1]), obj);				
			}
			_rcMaps[j] = map;
		}
	}
	
	/**
	 * Parses a line of <token, ID, count> into <token, ID> pairs, where 
	 * quoted tokens (potentially including separators) are supported.
	 * 
	 * @param entry
	 * @param pair
	 */
	public static void parseRecodeMapEntry(String entry, Pair<String,String> pair) {
		int ixq = entry.lastIndexOf('"');
		String token = UtilFunctions.unquote(entry.substring(0,ixq+1));
		int idx = ixq+2;
		while(entry.charAt(idx) != TfUtils.TXMTD_SEP.charAt(0))
			idx++;
		String id = entry.substring(ixq+2,idx); 
		pair.set(token, id);
	}
}
