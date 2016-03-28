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
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * Simple atomic decoder for recoded columns. This decoder builds internally
 * inverted recode maps from the given frame meta data. 
 *  
 */
public class DecoderRecode extends Decoder
{
	private int[] _rcCols = null; //0-based
	private HashMap<Long, Object>[] _rcMaps = null;
	
	@SuppressWarnings("unchecked")
	protected DecoderRecode(List<ValueType> schema, FrameBlock meta, int[] rcCols) {
		super(schema);
		
		//initialize recode maps according to schema
		_rcCols = rcCols;
		_rcMaps = new HashMap[_rcCols.length];
		for( int j=0; j<_rcCols.length; j++ ) {
			HashMap<Long, Object> map = new HashMap<Long, Object>();
			for( int i=0; i<meta.getNumRows(); i++ ) {
				if( meta.get(i, _rcCols[j])==null )
					break; //reached end of recode map
				String[] tmp = meta.get(i, _rcCols[j]).toString().split(Lop.DATATYPE_PREFIX);				
				Object obj = UtilFunctions.stringToObject(schema.get(_rcCols[j]), tmp[0]);
				map.put(Long.parseLong(tmp[1]), obj);				
			}
			_rcMaps[j] = map;
		}
	}

	@Override
	public void decode(double[] in, Object[] out) {
		for( int j=0; j<_rcCols.length; j++ ) {
			long key = UtilFunctions.toLong(in[_rcCols[j]]);
			out[_rcCols[j]] = _rcMaps[j].get(key);
		}
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		out.ensureAllocatedColumns(in.getNumRows());
		for( int i=0; i<in.getNumRows(); i++ ) {
			for( int j=0; j<_rcCols.length; j++ ) {
				double val = in.quickGetValue(i, _rcCols[j]);
				long key = UtilFunctions.toLong(val);
				out.set(i, _rcCols[j], _rcMaps[j].get(key));
			}
		}
		return out;
	}
}
