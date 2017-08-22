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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class EncoderRecode extends Encoder 
{	
	private static final long serialVersionUID = 8213163881283341874L;

	//recode maps and custom map for partial recode maps 
	private HashMap<Integer, HashMap<String, Long>> _rcdMaps  = new HashMap<Integer, HashMap<String, Long>>();
	private HashMap<Integer, HashSet<Object>> _rcdMapsPart = null;
	
	public EncoderRecode(JSONObject parsedSpec, String[] colnames, int clen)
		throws JSONException 
	{
		super(null, clen);
		
		if( parsedSpec.containsKey(TfUtils.TXMETHOD_RECODE) ) {
			_colList = TfMetaUtils.parseJsonIDList(parsedSpec, colnames, TfUtils.TXMETHOD_RECODE);
		}
	}
	
	public HashMap<Integer, HashMap<String,Long>> getCPRecodeMaps() { 
		return _rcdMaps; 
	}
	
	public HashMap<Integer, HashSet<Object>> getCPRecodeMapsPartial() { 
		return _rcdMapsPart; 
	}
	
	private long lookupRCDMap(int colID, String key) {
		if( !_rcdMaps.containsKey(colID) )
			return -1; //empty recode map
		Long tmp = _rcdMaps.get(colID).get(key);
		return (tmp!=null) ? tmp : -1;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		if( !isApplicable() )
			return out;
		
		//build and apply recode maps 
		build(in);
		apply(in, out);
		
		return out;
	}

	@Override
	public void build(FrameBlock in) {
		if( !isApplicable() )
			return;		

		Iterator<String[]> iter = in.getStringRowIterator(_colList);
		while( iter.hasNext() ) {
			String[] row = iter.next(); 
			for( int j=0; j<_colList.length; j++ ) {
				int colID = _colList[j]; //1-based
				//allocate column map if necessary
				if( !_rcdMaps.containsKey(colID) ) 
					_rcdMaps.put(colID, new HashMap<String,Long>());
				//probe and build column map
				HashMap<String,Long> map = _rcdMaps.get(colID);
				String key = row[j];
				if( key!=null && !key.isEmpty() && !map.containsKey(key) )
					map.put(key, Long.valueOf(map.size()+1));
			}
		}
	}

	public void buildPartial(FrameBlock in) {
		if( !isApplicable() )
			return;		

		//ensure allocated partial recode map
		if( _rcdMapsPart == null )
			_rcdMapsPart = new HashMap<Integer, HashSet<Object>>();
		
		//construct partial recode map (tokens w/o codes)
		//iterate over columns for sequential access
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			//allocate column map if necessary
			if( !_rcdMapsPart.containsKey(colID) ) 
				_rcdMapsPart.put(colID, new HashSet<Object>());
			HashSet<Object> map = _rcdMapsPart.get(colID);
			//probe and build column map
			for( int i=0; i<in.getNumRows(); i++ )
				map.add(in.get(i, colID-1));
			//cleanup unnecessary entries once
			map.remove(null);
			map.remove("");
		}
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		//apply recode maps column wise
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			for( int i=0; i<in.getNumRows(); i++ ) {
				Object okey = in.get(i, colID-1);
				String key = (okey!=null) ? okey.toString() : null;
				long code = lookupRCDMap(colID, key);
				out.quickSetValue(i, colID-1,
					(code >= 0) ? code : Double.NaN);
			}
		}
		
		return out;
	}

	@Override
	public FrameBlock getMetaData(FrameBlock meta) {
		if( !isApplicable() )
			return meta;
		
		//inverse operation to initRecodeMaps
		
		//allocate output rows
		int maxDistinct = 0;
		for( int j=0; j<_colList.length; j++ )
			if( _rcdMaps.containsKey(_colList[j]) )
				maxDistinct = Math.max(maxDistinct, _rcdMaps.get(_colList[j]).size());
		meta.ensureAllocatedColumns(maxDistinct);
		
		//create compact meta data representation
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			int rowID = 0;
			if( _rcdMaps.containsKey(_colList[j]) )
				for( Entry<String, Long> e : _rcdMaps.get(colID).entrySet() ) {
					String tmp = constructRecodeMapEntry(e.getKey(), e.getValue());
					meta.set(rowID++, colID-1, tmp); 
				}
			meta.getColumnMetadata(colID-1).setNumDistinct(
					_rcdMaps.get(colID).size());
		}
		
		return meta;
	}
	

	/**
	 * Construct the recodemaps from the given input frame for all 
	 * columns registered for recode.
	 * 
	 * @param meta frame block
	 */
	public void initMetaData( FrameBlock meta ) {
		if( meta == null || meta.getNumRows()<=0 )
			return;
		
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			_rcdMaps.put(colID, meta.getRecodeMap(colID-1));
		}
	}
	
	/**
	 * Returns the Recode map entry which consists of concatenation of code, delimiter and token. 
	 * 
	 * @param token	is part of Recode map
	 * @param code  is code for token
	 * @return the concatenation of token and code with delimiter in between
	 */
	public static String constructRecodeMapEntry(String token, Long code) {
		return token + Lop.DATATYPE_PREFIX + code.toString();
	}
	
	/**
	 * Splits a Recode map entry into its token and code.
	 * 
	 * @param value concatenation of token and code with delimiter in between
	 * @return string array of token and code
	 */
	public static String[] splitRecodeMapEntry(String value) {
		// Instead of using splitCSV which is forcing string with RFC-4180 format, 
		// using Lop.DATATYPE_PREFIX separator to split token and code 
		int pos = value.toString().lastIndexOf(Lop.DATATYPE_PREFIX);
		return new String[] {value.substring(0, pos), value.substring(pos+1)};
	}
}
