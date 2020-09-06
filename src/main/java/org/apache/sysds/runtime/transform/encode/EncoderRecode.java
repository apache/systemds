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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.runtime.util.IndexRange;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;

public class EncoderRecode extends Encoder 
{
	private static final long serialVersionUID = 8213163881283341874L;
	
	//recode maps and custom map for partial recode maps 
	private HashMap<Integer, HashMap<String, Long>> _rcdMaps  = new HashMap<>();
	private HashMap<Integer, HashSet<Object>> _rcdMapsPart = null;
	
	public EncoderRecode(JSONObject parsedSpec, String[] colnames, int clen, int minCol, int maxCol)
		throws JSONException 
	{
		super(null, clen);
		_colList = TfMetaUtils.parseJsonIDList(parsedSpec, colnames, TfMethod.RECODE.toString(), minCol, maxCol);
	}
	
	private EncoderRecode(int[] colList, int clen) {
		super(colList, clen);
	}
	
	public EncoderRecode() {
		this(new int[0], 0);
	}
	
	private EncoderRecode(int[] colList, int clen, HashMap<Integer, HashMap<String, Long>> rcdMaps) {
		super(colList, clen);
		_rcdMaps = rcdMaps;
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
					putCode(map, key);
			}
		}
	}

	/**
	 * Put the code into the map with the provided key. The code depends on the type of encoder. 
	 * @param map column map
	 * @param key key for the new entry
	 */
	protected void putCode(HashMap<String,Long> map, String key) {
		map.put(key, Long.valueOf(map.size()+1));
	}
	
	public void prepareBuildPartial() {
		//ensure allocated partial recode map
		if( _rcdMapsPart == null )
			_rcdMapsPart = new HashMap<>();
	}

	public void buildPartial(FrameBlock in) {
		if( !isApplicable() )
			return;
		
		//construct partial recode map (tokens w/o codes)
		//iterate over columns for sequential access
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			//allocate column map if necessary
			if( !_rcdMapsPart.containsKey(colID) ) 
				_rcdMapsPart.put(colID, new HashSet<>());
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
	public Encoder subRangeEncoder(IndexRange ixRange) {
		List<Integer> cols = new ArrayList<>();
		HashMap<Integer, HashMap<String, Long>> rcdMaps = new HashMap<>();
		for(int col : _colList) {
			if(ixRange.inColRange(col)) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				int corrColumn = (int) (col - (ixRange.colStart - 1));
				cols.add(corrColumn);
				// copy rcdMap for column
				rcdMaps.put(corrColumn, new HashMap<>(_rcdMaps.get(col)));
			}
		}
		if(cols.isEmpty())
			// empty encoder -> sub range encoder does not exist
			return null;

		int[] colList = cols.stream().mapToInt(i -> i).toArray();
		return new EncoderRecode(colList, (int) ixRange.colSpan(), rcdMaps);
	}

	@Override
	public void mergeAt(Encoder other, int row, int col) {
		if(other instanceof EncoderRecode) {
			mergeColumnInfo(other, col);
			
			// merge together overlapping columns or add new columns
			EncoderRecode otherRec = (EncoderRecode) other;
			for (int otherColID : other._colList) {
				int colID = otherColID + col - 1;
				//allocate column map if necessary
				if( !_rcdMaps.containsKey(colID) )
					_rcdMaps.put(colID, new HashMap<>());
				
				HashMap<String, Long> otherMap = otherRec._rcdMaps.get(otherColID);
				if(otherMap != null) {
					// for each column, add all non present recode values
					for(Map.Entry<String, Long> entry : otherMap.entrySet()) {
						if (lookupRCDMap(colID, entry.getKey()) == -1) {
							// key does not yet exist
							putCode(_rcdMaps.get(colID), entry.getKey());
						}
					}
				}
			}
			return;
		}
		super.mergeAt(other, row, col);
	}
	
	public int[] numDistinctValues() {
		int[] numDistinct = new int[_colList.length];
		
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			numDistinct[j] = _rcdMaps.get(colID).size();
		}
		return numDistinct;
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
		StringBuilder sb = new StringBuilder(); //for reuse
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j]; //1-based
			int rowID = 0;
			if( _rcdMaps.containsKey(_colList[j]) )
				for( Entry<String, Long> e : _rcdMaps.get(colID).entrySet() ) {
					meta.set(rowID++, colID-1, 
						constructRecodeMapEntry(e.getKey(), e.getValue(), sb)); 
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
	@Override
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
		StringBuilder sb = new StringBuilder(token.length()+16);
		return constructRecodeMapEntry(token, code, sb);
	}
	
	private static String constructRecodeMapEntry(String token, Long code, StringBuilder sb) {
		sb.setLength(0); //reset reused string builder
		return sb.append(token).append(Lop.DATATYPE_PREFIX)
			.append(code.longValue()).toString();
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
