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

package org.apache.sysml.runtime.transform.meta;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.decode.DecoderRecode;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class TfMetaUtils 
{
	/**
	 * 
	 * @param spec
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static boolean isIDSpecification(String spec) throws DMLRuntimeException {
		try {
			JSONObject jSpec = new JSONObject(spec);
			return jSpec.containsKey("ids") && jSpec.getBoolean("ids");
		}
		catch(JSONException ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	/**
	 * 
	 * @param spec
	 * @param colnames
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static boolean containsOmitSpec(String spec, List<String> colnames) throws DMLRuntimeException {
		return (TfMetaUtils.parseJsonIDList(spec, colnames, TfUtils.TXMETHOD_OMIT).length > 0);	
	}
	
	/**
	 * 
	 * @param spec
	 * @param colnames
	 * @param group
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int[] parseJsonIDList(String spec, List<String> colnames, String group) 
		throws DMLRuntimeException 
	{
		try {
			JSONObject jSpec = new JSONObject(spec);
			return parseJsonIDList(jSpec, colnames, group);
		}
		catch(JSONException ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	/**
	 * TODO consolidate external and internal json spec definitions
	 * 
	 * @param spec
	 * @param colnames
	 * @param group
	 * @return
	 * @throws JSONException
	 */
	public static int[] parseJsonIDList(JSONObject spec, List<String> colnames, String group) 
		throws JSONException
	{
		int[] colList = new int[0];
		boolean ids = spec.containsKey("ids") && spec.getBoolean("ids");
		
		if( spec.containsKey(group) ) {
			//parse attribute-array or plain array of IDs
			JSONArray attrs = null;
			if( spec.get(group) instanceof JSONObject ) {
				attrs = (JSONArray) ((JSONObject)spec.get(group)).get(TfUtils.JSON_ATTRS);
				ids = true; //file-based transform outputs ids w/o id tags
			}
			else
				attrs = (JSONArray)spec.get(group);			
			
			//construct ID list array
			colList = new int[attrs.size()];
			for(int i=0; i < colList.length; i++) {
				colList[i] = ids ? UtilFunctions.toInt(attrs.get(i)) : 
					(colnames.indexOf(attrs.get(i)) + 1);
				if( colList[i] <= 0 ) {
					throw new RuntimeException("Specified column '" + 
						attrs.get(i)+"' does not exist.");
				}
			}
		
			//ensure ascending order of column IDs
			Arrays.sort(colList);
		}
		
		return colList;
	}
	
	/**
	 * 
	 * @param spec
	 * @param group
	 * @return
	 * @throws JSONException
	 */
	public static int[] parseJsonObjectIDList(JSONObject spec, List<String> colnames, String group) 
		throws JSONException
	{
		int[] colList = new int[0];
		boolean ids = spec.containsKey("ids") && spec.getBoolean("ids");
		
		if( spec.containsKey(group) && spec.get(group) instanceof JSONArray )
		{
			JSONArray colspecs = (JSONArray)spec.get(group);
			colList = new int[colspecs.size()];
			for(int j=0; j<colspecs.size(); j++) {
				JSONObject colspec = (JSONObject) colspecs.get(j);
				colList[j] = ids ? colspec.getInt("id") : 
					(colnames.indexOf(colspec.get("name")) + 1);
				if( colList[j] <= 0 ) {
					throw new RuntimeException("Specified column '" +
						colspec.get(ids?"id":"name")+"' does not exist.");
				}
			}
			
			//ensure ascending order of column IDs
			Arrays.sort(colList);
		}
		
		return colList;
	}
	
	/**
	 * Reads transform meta data from an HDFS file path and converts it into an in-memory
	 * FrameBlock object.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  hdfs file path to meta data directory
	 * @param colDelim  separator for processing column names in the meta data file 'column.names'
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException
	 */
	public static FrameBlock readTransformMetaDataFromFile(String spec, String metapath, String colDelim) 
		throws IOException 
	{
		//read column names
		String colnamesStr = MapReduceTool.readStringFromHDFSFile(metapath+File.separator+TfUtils.TXMTD_COLNAMES);
		List<String> colnames = Arrays.asList(IOUtilFunctions.split(colnamesStr.trim(), colDelim));
		
		//read meta data (currently supported: recode, dummycode, bin, omit, impute)
		//note: recode/binning and impute might be applied on the same column
		HashMap<String,String> meta = new HashMap<String,String>();
		HashMap<String,String> mvmeta = new HashMap<String,String>();
		int rows = 0;
		for( int j=0; j<colnames.size(); j++ ) {
			String colName = colnames.get(j);
			//read recode maps for recoded or dummycoded columns
			String name = metapath+File.separator+"Recode"+File.separator+colName;
			if( MapReduceTool.existsFileOnHDFS(name+TfUtils.TXMTD_RCD_MAP_SUFFIX) ) {
				meta.put(colName, MapReduceTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_MAP_SUFFIX));
				String ndistinct = MapReduceTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			//read binning map for binned columns
			String name2 = metapath+File.separator+"Bin"+File.separator+colName;
			if( MapReduceTool.existsFileOnHDFS(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX) ) {
				String binmap = MapReduceTool.readStringFromHDFSFile(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX);
				meta.put(colName, binmap);
				rows = Math.max(rows, Integer.parseInt(binmap.split(TfUtils.TXMTD_SEP)[4]));
			}
			//read impute value for mv columns
			String name3 = metapath+File.separator+"Impute"+File.separator+colName;
			if( MapReduceTool.existsFileOnHDFS(name3+TfUtils.TXMTD_MV_FILE_SUFFIX) ) {
				String mvmap = MapReduceTool.readStringFromHDFSFile(name3+TfUtils.TXMTD_MV_FILE_SUFFIX);
				mvmeta.put(colName, mvmap);
			}
		}

		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, colnames);
		List<Integer> binIDs = parseBinningColIDs(spec, colnames);
		
		//create frame block from in-memory strings
		return convertToTransformMetaDataFrame(rows, colnames, recodeIDs, binIDs, meta, mvmeta);
	}

	/**
	 * Reads transform meta data from the class path and converts it into an in-memory
	 * FrameBlock object.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  resource path to meta data directory
	 * @param colDelim  separator for processing column names in the meta data file 'column.names'
	 * @return FrameBlock object representing transform metadata
	 * @throws IOException
	 */
	public static FrameBlock readTransformMetaDataFromPath(String spec, String metapath, String colDelim) 
		throws IOException 
	{
		//read column names
		String colnamesStr = IOUtilFunctions.toString(Connection.class.getResourceAsStream(metapath+"/"+TfUtils.TXMTD_COLNAMES));
		List<String> colnames = Arrays.asList(IOUtilFunctions.split(colnamesStr.trim(), colDelim));
		
		//read meta data (currently supported: recode, dummycode, bin, omit)
		//note: recode/binning and impute might be applied on the same column
		HashMap<String,String> meta = new HashMap<String,String>();
		HashMap<String,String> mvmeta = new HashMap<String,String>();
		int rows = 0;
		for( int j=0; j<colnames.size(); j++ ) {
			String colName = colnames.get(j);
			//read recode maps for recoded or dummycoded columns
			String name = metapath+"/"+"Recode"+"/"+colName;
			String map = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name+TfUtils.TXMTD_RCD_MAP_SUFFIX));
			if( map != null ) {
				meta.put(colName, map);
				String ndistinct = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX));
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			//read binning map for binned columns
			String name2 = metapath+"/"+"Bin"+"/"+colName;
			String map2 = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX));
			if( map2 != null ) {
				meta.put(colName, map2);
				rows = Math.max(rows, Integer.parseInt(map2.split(TfUtils.TXMTD_SEP)[4]));
			}
			//read impute value for mv columns
			String name3 = metapath+File.separator+"Impute"+File.separator+colName;
			String map3 = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name3+TfUtils.TXMTD_MV_FILE_SUFFIX));
			if( map3 != null ) {
				mvmeta.put(colName, map3);
			}
		}
		
		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, colnames);
		List<Integer> binIDs = parseBinningColIDs(spec, colnames);
		
		//create frame block from in-memory strings
		return convertToTransformMetaDataFrame(rows, colnames, recodeIDs, binIDs, meta, mvmeta);
	}
	
	/**
	 * Converts transform meta data into an in-memory FrameBlock object.
	 * 
	 * @param rows
	 * @param recodeIDs
	 * @param colnames
	 * @param meta
	 * @return
	 * @throws IOException
	 */
	private static FrameBlock convertToTransformMetaDataFrame(int rows, List<String> colnames, List<Integer> rcIDs, List<Integer> binIDs, 
			HashMap<String,String> meta, HashMap<String,String> mvmeta) 
		throws IOException 
	{
		//create frame block w/ pure string schema
		List<ValueType> schema = Collections.nCopies(colnames.size(), ValueType.STRING);
		FrameBlock ret = new FrameBlock(schema, colnames);
		ret.ensureAllocatedColumns(rows);
		
		//encode recode maps (recoding/dummycoding) into frame
		for( Integer colID : rcIDs ) {
			String name = colnames.get(colID-1);
			String map = meta.get(name);
			if( map == null )
				throw new IOException("Recode map for column '"+name+"' (id="+colID+") not existing.");
			
			InputStream is = new ByteArrayInputStream(map.getBytes("UTF-8"));
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			Pair<String,String> pair = new Pair<String,String>();
			String line; int rpos = 0; 
			while( (line = br.readLine()) != null ) {
				DecoderRecode.parseRecodeMapEntry(line, pair);
				String tmp = pair.getKey() + Lop.DATATYPE_PREFIX + pair.getValue();
				ret.set(rpos++, colID-1, tmp);
			}
			ret.getColumnMetadata(colID-1).setNumDistinct((long)rpos);
		}
		
		//encode bin maps (binning) into frame
		for( Integer colID : binIDs ) {
			String name = colnames.get(colID-1);
			String map = meta.get(name);
			if( map == null )
				throw new IOException("Binning map for column '"+name+"' (id="+colID+") not existing.");
			String[] fields = map.split(TfUtils.TXMTD_SEP);
			double min = UtilFunctions.parseToDouble(fields[1]);
			double binwidth = UtilFunctions.parseToDouble(fields[3]);
			int nbins = UtilFunctions.parseToInt(fields[4]);
			//materialize bins to support equi-width/equi-height
			for( int i=0; i<nbins; i++ ) { 
				String lbound = String.valueOf(min+i*binwidth);
				String ubound = String.valueOf(min+(i+1)*binwidth);
				ret.set(i, colID-1, lbound+Lop.DATATYPE_PREFIX+ubound);
			}
			ret.getColumnMetadata(colID-1).setNumDistinct((long)nbins);
		}
		
		//encode impute meta data into frame
		for( Entry<String, String> e : mvmeta.entrySet() ) {
			int colID = colnames.indexOf(e.getKey()) + 1;
			String mvVal = e.getValue().split(TfUtils.TXMTD_SEP)[1];
			ret.getColumnMetadata(colID-1).setMvValue(mvVal);
		}
		
		return ret;
	}
	
	/**
	 * Parses the given json specification and extracts a list of column ids
	 * that are subject to recoding.
	 * 
	 * @param spec
	 * @param coltypes
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	private static List<Integer> parseRecodeColIDs(String spec, List<String> colnames) 
		throws IOException 
	{	
		if( spec == null )
			throw new IOException("Missing transform specification.");
		
		List<Integer> specRecodeIDs = null;
		
		try {
			//parse json transform specification for recode col ids
			JSONObject jSpec = new JSONObject(spec);
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_RECODE)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_DUMMYCODE))); 
			specRecodeIDs = new ArrayList<Integer>(CollectionUtils.union(rcIDs, dcIDs));
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		
		return specRecodeIDs;
	}
	
	/**
	 * 
	 * @param spec
	 * @return
	 * @throws IOException
	 */
	public static List<Integer> parseBinningColIDs(String spec, List<String> colnames) 
		throws IOException 
	{
		try {
			JSONObject jSpec = new JSONObject(spec);
			return parseBinningColIDs(jSpec, colnames);
		}
		catch(JSONException ex) {
			throw new IOException(ex);
		}
	}
	
	/**
	 * 
	 * @param jSpec
	 * @return
	 * @throws IOException
	 */
	public static List<Integer> parseBinningColIDs(JSONObject jSpec, List<String> colnames) 
		throws IOException 
	{
		try {
			if( jSpec.containsKey(TfUtils.TXMETHOD_BIN) && jSpec.get(TfUtils.TXMETHOD_BIN) instanceof JSONArray ) {
				return Arrays.asList(ArrayUtils.toObject(
						TfMetaUtils.parseJsonObjectIDList(jSpec, colnames, TfUtils.TXMETHOD_BIN)));	
			}
			else { //internally generates
				return Arrays.asList(ArrayUtils.toObject(
						TfMetaUtils.parseJsonIDList(jSpec, colnames, TfUtils.TXMETHOD_BIN)));	
			}
		}
		catch(JSONException ex) {
			throw new IOException(ex);
		}
	}
}
