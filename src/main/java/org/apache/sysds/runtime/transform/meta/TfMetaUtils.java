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

package org.apache.sysds.runtime.transform.meta;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.decode.DecoderRecode;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class TfMetaUtils 
{
	public static boolean isIDSpec(String spec) {
		try {
			JSONObject jSpec = new JSONObject(spec);
			return isIDSpec(jSpec);
		}
		catch(JSONException ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	public static boolean isIDSpec(JSONObject spec) throws JSONException {
		return spec.containsKey("ids") && spec.getBoolean("ids");
	}

	public static boolean containsOmitSpec(String spec, String[] colnames) {
		return (TfMetaUtils.parseJsonIDList(spec, colnames, TfMethod.OMIT.toString()).length > 0);	
	}

	public static int[] parseJsonIDList(String spec, String[] colnames, String group) {
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
	 * @param spec transform specification as json string
	 * @param colnames column names
	 * @param group attribute name in json class
	 * @return list of column ids
	 * @throws JSONException if JSONException occurs
	 */
	public static int[] parseJsonIDList(JSONObject spec, String[] colnames, String group)
			throws JSONException
	{
		return parseJsonIDList(spec, colnames, group, -1, -1);
	}
	
	/**
	 * @param spec transform specification as json string
	 * @param colnames column names
	 * @param group attribute name in json class
	 * @param minCol start of columns to ignore (1-based, inclusive, if -1 not used)
	 * @param maxCol end of columns to ignore (1-based, exclusive, if -1 not used)
	 * @return list of column ids
	 * @throws JSONException if JSONException occurs
	 */
	public static int[] parseJsonIDList(JSONObject spec, String[] colnames, String group, int minCol, int maxCol)
		throws JSONException
	{
		int[] arr = new int[0];
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
			arr = parseJsonPlainArrayIDList(attrs, colnames, minCol, maxCol, ids);
		}
		return arr;
	}

	public static int parseJsonObjectID(JSONObject colspec, String[] colnames, int minCol, int maxCol, boolean ids) throws JSONException {
		int ix;
		if(ids) {
			ix = colspec.getInt("id");
			if(maxCol != -1 && ix >= maxCol)
				ix = -1;
			if(minCol != -1 && ix >= 0)
				ix -= minCol - 1;
		}
		else {
			ix = ArrayUtils.indexOf(colnames, colspec.get("name")) + 1;
		}
		if(ix > 0)
			return ix;
		else if(minCol == -1 && maxCol == -1)
			throw new RuntimeException("Specified column '" + colspec.get(ids ? "id" : "name") + "' does not exist.");
		return -1;
	}

	public static int[] parseJsonObjectIDList(JSONObject spec, String[] colnames, String group, int minCol, int maxCol)
		throws JSONException {
		int[] arr = new int[0];
		boolean ids = spec.containsKey("ids") && spec.getBoolean("ids");

		if(spec.containsKey(group) && spec.get(group) instanceof JSONArray) {
			JSONArray colspecs = (JSONArray) spec.get(group);
			arr = parseJsonArrayIDList(colspecs, colnames, minCol, maxCol, ids);
		}

		return arr;
	}
	
	public static int[] parseJsonArrayIDList(JSONArray arr, String[] colnames, int minCol, int maxCol, boolean ids)
		throws JSONException
	{
		List<Integer> colList = new ArrayList<>();
		for(Object o : arr) {
			JSONObject colspec = (JSONObject) o;
			int id = parseJsonObjectID(colspec, colnames, minCol, maxCol, ids);
			if(id > 0)
				colList.add(id);
		}

		// ensure ascending order of column IDs
		return colList.stream().mapToInt((i) -> i).sorted().toArray();
	}
	
	public static int[] parseJsonPlainArrayIDList(JSONArray arr, String[] colnames, int minCol, int maxCol, boolean ids) {
		List<Integer> colList = new ArrayList<>();
		
		//construct ID list array
		for(int i=0; i < arr.length(); i++) {
			int ix;
			if (ids) {
				ix = UtilFunctions.toInt(arr.get(i));
				if(maxCol != -1 && ix >= maxCol)
					ix = -1;
				if(minCol != -1 && ix >= 0)
					ix -= minCol - 1;
			}
			else {
				ix = ArrayUtils.indexOf(colnames, arr.get(i)) + 1;
			}
			if(ix > 0)
				colList.add(ix);
			else if(minCol == -1 && maxCol == -1)
				// only if we remove some columns, ix -1 is expected
				throw new RuntimeException("Specified column '" + arr.get(i) + "' does not exist.");
		}
		
		//ensure ascending order of column IDs
		return colList.stream().mapToInt((i) -> i)
			.sorted().toArray();
	}

	/**
	 * Get K value used for calculation during feature hashing from parsed specifications.
	 * @param parsedSpec parsed specifications
	 * @return K value
	 * @throws JSONException if JSONException occurs
	 */
	public static long getK(JSONObject parsedSpec) throws JSONException {
		return parsedSpec.getLong("K");
	}


	/**
	 * Reads transform meta data from an HDFS file path and converts it into an in-memory
	 * FrameBlock object.
	 * 
	 * @param spec      transform specification as json string
	 * @param metapath  hdfs file path to meta data directory
	 * @param colDelim  separator for processing column names in the meta data file 'column.names'
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	public static FrameBlock readTransformMetaDataFromFile(String spec, String metapath, String colDelim) 
		throws IOException 
	{
		//read column names
		String colnamesStr = HDFSTool.readStringFromHDFSFile(metapath+File.separator+TfUtils.TXMTD_COLNAMES);
		String[] colnames = IOUtilFunctions.split(colnamesStr.trim(), colDelim);
		
		//read meta data (currently supported: recode, dummycode, bin, omit, impute)
		//note: recode/binning and impute might be applied on the same column
		HashMap<String,String> meta = new HashMap<>();
		HashMap<String,String> mvmeta = new HashMap<>();
		int rows = 0;
		for( int j=0; j<colnames.length; j++ ) {
			String colName = colnames[j];
			//read recode maps for recoded or dummycoded columns
			String name = metapath+File.separator+"Recode"+File.separator+colName;
			if( HDFSTool.existsFileOnHDFS(name+TfUtils.TXMTD_RCD_MAP_SUFFIX) ) {
				meta.put(colName, HDFSTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_MAP_SUFFIX));
				String ndistinct = HDFSTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			//read binning map for binned columns
			String name2 = metapath+File.separator+"Bin"+File.separator+colName;
			if( HDFSTool.existsFileOnHDFS(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX) ) {
				String binmap = HDFSTool.readStringFromHDFSFile(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX);
				meta.put(colName, binmap);
				rows = Math.max(rows, Integer.parseInt(binmap.split(TfUtils.TXMTD_SEP)[4]));
			}
			//read impute value for mv columns
			String name3 = metapath+File.separator+"Impute"+File.separator+colName;
			if( HDFSTool.existsFileOnHDFS(name3+TfUtils.TXMTD_MV_FILE_SUFFIX) ) {
				String mvmap = HDFSTool.readStringFromHDFSFile(name3+TfUtils.TXMTD_MV_FILE_SUFFIX);
				mvmeta.put(colName, mvmap);
			}
		}

		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, colnames);
		List<Integer> binIDs = parseBinningColIDs(spec, colnames, -1, -1);
		
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
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	public static FrameBlock readTransformMetaDataFromPath(String spec, String metapath, String colDelim) 
		throws IOException 
	{
		//read column names
		String colnamesStr = getStringFromResource(metapath+"/"+TfUtils.TXMTD_COLNAMES);
		String[] colnames = IOUtilFunctions.split(colnamesStr.trim(), colDelim);
		
		//read meta data (currently supported: recode, dummycode, bin, omit)
		//note: recode/binning and impute might be applied on the same column
		HashMap<String,String> meta = new HashMap<>();
		HashMap<String,String> mvmeta = new HashMap<>();
		int rows = 0;
		for( int j=0; j<colnames.length; j++ ) {
			String colName = colnames[j];
			//read recode maps for recoded or dummycoded columns
			String name = metapath+"/"+"Recode"+"/"+colName;
			String map = getStringFromResource(name+TfUtils.TXMTD_RCD_MAP_SUFFIX);
			if( map != null ) {
				meta.put(colName, map);
				String ndistinct = getStringFromResource(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			//read binning map for binned columns
			String name2 = metapath+"/"+"Bin"+"/"+colName;
			String map2 = getStringFromResource(name2+TfUtils.TXMTD_BIN_FILE_SUFFIX);
			if( map2 != null ) {
				meta.put(colName, map2);
				rows = Math.max(rows, Integer.parseInt(map2.split(TfUtils.TXMTD_SEP)[4]));
			}
			//read impute value for mv columns
			String name3 = metapath+File.separator+"Impute"+File.separator+colName;
			String map3 = getStringFromResource(name3+TfUtils.TXMTD_MV_FILE_SUFFIX);
			if( map3 != null ) {
				mvmeta.put(colName, map3);
			}
		}
		
		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, colnames);
		List<Integer> binIDs = parseBinningColIDs(spec, colnames, -1, -1);
		
		//create frame block from in-memory strings
		return convertToTransformMetaDataFrame(rows, colnames, recodeIDs, binIDs, meta, mvmeta);
	}
	
	/**
	 * Converts transform meta data into an in-memory FrameBlock object.
	 * 
	 * @param rows number of rows
	 * @param colnames column names
	 * @param rcIDs recode IDs
	 * @param binIDs binning IDs
	 * @param meta ?
	 * @param mvmeta ?
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	private static FrameBlock convertToTransformMetaDataFrame(int rows, String[] colnames, List<Integer> rcIDs, List<Integer> binIDs, 
			HashMap<String,String> meta, HashMap<String,String> mvmeta) 
		throws IOException 
	{
		//create frame block w/ pure string schema
		ValueType[] schema = UtilFunctions.nCopies(colnames.length, ValueType.STRING);
		FrameBlock ret = new FrameBlock(schema, colnames);
		ret.ensureAllocatedColumns(rows);
		
		//encode recode maps (recoding/dummycoding) into frame
		for( Integer colID : rcIDs ) {
			String name = colnames[colID-1];
			String map = meta.get(name);
			if( map == null )
				throw new IOException("Recode map for column '"+name+"' (id="+colID+") not existing.");
			
			InputStream is = new ByteArrayInputStream(map.getBytes("UTF-8"));
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			Pair<String,String> pair = new Pair<>();
			String line; int rpos = 0; 
			while( (line = br.readLine()) != null ) {
				DecoderRecode.parseRecodeMapEntry(line, pair);
				String tmp = pair.getKey() + Lop.DATATYPE_PREFIX + pair.getValue();
				ret.set(rpos++, colID-1, tmp);
			}
			ret.getColumnMetadata(colID-1).setNumDistinct(rpos);
		}
		
		//encode bin maps (binning) into frame
		for( Integer colID : binIDs ) {
			String name = colnames[colID-1];
			String map = meta.get(name);
			if( map == null )
				throw new IOException("Binning map for column '"+name+"' (id="+colID+") not existing.");
			String[] fields = map.split(TfUtils.TXMTD_SEP);
			double min = Double.parseDouble(fields[1]);
			double binwidth = Double.parseDouble(fields[3]);
			int nbins = UtilFunctions.parseToInt(fields[4]);
			//materialize bins to support equi-width/equi-height
			for( int i=0; i<nbins; i++ ) { 
				String lbound = String.valueOf(min+i*binwidth);
				String ubound = String.valueOf(min+(i+1)*binwidth);
				ret.set(i, colID-1, lbound+Lop.DATATYPE_PREFIX+ubound);
			}
			ret.getColumnMetadata(colID-1).setNumDistinct(nbins);
		}
		
		//encode impute meta data into frame
		for( Entry<String, String> e : mvmeta.entrySet() ) {
			int colID = ArrayUtils.indexOf(colnames, e.getKey()) + 1;
			String mvVal = e.getValue().split(TfUtils.TXMTD_SEP)[1];
			ret.getColumnMetadata(colID-1).setMvValue(mvVal);
		}
		
		return ret;
	}
	
	/**
	 * Parses the given json specification and extracts a list of column ids
	 * that are subject to recoding.
	 * 
	 * @param spec transform specification as json string
	 * @param colnames column names
	 * @return list of column ids
	 * @throws IOException if IOException occurs
	 */
	private static List<Integer> parseRecodeColIDs(String spec, String[] colnames) 
		throws IOException 
	{	
		if( spec == null )
			throw new IOException("Missing transform specification.");
		
		List<Integer> specRecodeIDs = null;
		
		try {
			//parse json transform specification for recode col ids
			JSONObject jSpec = new JSONObject(spec);
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RECODE.toString())));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.DUMMYCODE.toString()))); 
			specRecodeIDs = CollectionUtils.unionDistinct(rcIDs, dcIDs);
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		
		return specRecodeIDs;
	}

	public static List<Integer> parseBinningColIDs(String spec, String[] colnames, int minCol, int maxCol)
		throws IOException 
	{
		try {
			JSONObject jSpec = new JSONObject(spec);
			return parseBinningColIDs(jSpec, colnames, minCol, maxCol);
		}
		catch(JSONException ex) {
			throw new IOException(ex);
		}
	}

	public static List<Integer> parseBinningColIDs(JSONObject jSpec, String[] colnames, int minCol, int maxCol)
		throws IOException 
	{
		try {
			String binKey = TfMethod.BIN.toString();
			if( jSpec.containsKey(binKey) && jSpec.get(binKey) instanceof JSONArray ) {
				return Arrays.asList(ArrayUtils.toObject(
					parseJsonObjectIDList(jSpec, colnames, binKey, minCol, maxCol)));
			}
			else { //internally generated
				return Arrays.asList(ArrayUtils.toObject(
					parseJsonIDList(jSpec, colnames, binKey)));
			}
		}
		catch(JSONException ex) {
			throw new IOException(ex);
		}
	}
	
	public static List<Integer> parseUDFColIDs(JSONObject jSpec, String[] colnames, int minCol, int maxCol)
		throws IOException 
	{
		try {
			String binKey = TfMethod.UDF.toString();
			if( jSpec.containsKey(binKey) ) {
				JSONArray bin = jSpec.getJSONObject(binKey).getJSONArray("ids");
				return Arrays.asList(ArrayUtils.toObject(
					parseJsonPlainArrayIDList(bin, colnames, minCol, maxCol, true)));
			}
		}
		catch(JSONException ex) {
			throw new IOException(ex);
		}
		return new ArrayList<>();
	}
	
	private static String getStringFromResource(String path) throws IOException {
		try(InputStream is = Connection.class.getResourceAsStream(path) ) {
			return IOUtilFunctions.toString(is);
		}
	}

	@SuppressWarnings("unchecked")
	public static void checkValidEncoders(JSONObject jSpec) {
		Set<String> validEncoders = new HashSet<>();
		validEncoders.addAll(Arrays.asList("ids","K"));
		for( TfMethod tf : TfMethod.values() )
			validEncoders.add(tf.toString());
		Iterator<String> keys = jSpec.keys();
		while( keys.hasNext() ) {
			String key = keys.next();
			if( !validEncoders.contains(key) )
				throw new DMLRuntimeException("Transform specification includes an invalid encoder: "+key);
		}
	}
}
