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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.TfUtils;
import org.apache.sysml.runtime.transform.decode.DecoderRecode;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

public class TfMetaUtils 
{
	private static final Log LOG = LogFactory.getLog(TfMetaUtils.class.getName());

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
		//NOTE: this implementation assumes column alignment of colnames and coltypes
		
		//read column types (for sanity check column names)
		String coltypesStr = MapReduceTool.readStringFromHDFSFile(metapath+File.separator+TfUtils.TXMTD_COLTYPES);
		List<String> coltypes = Arrays.asList(IOUtilFunctions.split(coltypesStr.trim(), TfUtils.TXMTD_SEP));
		
		//read column names
		String colnamesStr = MapReduceTool.readStringFromHDFSFile(metapath+File.separator+TfUtils.TXMTD_COLNAMES);
		List<String> colnames = Arrays.asList(IOUtilFunctions.split(colnamesStr.trim(), colDelim));
		if( coltypes.size() != colnames.size() ) {
			LOG.warn("Number of columns names: "+colnames.size()+" (expected: "+coltypes.size()+").");
			LOG.warn("--Sample column names: "+(!colnames.isEmpty()?colnames.get(0):"null"));
		}
		
		//read meta data (currently only recode supported, without parsing spec)
		HashMap<String,String> meta = new HashMap<String,String>();
		int rows = 0;
		for( int j=0; j<colnames.size(); j++ ) {
			String colName = colnames.get(j);
			String name = metapath+File.separator+"Recode"+File.separator+colName;
			if( MapReduceTool.existsFileOnHDFS(name+TfUtils.TXMTD_RCD_MAP_SUFFIX) ) {
				meta.put(colName, MapReduceTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_MAP_SUFFIX));
				String ndistinct = MapReduceTool.readStringFromHDFSFile(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			else if( coltypes.get(j).equals("2") ) {
				LOG.warn("Recode map for column '"+colName+"' does not exist.");
			}
		}

		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, coltypes);
		
		//create frame block from in-memory strings
		return convertToTransformMetaDataFrame(rows, recodeIDs, colnames, meta);
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
		//NOTE: this implementation assumes column alignment of colnames and coltypes
		
		//read column types (for sanity check column names)
		String coltypesStr = IOUtilFunctions.toString(Connection.class.getResourceAsStream(metapath+"/"+TfUtils.TXMTD_COLTYPES));
		List<String> coltypes = Arrays.asList(IOUtilFunctions.split(coltypesStr.trim(), TfUtils.TXMTD_SEP));
		
		//read column names
		String colnamesStr = IOUtilFunctions.toString(Connection.class.getResourceAsStream(metapath+"/"+TfUtils.TXMTD_COLNAMES));
		List<String> colnames = Arrays.asList(IOUtilFunctions.split(colnamesStr.trim(), colDelim));
		if( coltypes.size() != colnames.size() ) {
			LOG.warn("Number of columns names: "+colnames.size()+" (expected: "+coltypes.size()+").");
			LOG.warn("--Sample column names: "+(!colnames.isEmpty()?colnames.get(0):"null"));
		}
		
		//read meta data (currently only recode supported, without parsing spec)
		HashMap<String,String> meta = new HashMap<String,String>();
		int rows = 0;
		for( int j=0; j<colnames.size(); j++ ) {
			String colName = colnames.get(j);
			String name = metapath+"/"+"Recode"+"/"+colName;
			String map = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name+TfUtils.TXMTD_RCD_MAP_SUFFIX));
			if( map != null ) {
				meta.put(colName, map);
				String ndistinct = IOUtilFunctions.toString(Connection.class.getResourceAsStream(name+TfUtils.TXMTD_RCD_DISTINCT_SUFFIX));
				rows = Math.max(rows, Integer.parseInt(ndistinct));
			}
			else if( coltypes.get(j).equals("2") ) {
				LOG.warn("Recode map for column '"+colName+"' does not exist.");
			}
		}
		
		//get list of recode ids
		List<Integer> recodeIDs = parseRecodeColIDs(spec, coltypes);
		
		//create frame block from in-memory strings
		return convertToTransformMetaDataFrame(rows, recodeIDs, colnames, meta);
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
	private static FrameBlock convertToTransformMetaDataFrame(int rows, List<Integer> recodeIDs, List<String> colnames, HashMap<String,String> meta) 
		throws IOException 
	{
		//create frame block w/ pure string schema
		List<ValueType> schema = Collections.nCopies(colnames.size(), ValueType.STRING);
		FrameBlock ret = new FrameBlock(schema, colnames);
		ret.ensureAllocatedColumns(rows);
		
		//encode recode maps into frame
		for( Integer colID : recodeIDs ) {
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
	private static ArrayList<Integer> parseRecodeColIDs(String spec, List<String> coltypes) 
		throws IOException 
	{	
		ArrayList<Integer> specRecodeIDs = new ArrayList<Integer>();
		
		try {
			if( spec != null ) {
				//parse json transform specification for recode col ids
				JSONObject jSpec = new JSONObject(spec);
				if ( jSpec.containsKey(TfUtils.TXMETHOD_RECODE))  {
					JSONArray attrs = null; //TODO simplify once json spec consolidated
					if( jSpec.get(TfUtils.TXMETHOD_RECODE) instanceof JSONObject ) {
						JSONObject obj = (JSONObject) jSpec.get(TfUtils.TXMETHOD_RECODE);
						attrs = (JSONArray) obj.get(TfUtils.JSON_ATTRS);
					}
					else
						attrs = (JSONArray)jSpec.get(TfUtils.TXMETHOD_RECODE);				
					for(int j=0; j<attrs.length(); j++) 
						specRecodeIDs.add(UtilFunctions.toInt(attrs.get(j)));
				}
			}
			else {
				//obtain recode col ids from coltypes 
				for( int j=0; j<coltypes.size(); j++ )
					if( coltypes.get(j).equals("2") )
						specRecodeIDs.add(j+1);
			}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		
		return specRecodeIDs;
	}
}
