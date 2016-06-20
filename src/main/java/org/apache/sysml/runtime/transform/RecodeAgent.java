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

package org.apache.sysml.runtime.transform;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.MVImputeAgent.MVMethod;
import org.apache.sysml.runtime.transform.decode.DecoderRecode;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class RecodeAgent extends Encoder 
{	
	private static final long serialVersionUID = 8213163881283341874L;

	private int[] _mvrcdList = null;
	private int[] _fullrcdList = null;

	// HashMap< columnID, HashMap<distinctValue, count> >
	private HashMap<Integer, HashMap<String, Long>> _rcdMaps  = new HashMap<Integer, HashMap<String, Long>>();
	private HashMap<Integer, HashMap<String,String>> _finalMaps = null;
	
	public RecodeAgent(JSONObject parsedSpec, int clen)
		throws JSONException 
	{
		super(null, clen);
		int rcdCount = 0;
		
		if( parsedSpec.containsKey(TfUtils.TXMETHOD_RECODE) ) {
			int[] collist = TfMetaUtils.parseJsonIDList(parsedSpec, TfUtils.TXMETHOD_RECODE);
			rcdCount = initColList(collist);
		}
		
		if ( parsedSpec.containsKey(TfUtils.TXMETHOD_MVRCD)) {
			_mvrcdList = TfMetaUtils.parseJsonIDList(parsedSpec, TfUtils.TXMETHOD_MVRCD);
			rcdCount += _mvrcdList.length;
		}
		
		if ( rcdCount > 0 ) {
			_fullrcdList = new int[rcdCount];
			int idx = -1;
			if(_colList != null)
				for(int i=0; i < _colList.length; i++)
					_fullrcdList[++idx] = _colList[i]; 
			
			if(_mvrcdList != null)
				for(int i=0; i < _mvrcdList.length; i++)
					_fullrcdList[++idx] = _mvrcdList[i]; 
		}
	}
	
	public HashMap<Integer, HashMap<String,Long>> getCPRecodeMaps() { 
		return _rcdMaps; 
	}
	
	public HashMap<Integer, HashMap<String,String>> getRecodeMaps() {
		return _finalMaps;
	}
	
	void prepare(String[] words, TfUtils agents) {
		if ( _colList == null && _mvrcdList == null )
			return;
		
		String w = null;
		for (int colID : _fullrcdList) {
			w = UtilFunctions.unquote(words[colID-1].trim());
			if(_rcdMaps.get(colID) == null ) 
				_rcdMaps.put(colID, new HashMap<String, Long>());
			
			HashMap<String, Long> map = _rcdMaps.get(colID);
			Long count = map.get(w);
			if(count == null)
				map.put(w, new Long(1));
			else
				map.put(w, count+1);
		}
	}
	
	private HashMap<String, Long> handleMVConstant(int colID, TfUtils agents, HashMap<String, Long> map)
	{
		MVImputeAgent mvagent = agents.getMVImputeAgent();
		if ( mvagent.getMethod(colID) == MVMethod.CONSTANT ) 
		{
			// check if the "replacement" is part of the map. If not, add it.
			String repValue = mvagent.getReplacement(colID);
			if(repValue == null)
				throw new RuntimeException("Expecting a constant replacement value for column ID " + colID);
			
			repValue = UtilFunctions.unquote(repValue);
			Long count = map.get(repValue);
			long mvCount = agents.getValid() - mvagent.getNonMVCount(colID);
			if(count == null)
				map.put(repValue, mvCount);
			else
				map.put(repValue, count + mvCount);
		}
		return map;
	}
	
	/**
	 * Method to output transformation metadata from the mappers. 
	 * This information is collected and merged by the reducers.
	 * 
	 * @param out
	 * @throws IOException
	 */
	@Override
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TfUtils agents) throws IOException {
		mapOutputHelper(taskID, out, null, agents);
	}
	
	public ArrayList<Pair<Integer, DistinctValue>> mapOutputTransformationMetadata(int taskID, ArrayList<Pair<Integer, DistinctValue>> list, TfUtils agents) throws IOException {
		mapOutputHelper(taskID, null, list, agents);
		return list;
	}
	
	public void mapOutputHelper(int taskID, OutputCollector<IntWritable, DistinctValue> out, ArrayList<Pair<Integer, DistinctValue>> list, TfUtils agents) throws IOException {
		if ( _colList == null  && _mvrcdList == null )
			return;
		
		try 
		{ 
			for(int i=0; i < _fullrcdList.length; i++) 
			{
				int colID = _fullrcdList[i];
				HashMap<String, Long> map = _rcdMaps.get(colID);
				
				if(map != null) 
				{
					map = handleMVConstant(colID, agents,  map);
					
					if ( out != null ) {
						IntWritable iw = new IntWritable(colID);
						for(String s : map.keySet()) 
							out.collect(iw, new DistinctValue(s, map.get(s)));
					}
					else if ( list != null ) {
						for(String s : map.keySet()) 
							list.add(new Pair<Integer,DistinctValue>(colID, new DistinctValue(s, map.get(s))) );
					}
				}
			}
		} catch(Exception e) {
			throw new IOException(e);
		}
	}
	
	/**
	 * Function to output transformation metadata, including: 
	 * - recode maps, 
	 * - number of distinct values, 
	 * - mode, and 
	 * - imputation value (in the case of global_mode)
	 * 
	 * The column for which this function is invoked can be one of the following:
	 * - just recoded						(write .map, .ndistinct, .mode)
	 * - just mv imputed (w/ global_mode)	(write .impute)
	 * - both recoded and mv imputed		(write .map, .ndistinct, .mode, .impute)
	 * 
	 * @param map
	 * @param outputDir
	 * @param colID
	 * @param fs
	 * @param mvagent
	 * @throws IOException
	 */
	private void writeMetadata(HashMap<String,Long> map, String outputDir, int colID, FileSystem fs, TfUtils agents, boolean fromCP) throws IOException {
		// output recode maps and mode
		
		MVImputeAgent mvagent = agents.getMVImputeAgent();
		String mode = null;
		Long count = null;
		int rcdIndex = 0, modeIndex = 0;
		long maxCount = Long.MIN_VALUE;
		
		boolean isRecoded = (isApplicable(colID) != -1);
		boolean isModeImputed = (mvagent.getMethod(colID) == MVMethod.GLOBAL_MODE);
		
		Path pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + TfUtils.TXMTD_RCD_MAP_SUFFIX);
		BufferedWriter br=null;
		if(isRecoded)
			br = new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));		

		// remove NA strings
		if ( agents.getNAStrings() != null)
			for(String naword : agents.getNAStrings()) 
				map.remove(naword);
		
		if(fromCP)
			map = handleMVConstant(colID, agents,  map);
		
		if ( map.size() == 0 ) 
			throw new RuntimeException("Can not proceed since \"" + agents.getName(colID) + "\" (id=" + colID + ") contains only the missing values, and not a single valid value -- set imputation method to \"constant\".");
		
		// Order entries by category (string) value
		List<String> newNames = new ArrayList<String>(map.keySet());
		Collections.sort(newNames);

		for(String w : newNames) { //map.keySet()) {
				count = map.get(w);
				++rcdIndex;
				
				// output (w, count, rcdIndex)
				if(br != null)		
					br.write(UtilFunctions.quote(w) + TfUtils.TXMTD_SEP + rcdIndex + TfUtils.TXMTD_SEP + count  + "\n");
				
				if(maxCount < count) {
					maxCount = count;
					mode = w;
					modeIndex = rcdIndex;
				}
				
				// Replace count with recode index (useful when invoked from CP)
				map.put(w, (long)rcdIndex);
		}
		
		if(br != null)		
			br.close();
		
		if ( mode == null ) {
			mode = "";
			maxCount = 0;
		}
		
		if ( isRecoded ) 
		{
			// output mode
			pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + TfUtils.MODE_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(UtilFunctions.quote(mode) + "," + modeIndex + "," + maxCount );
			br.close();
		
			// output number of distinct values
			pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + TfUtils.TXMTD_RCD_DISTINCT_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(""+map.size());
			br.close();
		}
		
		if (isModeImputed) 
		{
			pt=new Path(outputDir+"/Impute/"+ agents.getName(colID) + TfUtils.TXMTD_MV_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(colID + "," + UtilFunctions.quote(mode));
			br.close();
		}
		
	}
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs, TfUtils agents) throws IOException {
		if(_colList == null && _mvrcdList == null )
			return;
		
		for(int i=0; i<_fullrcdList.length; i++) {
			int colID = _fullrcdList[i];
			writeMetadata(_rcdMaps.get(colID), outputDir, colID, fs, agents, true);
		}
	}
	
	/** 
	 * Method to merge map output transformation metadata.
	 * 
	 * @param values
	 * @return
	 * @throws IOException 
	 */
	@Override
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, FileSystem fs, TfUtils agents) throws IOException {
		HashMap<String, Long> map = new HashMap<String,Long>();
		
		DistinctValue d = new DistinctValue();
		String word = null;
		Long count = null, val = null;
		while(values.hasNext()) {
			d.reset();
			d = values.next();
			
			word = d.getWord();
			count = d.getCount();
			
			val = map.get(word);
			if(val == null) 
				map.put(word, count);
			else 
				map.put(word, val+count);
		}
		
		writeMetadata(map, outputDir, colID, fs, agents, false);
	}
	
	/**
	 * Method to load recode maps of all attributes, at once.
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents) throws IOException {
		if( !isApplicable() )
			return;
		
		_finalMaps = new HashMap<Integer, HashMap<String, String>>();
	
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_colList.length;i++) {
				int colID = _colList[i];
				
				Path path = new Path( txMtdDir + "/Recode/" + agents.getName(colID) + TfUtils.TXMTD_RCD_MAP_SUFFIX);
				TfUtils.checkValidInputFile(fs, path, true); 
				
				HashMap<String,String> map = new HashMap<String,String>();
				Pair<String,String> pair = new Pair<String,String>();
				
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
				String line = null;
				
				// Example line to parse: "WN (1)67492",1,61975
				while((line=br.readLine())!=null) {
					DecoderRecode.parseRecodeMapEntry(line, pair);
					map.put(pair.getKey(), pair.getValue());
				}
				br.close();
				_finalMaps.put(colID, map);
			}
		}
		else {
			fs.close();
			throw new RuntimeException("Path to recode maps must be a directory: " + txMtdDir);
		}
	}	

	/**
	 * 
	 * @param colID
	 * @param key
	 * @return
	 */
	private String lookupRCDMap(int colID, String key) {
		if( _finalMaps!=null )
			return _finalMaps.get(colID).get(key);
		else { //used for cp
			Long tmp = _rcdMaps.get(colID).get(key);
			return (tmp!=null) ? Long.toString(tmp) : null;
		}
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
		
		Iterator<String[]> iter = in.getStringRowIterator();
		while( iter.hasNext() ) {
			String[] row = iter.next(); 
			for( int j=0; j<_colList.length; j++ ) {
				int colID = _colList[j]; //1-based
				//allocate column map if necessary
				if( !_rcdMaps.containsKey(colID) ) 
					_rcdMaps.put(colID, new HashMap<String,Long>());
				//probe and build column map
				HashMap<String,Long> map = _rcdMaps.get(colID);
				String key = row[colID-1];
				if( !map.containsKey(key) )
					map.put(key, new Long(map.size()+1));
			}
		}
	}
	
	/**
	 * Method to apply transformations.
	 * 
	 * @param words
	 * @return
	 */
	@Override
	public String[] apply(String[] words) 
	{
		if( !isApplicable() )
			return words;
		
		//apply recode maps on relevant columns of given row
		for(int i=0; i < _colList.length; i++) {
			//prepare input and get code
			int colID = _colList[i];
			String key = UtilFunctions.unquote(words[colID-1].trim());
			String val = lookupRCDMap(colID, key);			
			// replace unseen keys with NaN 
			words[colID-1] = (val!=null) ? val : "NaN";
		}
			
		return words;
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		//apply recode maps column wise
		for( int j=0; j<_colList.length; j++ ) {
			int colID = _colList[j];
			for( int i=0; i<in.getNumRows(); i++ ) {
				Object okey = in.get(i, colID-1);
				String key = (okey!=null) ? okey.toString() : null;
				String val = lookupRCDMap(colID, key);			
				out.quickSetValue(i, colID-1, (val!=null) ? 
						Double.parseDouble(val) : Double.NaN);
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
	 * @param frame
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
	 * 
	 * @param token
	 * @param code
	 * @return
	 */
	public static String constructRecodeMapEntry(String token, Long code) {
		return token + Lop.DATATYPE_PREFIX + code.toString();
	}
}
 