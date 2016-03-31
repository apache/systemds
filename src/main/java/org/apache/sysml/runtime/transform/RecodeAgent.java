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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import scala.Tuple2;

import com.google.common.collect.Ordering;

import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.transform.MVImputeAgent.MVMethod;
import org.apache.sysml.runtime.transform.decode.DecoderRecode;
import org.apache.sysml.runtime.util.UtilFunctions;

public class RecodeAgent extends TransformationAgent {
	
	private static final long serialVersionUID = 8213163881283341874L;

	private int[] _rcdList = null;
	private int[] _mvrcdList = null;
	private int[] _fullrcdList = null;

	// HashMap< columnID, HashMap<distinctValue, count> >
	private HashMap<Integer, HashMap<String, Long>> _rcdMaps  = new HashMap<Integer, HashMap<String, Long>>();
	
	RecodeAgent(JSONObject parsedSpec) throws JSONException {
		
		int rcdCount = 0;
		
		if ( parsedSpec.containsKey(TX_METHOD.RECODE.toString())) 
		{
			//TODO consolidate external and internal json spec definitions
			JSONArray attrs = null;
			if( parsedSpec.get(TX_METHOD.RECODE.toString()) instanceof JSONObject ) {
				JSONObject obj = (JSONObject) parsedSpec.get(TX_METHOD.RECODE.toString());
				attrs = (JSONArray) obj.get(JSON_ATTRS);
			}
			else
				attrs = (JSONArray)parsedSpec.get(TX_METHOD.RECODE.toString());
			
			_rcdList = new int[attrs.size()];
			for(int i=0; i < _rcdList.length; i++) 
				_rcdList[i] = UtilFunctions.toInt(attrs.get(i));
			rcdCount = _rcdList.length;
		}
		
		if ( parsedSpec.containsKey(TX_METHOD.MVRCD.toString())) 
		{
			JSONObject obj = (JSONObject) parsedSpec.get(TX_METHOD.MVRCD.toString());
			JSONArray attrs = (JSONArray) obj.get(JSON_ATTRS);
			
			_mvrcdList = new int[attrs.size()];
			for(int i=0; i < _mvrcdList.length; i++) 
				_mvrcdList[i] = UtilFunctions.toInt(attrs.get(i));
			rcdCount += attrs.size();
		}
		
		if ( rcdCount > 0 )
		{
			_fullrcdList = new int[rcdCount];
			int idx = -1;
			if(_rcdList != null)
				for(int i=0; i < _rcdList.length; i++)
					_fullrcdList[++idx] = _rcdList[i]; 
			
			if(_mvrcdList != null)
				for(int i=0; i < _mvrcdList.length; i++)
					_fullrcdList[++idx] = _mvrcdList[i]; 
		}
	}
	
	/**
	 * Construct the recodemaps from the given input frame for all 
	 * columns registered for recode.
	 * 
	 * @param frame
	 */
	public void initRecodeMaps( FrameBlock frame ) {
		for( int j=0; j<_rcdList.length; j++ ) {
			int colID = _rcdList[j]; //1-based
			HashMap<String,Long> map = new HashMap<String,Long>();
			for( int i=0; i<frame.getNumRows(); i++ ) {
				String[] tmp = frame.get(i, colID-1).toString().split(Lop.DATATYPE_PREFIX);
				map.put(tmp[0], Long.parseLong(tmp[1]));
			}
			_rcdMaps.put(colID, map);
		}
	}
	
	void prepare(String[] words, TfUtils agents) {
		if ( _rcdList == null && _mvrcdList == null )
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
	
	public ArrayList<Tuple2<Integer, DistinctValue>> mapOutputTransformationMetadata(int taskID, ArrayList<Tuple2<Integer, DistinctValue>> list, TfUtils agents) throws IOException {
		mapOutputHelper(taskID, null, list, agents);
		return list;
	}
	
	public void mapOutputHelper(int taskID, OutputCollector<IntWritable, DistinctValue> out, ArrayList<Tuple2<Integer, DistinctValue>> list, TfUtils agents) throws IOException {
		if ( _rcdList == null  && _mvrcdList == null )
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
							list.add(new Tuple2<Integer,DistinctValue>(colID, new DistinctValue(s, map.get(s))) );
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
		
		boolean isRecoded = (isRecoded(colID) != -1);
		boolean isModeImputed = (mvagent.getMethod(colID) == MVMethod.GLOBAL_MODE);
		
		Path pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + RCD_MAP_FILE_SUFFIX);
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
		Ordering<String> valueComparator = Ordering.natural();
		List<String> newNames = valueComparator.sortedCopy(map.keySet());

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
			pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + MODE_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(UtilFunctions.quote(mode) + "," + modeIndex + "," + maxCount );
			br.close();
		
			// output number of distinct values
			pt=new Path(outputDir+"/Recode/"+ agents.getName(colID) + NDISTINCT_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(""+map.size());
			br.close();
		}
		
		if (isModeImputed) 
		{
			pt=new Path(outputDir+"/Impute/"+ agents.getName(colID) + MV_FILE_SUFFIX);
			br=new BufferedWriter(new OutputStreamWriter(fs.create(pt,true)));
			br.write(colID + "," + UtilFunctions.quote(mode));
			br.close();
		}
		
	}
	
	public void outputTransformationMetadata(String outputDir, FileSystem fs, TfUtils agents) throws IOException {
		if(_rcdList == null && _mvrcdList == null )
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
	
	// ------------------------------------------------------------------------------------------------
	
	public HashMap<Integer, HashMap<String,Long>> getCPRecodeMaps() { return _rcdMaps; }
	
	HashMap<Integer, HashMap<String,String>> _finalMaps = null;
	public HashMap<Integer, HashMap<String,String>> getRecodeMaps() {
		return _finalMaps;
	}
	
	/**
	 * Method to load recode maps of all attributes, at once.
	 * 
	 * @param job
	 * @throws IOException
	 */
	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents) throws IOException {
		if ( _rcdList == null )
			return;
		
		_finalMaps = new HashMap<Integer, HashMap<String, String>>();
	
		if(fs.isDirectory(txMtdDir)) {
			for(int i=0; i<_rcdList.length;i++) {
				int colID = _rcdList[i];
				
				Path path = new Path( txMtdDir + "/Recode/" + agents.getName(colID) + RCD_MAP_FILE_SUFFIX);
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
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	public int isRecoded(int colID)
	{
		if(_rcdList == null)
			return -1;
		
		int idx = Arrays.binarySearch(_rcdList, colID);
		return ( idx >= 0 ? idx : -1);
	}

	

	/**
	 * Method to apply transformations.
	 * 
	 * @param words
	 * @return
	 */
	@Override
	public String[] apply(String[] words, TfUtils agents) {
		if ( _rcdList == null )
			return words;
		
		//apply recode maps on relevant columns of given row
		for(int i=0; i < _rcdList.length; i++) {
			//prepare input and get code
			int colID = _rcdList[i];
			String key = UtilFunctions.unquote(words[colID-1].trim());
			String val = _finalMaps.get(colID).get(key);			
			// replace unseen keys with NaN 
			words[colID-1] = (val!=null) ? val : "NaN";
		}
			
		return words;
	}
	
	/**
	 * 
	 * @param words
	 * @param agents
	 * @return
	 */
	public String[] cp_apply(String[] words, TfUtils agents) {
		if ( _rcdList == null )
			return words;
		
		//apply recode maps on relevant columns of given row
		for(int i=0; i < _rcdList.length; i++) {
			//prepare input and get code
			int colID = _rcdList[i];
			String key = UtilFunctions.unquote(words[colID-1].trim());
			Long val = _rcdMaps.get(colID).get(key);			
			// replace unseen keys with NaN
			words[colID-1] = (val!=null) ? Long.toString(val) : "NaN";
		}
			
		return words;
	}
	
	public void printMaps() {
		for(Integer k : _rcdMaps.keySet()) {
			System.out.println("Column " + k);
			HashMap<String,Long> map = _rcdMaps.get(k);
			for(String w : map.keySet()) {
				System.out.println("    " + w + " : " + map.get(w));
			}
		}
	}
	
	public void print() {
		System.out.print("Recoding List: \n    ");
		for(int i : _rcdList) {
			System.out.print(i + " ");
		}
		System.out.println();
	}
}
 