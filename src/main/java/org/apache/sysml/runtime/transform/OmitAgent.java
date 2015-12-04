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

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

import org.apache.sysml.runtime.util.UtilFunctions;

public class OmitAgent extends TransformationAgent {
	
	private static final long serialVersionUID = 1978852120416654195L;

	private int[] _omitList = null;

	OmitAgent() { }
	
	OmitAgent(int[] list) {
		_omitList = list;
	}
	
	public OmitAgent(JSONObject parsedSpec) throws JSONException {
		if (!parsedSpec.containsKey(TX_METHOD.OMIT.toString()))
			return;
		JSONObject obj = (JSONObject) parsedSpec.get(TX_METHOD.OMIT.toString());
		JSONArray attrs = (JSONArray) obj.get(JSON_ATTRS);
		
		_omitList = new int[attrs.size()];
		for(int i=0; i < _omitList.length; i++) 
			_omitList[i] = UtilFunctions.toInt(attrs.get(i));
	}
	
	public boolean omit(String[] words, TfUtils agents) 
	{
		if(_omitList == null)
			return false;
		
		for(int i=0; i<_omitList.length; i++) 
		{
			int colID = _omitList[i];
			if(agents.isNA(UtilFunctions.unquote(words[colID-1].trim())))
				return true;
		}
		return false;
	}
	
	public boolean isApplicable() 
	{
		return (_omitList != null);
	}
	
	/**
	 * Check if the given column ID is subjected to this transformation.
	 * 
	 */
	public int isOmitted(int colID)
	{
		if(_omitList == null)
			return -1;
		
		int idx = Arrays.binarySearch(_omitList, colID);
		return ( idx >= 0 ? idx : -1);
	}

	@Override
	public void print() {
		System.out.print("Omit List: \n    ");
		for(int i : _omitList) 
			System.out.print(i + " ");
		System.out.println();
	}

	@Override
	public void mapOutputTransformationMetadata(
			OutputCollector<IntWritable, DistinctValue> out, int taskID,
			TfUtils agents) throws IOException {
	}

	@Override
	public void mergeAndOutputTransformationMetadata(
			Iterator<DistinctValue> values, String outputDir, int colID,
			FileSystem fs, TfUtils agents) throws IOException {
	}

	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents)
			throws IOException {
	}

	@Override
	public String[] apply(String[] words, TfUtils agents) {
		return null;
	}


}
 