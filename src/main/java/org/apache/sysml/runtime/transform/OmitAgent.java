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
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.util.UtilFunctions;

public class OmitAgent extends Encoder 
{	
	private static final long serialVersionUID = 1978852120416654195L;

	public OmitAgent() { 
		super(null);
	}
	
	public OmitAgent(int[] list) {
		super(list);
	}
	
	public OmitAgent(JSONObject parsedSpec) 
		throws JSONException 
	{
		super(null);
		if (!parsedSpec.containsKey(TfUtils.TXMETHOD_OMIT))
			return;
		JSONObject obj = (JSONObject) parsedSpec.get(TfUtils.TXMETHOD_OMIT);
		initColList((JSONArray) obj.get(TfUtils.JSON_ATTRS));
	}
	
	public boolean omit(String[] words, TfUtils agents) 
	{
		if( !isApplicable() )
			return false;
		
		for(int i=0; i<_colList.length; i++) {
			int colID = _colList[i];
			if(TfUtils.isNA(agents.getNAStrings(),UtilFunctions.unquote(words[colID-1].trim())))
				return true;
		}
		return false;
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
	public String[] apply(String[] words) {
		return null;
	}
	
	@Override
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		return null;
	}

	@Override
	public double[] encode(String[] in, double[] out) {
		return null;
	}

	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		return null;
	}

	@Override
	public void build(String[] in) {
		
	}

	@Override
	public void build(FrameBlock in) {	
	
	}

	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		return null;
	}
}
 