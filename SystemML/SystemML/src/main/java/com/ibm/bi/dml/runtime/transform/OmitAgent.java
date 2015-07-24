/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

public class OmitAgent extends TransformationAgent {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int[] _omitList = null;

	OmitAgent() { }
	
	OmitAgent(int[] list) {
		_omitList = list;
	}
	
	OmitAgent(JSONObject parsedSpec) {
		Object obj = parsedSpec.get(TX_METHOD.OMIT.toString());
		if(obj == null) {
			
		}
		else {
			JSONArray attrs = (JSONArray) ((JSONObject)obj).get(JSON_ATTRS);
			_omitList = new int[attrs.size()];
			for(int i=0; i < _omitList.length; i++) 
				_omitList[i] = ((Long) attrs.get(i)).intValue();
		}
	}
	
	boolean omit(String[] words) 
	{
		if(_omitList == null)
			return false;
		
		for(int i=0; i<_omitList.length; i++) 
		{
			int colID = _omitList[i];
			if(MVImputeAgent.isNA(words[colID-1], TransformationAgent.NAstrings))
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
		
		for(int i=0; i < _omitList.length; i++)
			if( _omitList[i] == colID )
				return i;
		
		return -1;
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
			TransformationAgent agent) throws IOException {
	}

	@Override
	public void mergeAndOutputTransformationMetadata(
			Iterator<DistinctValue> values, String outputDir, int colID,
			JobConf job) throws IOException {
	}

	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir)
			throws IOException {
	}

	@Override
	public String[] apply(String[] words) {
		return null;
	}


}
 