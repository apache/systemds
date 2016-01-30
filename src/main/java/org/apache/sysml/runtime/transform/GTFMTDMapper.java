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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.wink.json4j.JSONException;
import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;


public class GTFMTDMapper implements Mapper<LongWritable, Text, IntWritable, DistinctValue>{
	
	private OutputCollector<IntWritable, DistinctValue> _collector = null; 
	private int _mapTaskID = -1;
	
	TfUtils _agents = null;

	private boolean _partFileWithHeader = false;
	private boolean _firstRecordInSplit = true;
	private String _partFileName = null;
	private long _offsetInPartFile = -1;
	
	// ----------------------------------------------------------------------------------------------
	
	/**
	 * Configure the information used in the mapper, and setup transformation agents.
	 */
	@Override
	public void configure(JobConf job) {
		String[] parts = job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID).split("_");
		if ( parts[0].equalsIgnoreCase("task")) {
			_mapTaskID = Integer.parseInt(parts[parts.length-1]);
		}
		else if ( parts[0].equalsIgnoreCase("attempt")) {
			_mapTaskID = Integer.parseInt(parts[parts.length-2]);
		}
		else {
			throw new RuntimeException("Unrecognized format for taskID: " + job.get(MRConfigurationNames.MR_TASK_ATTEMPT_ID));
		}

		try {
			_partFileName = TfUtils.getPartFileName(job);
			_partFileWithHeader = TfUtils.isPartFileWithHeader(job);
			_agents = new TfUtils(job);
		} catch(IOException e) { throw new RuntimeException(e); }
		  catch(JSONException e)  { throw new RuntimeException(e); }

	}
	
	
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<IntWritable, DistinctValue> out, Reporter reporter) throws IOException  {
		
		if(_firstRecordInSplit)
		{
			_firstRecordInSplit = false;
			_collector = out;
			_offsetInPartFile = rawKey.get();
		}
		
		// ignore header
		if (_agents.hasHeader() && rawKey.get() == 0 && _partFileWithHeader)
			return;
		
		_agents.prepareTfMtd(rawValue.toString());
	}

	@Override
	public void close() throws IOException {
		_agents.getMVImputeAgent().mapOutputTransformationMetadata(_collector, _mapTaskID, _agents);
		_agents.getRecodeAgent().mapOutputTransformationMetadata(_collector, _mapTaskID, _agents);
		_agents.getBinAgent().mapOutputTransformationMetadata(_collector, _mapTaskID, _agents);
		
		// Output part-file offsets to create OFFSETS_FILE, which is to be used in csv reblocking.
		// OffsetCount is denoted as a DistinctValue by concatenating parfile name and offset within partfile.
		_collector.collect(new IntWritable((int)_agents.getNumCols()+1), new DistinctValue(new OffsetCount(_partFileName, _offsetInPartFile, _agents.getValid())));
		
		// reset global variables, required when the jvm is reused.
		_firstRecordInSplit = true;
		_offsetInPartFile = -1;
		_partFileWithHeader = false;
	}
	
}
