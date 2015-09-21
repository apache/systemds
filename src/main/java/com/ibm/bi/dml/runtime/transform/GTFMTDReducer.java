/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.transform;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.wink.json4j.JSONException;

import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class GTFMTDReducer implements Reducer<IntWritable, DistinctValue, Text, LongWritable> {
	
	private JobConf _rJob = null;
	TfUtils _agents = null;
	
	@Override
	public void configure(JobConf job) {
		_rJob = job;
		
		try {
			String outputDir = MRJobConfiguration.getOutputs(job)[0];
			_agents = new TfUtils(job, outputDir);
		} 
		catch(IOException e)  { throw new RuntimeException(e); }
		catch(JSONException e)  { throw new RuntimeException(e); }
	}

	@Override
	public void close() throws IOException {
	}
	
	@Override
	public void reduce(IntWritable key, Iterator<DistinctValue> values,
			OutputCollector<Text, LongWritable> output, Reporter reporter)
			throws IOException {
		
		FileSystem fs = FileSystem.get(_rJob);
		
		int colID = key.get();
		
		if(colID < 0) 
		{
			// process mapper output for MV and Bin agents
			colID = colID*-1;
			_agents.getMVImputeAgent().mergeAndOutputTransformationMetadata(values, _agents.getTfMtdDir(), colID, fs, _agents);
		}
		else if ( colID == _agents.getNumCols() + 1)
		{
			// process mapper output for OFFSET_FILE
			ArrayList<OffsetCount> list = new ArrayList<OffsetCount>();
			while(values.hasNext())
				list.add(new OffsetCount(values.next().getOffsetCount()));
			
			long numTfRows = generateOffsetsFile(list);
			reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_ROWS, numTfRows);

		}
		else 
		{
			// process mapper output for Recode agent
			_agents.getRecodeAgent().mergeAndOutputTransformationMetadata(values, _agents.getTfMtdDir(), colID, fs, _agents);
		}
		
	}
	
	@SuppressWarnings("unchecked")
	private long generateOffsetsFile(ArrayList<OffsetCount> list) throws IllegalArgumentException, IOException 
	{
		Collections.sort(list);
		
		@SuppressWarnings("deprecation")
		SequenceFile.Writer writer = new SequenceFile.Writer(
				FileSystem.get(_rJob), _rJob, 
				new Path(_agents.getOffsetFile()+"/part-00000"), 
				ByteWritable.class, OffsetCount.class);
		
		long lineOffset=0;
		for(OffsetCount oc: list)
		{
			long count=oc.count;
			oc.count=lineOffset;
			writer.append(new ByteWritable((byte)0), oc);
			lineOffset+=count;
		}
		writer.close();
		list.clear();
		
		return lineOffset;
	}
	
}

