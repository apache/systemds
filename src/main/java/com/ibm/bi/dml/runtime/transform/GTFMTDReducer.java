/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.regex.Pattern;

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

import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONObject;


public class GTFMTDReducer implements Reducer<IntWritable, DistinctValue, Text, LongWritable> {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//private static boolean _hasHeader = false;
	//private static String[] _naStrings = null;	// Strings denoting missing value
	private static String _outputDir = null;
	private static JobConf _rJob = null;
	private static long _numCols = 0;
	private static String _offsetFile = null;
	private static Pattern _delim = null;		// Delimiter in the input file
	private static String[] _naStrings = null;	// Strings denoting missing value

	TfAgents _agents = null;
	
	@Override
	public void configure(JobConf job) {
		_outputDir = MRJobConfiguration.getOutputs(job)[0];
		//_hasHeader = Boolean.parseBoolean(job.get(MRJobConfiguration.RCD_HAS_HEADER));
		//_naStrings = job.get(MRJobConfiguration.RCD_NA_STRINGS).split(",");
		_rJob = job;
		_numCols = UtilFunctions.parseToLong( job.get(MRJobConfiguration.TF_NUM_COLS) );		// #of columns in input data
		_offsetFile = job.get(MRJobConfiguration.TF_OFFSETS_FILE);
		
		_delim = Pattern.compile(Pattern.quote(job.get(MRJobConfiguration.TF_DELIM)));
		_naStrings = DataTransform.parseNAStrings(job);
		
		TransformationAgent.init(_naStrings, job.get(MRJobConfiguration.TF_HEADER), _delim.pattern());
		
		try {
			String specFile = job.get(MRJobConfiguration.TF_SPEC_FILE);
			FileSystem fs = FileSystem.get(job);
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(specFile))));
			JSONObject spec = JSONObject.parse(br);
			MVImputeAgent mia = new MVImputeAgent(spec);
			RecodeAgent ra = new RecodeAgent(spec);
			_agents = new TfAgents(null, mia, ra, null, null);
		} 
		catch(IOException e) 
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public void close() throws IOException {
		// TODO Auto-generated method stub
		
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void reduce(IntWritable key, Iterator<DistinctValue> values,
			OutputCollector<Text, LongWritable> output, Reporter reporter)
			throws IOException {
		
		int colID = key.get();
		if(colID < 0) 
		{
			// process mapper output for MV and Bin agents
			colID = colID*-1;
			_agents.getMVImputeAgent().mergeAndOutputTransformationMetadata(values, _outputDir, colID, _rJob, _agents);
		}
		else if ( colID == _numCols + 1)
		{
			// process mapper output for OFFSET_FILE
			ArrayList<OffsetCount> list = new ArrayList<OffsetCount>();
			while(values.hasNext())
				list.add(new OffsetCount(values.next().getOffsetCount()));
			Collections.sort(list);
			
			FileSystem fs = FileSystem.get(_rJob);
			@SuppressWarnings("deprecation")
			SequenceFile.Writer writer = new SequenceFile.Writer(fs, _rJob, new Path(_offsetFile+"/part-00000"), ByteWritable.class, OffsetCount.class);
			
			long lineOffset=0;
			for(OffsetCount oc: list)
			{
				long count=oc.count;
				oc.count=lineOffset;
				writer.append(new ByteWritable((byte)0), oc);
				lineOffset+=count;
			}
			reporter.incrCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_ROWS, lineOffset);
			writer.close();
			list.clear();

		}
		else 
		{
			// process mapper output for Recode agent
			_agents.getRecodeAgent().mergeAndOutputTransformationMetadata(values, _outputDir, colID, _rJob, _agents);
		}
		
	}
}
