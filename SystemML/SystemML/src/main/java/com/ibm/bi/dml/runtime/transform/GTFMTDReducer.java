package com.ibm.bi.dml.runtime.transform;
import java.io.IOException;
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

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


public class GTFMTDReducer implements Reducer<IntWritable, DistinctValue, Text, LongWritable> {
	
	//private static boolean _hasHeader = false;
	//private static String[] _naStrings = null;	// Strings denoting missing value
	private static String _outputDir = null;
	private static JobConf _rJob = null;
	private static long _numCols = 0;
	private static String _offsetFile = null;
	private static Pattern _delim = null;		// Delimiter in the input file
	private static String[] _naStrings = null;	// Strings denoting missing value
	
	@Override
	public void configure(JobConf job) {
		_outputDir = MRJobConfiguration.getOutputs(job)[0];
		//_hasHeader = Boolean.parseBoolean(job.get(MRJobConfiguration.RCD_HAS_HEADER));
		//_naStrings = job.get(MRJobConfiguration.RCD_NA_STRINGS).split(",");
		_rJob = job;
		_numCols = UtilFunctions.parseToLong( job.get(MRJobConfiguration.TF_NUM_COLS) );		// #of columns in input data
		_offsetFile = job.get(MRJobConfiguration.TF_OFFSETS_FILE);
		
		_delim = Pattern.compile(Pattern.quote(job.get(MRJobConfiguration.TF_DELIM)));
		if ( job.get(MRJobConfiguration.TF_NA_STRINGS) == null)
			_naStrings = null;
		else
			_naStrings = Pattern.compile(Pattern.quote(DataExpression.DELIM_NA_STRING_SEP)).split(job.get(MRJobConfiguration.TF_NA_STRINGS), -1);
		
		TransformationAgent.init(_naStrings, job.get(MRJobConfiguration.TF_HEADER), _delim.pattern());
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
			MVImputeAgent mia = new MVImputeAgent();
			mia.mergeAndOutputTransformationMetadata(values, _outputDir, colID, _rJob);
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
			RecodeAgent ra = new RecodeAgent();
			ra.mergeAndOutputTransformationMetadata(values, _outputDir, colID, _rJob);
		}
		
	}
}
