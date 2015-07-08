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
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.json.java.JSONObject;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;


public class GTFMTDMapper implements Mapper<LongWritable, Text, IntWritable, DistinctValue>{
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private OutputCollector<IntWritable, DistinctValue> _collector = null; 
	private int _mapTaskID = -1;
	
	private static boolean _hasHeader = false;	
	private static Pattern _delim = null;		// Delimiter in the input file
	private static String[] _naStrings = null;	// Strings denoting missing value
	private static String   _specFile = null;	// Transform specification file on HDFS
	private static long _numCols = 0;
	
	MVImputeAgent _mia = null;
	RecodeAgent _ra = null;	
	BinAgent _ba = null;
	DummycodeAgent _da = null;

	private static boolean _partFileWithHeader = false;
	
	private static boolean _firstRecordInSplit = true;
	private static String _partFileName = null;
	private static long _offsetInPartFile = -1;
	
	/*private int[] str2intArray(String s) {
		String[] fields = s.split(",");
		int[] ret = new int[fields.length];
		for(int i=0; i < fields.length; i++)
			ret[i] = Integer.parseInt(fields[i]);
		return ret;
	}*/
	
	
	private JSONObject loadTransformSpecs(JobConf job) throws IllegalArgumentException, IOException {
		_hasHeader = Boolean.parseBoolean(job.get(MRJobConfiguration.TF_HAS_HEADER));
		_delim = Pattern.compile(Pattern.quote(job.get(MRJobConfiguration.TF_DELIM)));
		if ( job.get(MRJobConfiguration.TF_NA_STRINGS) == null)
			_naStrings = null;
		else
			_naStrings = Pattern.compile(Pattern.quote(DataExpression.DELIM_NA_STRING_SEP)).split(job.get(MRJobConfiguration.TF_NA_STRINGS), -1);
		_specFile = job.get(MRJobConfiguration.TF_SPEC_FILE);
		_numCols = UtilFunctions.parseToLong( job.get(MRJobConfiguration.TF_NUM_COLS) );		// #of columns in input data
	
		FileSystem fs = FileSystem.get(job);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(_specFile))));
		return JSONObject.parse(br);
	}
	
	@SuppressWarnings("deprecation")
	@Override
	public void configure(JobConf job) {
		String[] parts = job.get("mapred.task.id").split("_");
		if ( parts[0].equalsIgnoreCase("task")) {
			_mapTaskID = Integer.parseInt(parts[parts.length-1]);
		}
		else if ( parts[0].equalsIgnoreCase("attempt")) {
			_mapTaskID = Integer.parseInt(parts[parts.length-2]);
		}
		else {
			throw new RuntimeException("Unrecognized format for taskID: " + job.get("mapred.task.id"));
		}

		FileSystem fs;
		try {
			JSONObject spec = loadTransformSpecs(job);

			TransformationAgent.init(_naStrings, job.get(MRJobConfiguration.TF_HEADER), _delim.pattern());
			_mia = new MVImputeAgent(spec);
			_ra = new RecodeAgent(spec);
			_ba = new BinAgent(spec);
			_da = new DummycodeAgent(spec, _numCols);
			
			fs = FileSystem.get(job);
			Path thisPath=new Path(job.get("map.input.file")).makeQualified(fs);
			_partFileName = thisPath.toString();
			Path smallestFilePath=new Path(job.get(MRJobConfiguration.TF_SMALLEST_FILE)).makeQualified(fs);
			if(_partFileName.toString().equals(smallestFilePath.toString()))
				_partFileWithHeader=true;
			else
				_partFileWithHeader = false;
			
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	
	public void map(LongWritable rawKey, Text rawValue, OutputCollector<IntWritable, DistinctValue> out, Reporter reporter) throws IOException  {
		
		if(_firstRecordInSplit)
		{
			_firstRecordInSplit = false;
			_collector = out;
			_offsetInPartFile = rawKey.get();
		}
		
		
		// ignore header
		// TODO: check the behavior in multi-part CSV file
		if (_hasHeader && rawKey.get() == 0 && _partFileWithHeader)
			return;
		
		String[] words = _delim.split(rawValue.toString(),-1);
		_mia.prepare(words);
		_ra.prepare(words);
		_ba.prepare(words);
		
		TransformationAgent._numRecordsInPartFile++;
	}

	@Override
	public void close() throws IOException {
		_mia.mapOutputTransformationMetadata(_collector, _mapTaskID, _mia);
		_ra.mapOutputTransformationMetadata(_collector, _mapTaskID, _mia);
		_ba.mapOutputTransformationMetadata(_collector, _mapTaskID, _mia);
		
		// Output part-file offsets to create OFFSETS_FILE, which is to be used in csv reblocking.
		// OffsetCount is denoted as a DistinctValue by concatenating parfile name and offset within partfile.
		_collector.collect(new IntWritable((int)_numCols+1), new DistinctValue(new OffsetCount(_partFileName, _offsetInPartFile, TransformationAgent._numRecordsInPartFile)));
		
		// reset global variables, required when the jvm is reused.
		_firstRecordInSplit = true;
		_offsetInPartFile = -1;
		TransformationAgent._numRecordsInPartFile = 0;
		_partFileWithHeader = false;
	}
}
