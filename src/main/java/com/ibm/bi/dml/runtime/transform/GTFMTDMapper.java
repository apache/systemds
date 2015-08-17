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
import org.apache.wink.json4j.JSONObject;

import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.JSONHelper;


public class GTFMTDMapper implements Mapper<LongWritable, Text, IntWritable, DistinctValue>{
	
	
	
	private OutputCollector<IntWritable, DistinctValue> _collector = null; 
	private int _mapTaskID = -1;
	
	private static boolean _hasHeader = false;	
	private static Pattern _delim = null;		// Delimiter in the input file
	private static String[] _naStrings = null;	// Strings denoting missing value
	private static String   _specFile = null;	// Transform specification file on HDFS
	private static long _numCols = 0;
	
	OmitAgent _oa = null;
	MVImputeAgent _mia = null;
	RecodeAgent _ra = null;	
	BinAgent _ba = null;
	DummycodeAgent _da = null;
	TfAgents _agents = null;

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
		_naStrings = DataTransform.parseNAStrings(job);
		_specFile = job.get(MRJobConfiguration.TF_SPEC_FILE);
		_numCols = UtilFunctions.parseToLong( job.get(MRJobConfiguration.TF_NUM_COLS) );		// #of columns in input data
	
		FileSystem fs = FileSystem.get(job);
		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(_specFile))));
		return JSONHelper.parse(br);
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
			
			_oa = new OmitAgent(spec);
			_mia = new MVImputeAgent(spec);
			_ra = new RecodeAgent(spec);
			_ba = new BinAgent(spec);
			_da = new DummycodeAgent(spec, _numCols);
			_agents = new TfAgents(_oa, _mia, _ra, _ba, _da);
			
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
		if(!_oa.omit(words))
		{
			_mia.prepare(words);
			_ra.prepare(words);
			_ba.prepare(words);
			TransformationAgent._numValidRecords++;
		}
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
		TransformationAgent._numValidRecords = 0;
		
		_partFileWithHeader = false;
	}
}
