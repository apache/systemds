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

package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;
import org.apache.sysml.runtime.transform.TfUtils;

public class CSVAssignRowIDMapper extends MapReduceBase implements Mapper<LongWritable, Text, ByteWritable, OffsetCount>
{	
	private ByteWritable outKey = new ByteWritable();
	private long fileOffset = 0;
	private long num = 0;
	private boolean first = true;
	private OutputCollector<ByteWritable, OffsetCount> outCache = null;
	private String delim = " ";
	private boolean ignoreFirstLine = false;
	private boolean realFirstLine = false;
	private String filename = "";
	private boolean headerFile = false;
	
	// members relevant to transform
	private TfUtils _agents = null;
	
	@Override
	public void map(LongWritable key, Text value,
			OutputCollector<ByteWritable, OffsetCount> out, Reporter report)
			throws IOException 
	{
		if(first) {
			first = false;
			fileOffset = key.get();
			outCache = out;
		}
		
		//getting the number of colums
		if(key.get()==0 && headerFile) {
			if(!ignoreFirstLine) {
				report.incrCounter(CSVReblockMR.NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim, -1).length);
				num += omit(value.toString()) ? 0 : 1;
			}
			else
				realFirstLine = true;
		}
		else {
			if(realFirstLine) {
				report.incrCounter(CSVReblockMR.NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim, -1).length);
				realFirstLine = false;
			}
			num += omit(value.toString()) ? 0 : 1;
		}
	}
	
	@Override
	@SuppressWarnings("deprecation")
	public void configure(JobConf job)
	{	
		byte thisIndex;
		try {
			//it doesn't make sense to have repeated file names in the input, since this is for reblock
			thisIndex = MRJobConfiguration.getInputMatrixIndexesInMapper(job).get(0);
			outKey.set(thisIndex);
			FileSystem fs = FileSystem.get(job);
			Path thisPath = new Path(job.get(MRConfigurationNames.MR_MAP_INPUT_FILE)).makeQualified(fs);
			filename = thisPath.toString();
			String[] strs = job.getStrings(CSVReblockMR.SMALLEST_FILE_NAME_PER_INPUT);
			Path headerPath = new Path(strs[thisIndex]).makeQualified(fs);
			headerFile = headerPath.toString().equals(filename);
		
			CSVReblockInstruction[] reblockInstructions = MRJobConfiguration.getCSVReblockInstructions(job);
			for(CSVReblockInstruction ins: reblockInstructions)
				if(ins.input == thisIndex) {
					delim = Pattern.quote(ins.delim); 
					ignoreFirstLine = ins.hasHeader;
					break;
				}
		
			// load properties relevant to transform
			boolean omit = job.getBoolean(MRJobConfiguration.TF_TRANSFORM, false);
			if ( omit ) 
				_agents = new TfUtils(job, true);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	private boolean omit(String line) {
		if(_agents == null)
			return false;		
		return _agents.omit( line.split(delim, -1) );
	}
	
	@Override
	public void close() throws IOException {
		if( outCache != null ) //robustness empty splits
			outCache.collect(outKey, new OffsetCount(filename, fileOffset, num));
	}
}
