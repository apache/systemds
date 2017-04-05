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

package org.apache.sysml.runtime.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;

/**
 * Custom input format and record reader to redirect common implementation of csv read 
 * over record readers (which are required for the parallel readers) to an input stream.
 * 
 */
public class InputStreamInputFormat implements InputFormat<LongWritable, Text>
{
	private final InputStream _input;
		
	public InputStreamInputFormat(InputStream is) {
		_input = is;
	}

	@Override
	public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
		//return dummy handle - stream accessed purely over record reader
		return new InputSplit[]{new FileSplit(null)};
	}

	@Override
	public RecordReader<LongWritable,Text> getRecordReader(InputSplit split, JobConf job, Reporter reporter) throws IOException {
		return new InputStreamRecordReader(_input);
	}
		
	private static class InputStreamRecordReader implements RecordReader<LongWritable, Text>
	{
		private final BufferedReader _reader;
		
		public InputStreamRecordReader(InputStream is) {
			_reader = new BufferedReader(new InputStreamReader( is ));
		}
		
		@Override
		public LongWritable createKey() {
			return new LongWritable();
		}
		@Override
		public Text createValue() {
			return new Text();
		}			
		@Override
		public float getProgress() throws IOException {
			return 0;
		}
		@Override
		public long getPos() throws IOException {
			return 0;
		}
		@Override
		public boolean next(LongWritable key, Text value) throws IOException {
			String line = _reader.readLine();
			if( line != null )
				value.set(line);
			return (line != null);
		}
		@Override
		public void close() throws IOException {
			_reader.close();
		}
	}
}
