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

package org.apache.sysml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;

/**
 * Multi-threaded frame text csv reader.
 * 
 */
public class FrameReaderTextCSVParallel extends FrameReaderTextCSV
{
	public FrameReaderTextCSVParallel(CSVFileFormatProperties props) {
		super(props);
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException 
	 */
	@Override
	protected void readCSVFrameFromHDFS( Path path, JobConf job, FileSystem fs, 
			FrameBlock dest, List<ValueType> schema, List<String> names, long rlen, long clen) 
		throws IOException
	{
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, numThreads); 
		splits = IOUtilFunctions.sortInputSplits(splits);

		try 
		{
			ExecutorService pool = Executors.newFixedThreadPool(numThreads);
			
			//compute num rows per split
			ArrayList<CountRowsTask> tasks = new ArrayList<CountRowsTask>();
			for( int i=0; i<splits.length; i++ )
				tasks.add(new CountRowsTask(splits[i], informat, job, _props.hasHeader(), i==0));
			List<Future<Long>> cret = pool.invokeAll(tasks);

			//compute row offset per split via cumsum on row counts
			long offset = 0;
			List<Long> offsets = new ArrayList<Long>();
			for( Future<Long> count : cret ) {
				offsets.add(offset);
				offset += count.get();
			}
			
			//read individial splits
			ArrayList<ReadRowsTask> tasks2 = new ArrayList<ReadRowsTask>();
			for( int i=0; i<splits.length; i++ )
				tasks2.add( new ReadRowsTask(splits[i], informat, job, dest, offsets.get(i).intValue(), i==0));
			List<Future<Object>> rret = pool.invokeAll(tasks2);
			pool.shutdown();
			
			//error handling
			for( Future<Object> read : rret )
				read.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of text csv input.", e);
		}
	}

	@Override
	protected Pair<Integer,Integer> computeCSVSize( Path path, JobConf job, FileSystem fs) 
		throws IOException 
	{		
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, numThreads);
		
		//compute number of columns
		RecordReader<LongWritable, Text> reader = informat.getRecordReader(splits[0], job, Reporter.NULL);
		LongWritable key = new LongWritable();
		Text value = new Text();
		reader.next(key, value);
		int ncol = StringUtils.countMatches(value.toString(), _props.getDelim()) + 1;
		reader.close();
		
		//compute number of rows
		ExecutorService pool = Executors.newFixedThreadPool(numThreads);
		
		//compute num rows per split
		int nrow = 0;
		try {
			ArrayList<CountRowsTask> tasks = new ArrayList<CountRowsTask>();
			for( int i=0; i<splits.length; i++ )
				tasks.add(new CountRowsTask(splits[i], informat, job, _props.hasHeader(), i==0));
			List<Future<Long>> cret = pool.invokeAll(tasks);
			for( Future<Long> count : cret ) 
				nrow += count.get().intValue();
		}
		catch (Exception e) {
			throw new IOException("Failed parallel read of text csv input.", e);
		}
		
		return new Pair<Integer,Integer>(nrow, ncol);
	}
	
	/**
	 * 
	 * 
	 */
	private static class CountRowsTask implements Callable<Long> 
	{
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private boolean _hasHeader = false;
		private boolean _firstSplit = false;

		public CountRowsTask(InputSplit split, TextInputFormat informat, JobConf job, boolean hasHeader, boolean first) {
			_split = split;
			_informat = informat;
			_job = job;
			_hasHeader = hasHeader;
			_firstSplit = first;
		}

		@Override
		public Long call() 
			throws Exception 
		{
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text value = new Text();
			long nrows = 0;
			
			// count rows from the first non-header row
			try {
				if ( _firstSplit && _hasHeader )
					reader.next(key, value);
				while (reader.next(key, value))
					nrows++;
			} 
			finally {
				IOUtilFunctions.closeSilently(reader);
			}

			return nrows;
		}
	}

	/**
	 * 
	 * 
	 */
	private class ReadRowsTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private FrameBlock _dest = null;
		private int _offset = -1;
		private boolean _isFirstSplit = false;
		
		
		public ReadRowsTask(InputSplit split, TextInputFormat informat, JobConf job, 
				FrameBlock dest, int offset, boolean first) 
		{
			_split = split;
			_informat = informat;
			_job = job;
			_dest = dest;
			_offset = offset;
			_isFirstSplit = first;
		}

		@Override
		public Object call() 
			throws Exception 
		{
			readCSVFrameFromInputSplit(_split, _informat, _job, _dest, _dest.getSchema(), 
					_dest.getColumnNames(), _dest.getNumRows(), _dest.getNumColumns(), _offset, _isFirstSplit);
			return null;
		}
	}
}
