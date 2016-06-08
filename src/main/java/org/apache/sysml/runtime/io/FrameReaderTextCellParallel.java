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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;

/**
 * Multi-threaded frame textcell reader.
 * 
 */
public class FrameReaderTextCellParallel extends FrameReaderTextCell
{	
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
	 * @throws IOException
	 */
	@Override
	protected void readTextCellFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, 
			List<ValueType> schema, List<String> names, long rlen, long clen)
		throws IOException
	{
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		try 
		{
			//create read tasks for all splits
			ExecutorService pool = Executors.newFixedThreadPool(numThreads);
			InputSplit[] splits = informat.getSplits(job, numThreads);
			ArrayList<ReadTask> tasks = new ArrayList<ReadTask>();
			for( InputSplit split : splits )
				tasks.add(new ReadTask(split, informat, job, dest));
			
			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
				
			//check for exceptions
			for( Future<Object> task : rt )
				task.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of text cell input.", e);
		}
	}
	
	/**
	 * 
	 */
	public class ReadTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private FrameBlock _dest = null;
		
		public ReadTask( InputSplit split, TextInputFormat informat, JobConf job, FrameBlock dest ) {
			_split = split;
			_informat = informat;
			_job = job;
			_dest = dest;
		}

		@Override
		public Object call() throws Exception {
			readTextCellFrameFromInputSplit(_split, _informat, _job, _dest);
			return null;
		}
	}
}
