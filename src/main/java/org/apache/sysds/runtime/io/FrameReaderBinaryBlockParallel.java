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

package org.apache.sysds.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;


/**
 * Multi-threaded frame binary block reader.
 * 
 */
public class FrameReaderBinaryBlockParallel extends FrameReaderBinaryBlock
{
	@Override
	protected void readBinaryBlockFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		int numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
		
		try 
		{
			//create read tasks for all files
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<ReadFileTask> tasks = new ArrayList<>();
			for( Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path) )
				tasks.add(new ReadFileTask(lpath, job, fs, dest));

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();
			
			//check for exceptions
			for( Future<Object> task : rt )
				task.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of binary block input.", e);
		}
	}

	private class ReadFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private FrameBlock _dest = null;
		
		public ReadFileTask(Path path, JobConf job, FileSystem fs, FrameBlock dest) {
			_path = path;
			_fs = fs;
			_job = job;
			_dest = dest;
		}

		@Override
		public Object call() throws Exception {
			readBinaryBlockFrameFromSequenceFile(_path, _job, _fs, _dest);
			return null;
		}
	}
}
