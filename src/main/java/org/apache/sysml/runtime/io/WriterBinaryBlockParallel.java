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
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

public class WriterBinaryBlockParallel extends WriterBinaryBlock
{
	public WriterBinaryBlockParallel( int replication ) {
		super(replication);
	}
	
	@Override
	protected void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException, DMLRuntimeException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int)(OptimizerUtils.estimatePartitionedSizeExactSparsity(rlen, clen, 
				brlen, bclen, src.getNonZeros()) / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);
		
		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelBinaryWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( numThreads <= 1 ) {
			super.writeBinaryBlockMatrixToHDFS(path, job, fs, src, rlen, clen, brlen, bclen);
			return;
		}

		//create directory for concurrent tasks
		MapReduceTool.createDirIfNotExistOnHDFS(path.toString(), DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//create and execute write tasks
		try 
		{
			ExecutorService pool = Executors.newFixedThreadPool(numThreads);
			ArrayList<WriteFileTask> tasks = new ArrayList<WriteFileTask>();
			int blklen = (int)Math.ceil((double)rlen / brlen / numThreads) * brlen;
			for(int i=0; i<numThreads & i*blklen<rlen; i++) {
				Path newPath = new Path(path, String.format("0-m-%05d",i));
				tasks.add(new WriteFileTask(newPath, job, fs, src, i*blklen, Math.min((i+1)*blklen, rlen), brlen, bclen));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
			
			//check for exceptions 
			for( Future<Object> task : rt )
				task.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel write of binary block input.", e);
		}
	}

	/**
	 * 
	 */
	private class WriteFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private MatrixBlock _src = null;
		private long _rl = -1;
		private long _ru = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public WriteFileTask(Path path, JobConf job, FileSystem fs, MatrixBlock src, long rl, long ru, int brlen, int bclen) {
			_path = path;
			_fs = fs;
			_job = job;
			_src = src;
			_rl = rl;
			_ru = ru;
			_brlen = brlen;
			_bclen = bclen;
		}
	
		@Override
		public Object call() 
			throws Exception 
		{
			writeBinaryBlockMatrixToSequenceFile(_path, _job, _fs, _src, _brlen, _bclen, (int)_rl, (int)_ru);
			return null;
		}
	}
}
