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
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;

/**
 * Single-threaded frame text csv writer.
 * 
 */
public class FrameWriterTextCSVParallel extends FrameWriterTextCSV
{
	public FrameWriterTextCSVParallel( FileFormatPropertiesCSV props ) {
		super(props);
	}

	@Override
	protected void writeCSVFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen, FileFormatPropertiesCSV csvprops ) 
		throws IOException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = Math.max((int)(OptimizerUtils.estimateSizeTextOutput(rlen, clen, rlen*clen, 
				              FileFormat.CSV)  / InfrastructureAnalyzer.getHDFSBlockSize()), 1);
		
		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( numThreads <= 1 ) {
			super.writeCSVFrameToHDFS(path, job, src, rlen, clen, csvprops);
			return;
		}
		
		//create directory for concurrent tasks
		HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//create and execute tasks
		try 
		{
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<WriteFileTask> tasks = new ArrayList<>();
			int blklen = (int)Math.ceil((double)rlen / numThreads);
			for(int i=0; i<numThreads & i*blklen<rlen; i++) {
				Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
				tasks.add(new WriteFileTask(newPath, job, fs, src, i*blklen, (int)Math.min((i+1)*blklen, rlen), csvprops));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
			
			//check for exceptions 
			for( Future<Object> task : rt )
				task.get();
			
			// delete crc files if written to local file system
			if (fs instanceof LocalFileSystem) {
				for(int i=0; i<numThreads & i*blklen<rlen; i++) 
					IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs,
						new Path(path, IOUtilFunctions.getPartFileName(i)));
			}
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel write of csv output.", e);
		}
	}
	
	private class WriteFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private FrameBlock _src = null;
		private int _rl = -1;
		private int _ru = -1;
		private FileFormatPropertiesCSV _csvprops = null;
		
		public WriteFileTask(Path path, JobConf job, FileSystem fs, FrameBlock src, int rl, int ru, FileFormatPropertiesCSV csvprops) {
			_path = path;
			_fs = fs;
			_job = job;
			_src = src;
			_rl = rl;
			_ru = ru;
			_csvprops = csvprops;
		}
	
		@Override
		public Object call() throws Exception  {
			writeCSVFrameToFile(_path, _job, _fs, _src, _rl, _ru, _csvprops);
			return null;
		}
	}
}
