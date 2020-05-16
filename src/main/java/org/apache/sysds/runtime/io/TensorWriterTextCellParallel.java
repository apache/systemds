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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class TensorWriterTextCellParallel extends TensorWriterTextCell {
	@Override
	protected void writeTextCellTensorToHDFS(Path path, FileSystem fs, TensorBlock src) throws IOException {
		//estimate output size and number of output blocks (min 1)
		// TODO accurate estimation
		int numPartFiles = (int) (OptimizerUtils.estimateSizeTextOutput(src.getDims(), src.getLength(),
				FileFormat.TEXT) / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);
		
		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( numThreads <= 1 ) {
			super.writeTextCellTensorToHDFS(path, fs, src);
			return;
		}
		
		//create directory for concurrent tasks
		HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//create and execute tasks
		try {
			ExecutorService pool = CommonThreadPool.get(numThreads);
			ArrayList<WriteTextTask> tasks = new ArrayList<>();
			int rlen = src.getNumRows();
			int blklen = (int) Math.ceil((double) rlen / numThreads);
			for (int i = 0; i < numThreads & i * blklen < rlen; i++) {
				Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
				tasks.add(new WriteTextTask(newPath, fs, src, i * blklen, Math.min((i + 1) * blklen, rlen)));
			}
			
			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();
			
			//check for exceptions
			for (Future<Object> task : rt)
				task.get();
			
			// delete crc files if written to local file system
			if( fs instanceof LocalFileSystem ) {
				for (int i = 0; i < numThreads & i * blklen < rlen; i++)
					IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs,
							new Path(path, IOUtilFunctions.getPartFileName(i)));
			}
		}
		catch (Exception e) {
			throw new IOException("Failed parallel write of text output.", e);
		}
	}
	
	private static class WriteTextTask implements Callable<Object> {
		private FileSystem _fs;
		private TensorBlock _src;
		private Path _path;
		private int _rl;
		private int _ru;
		
		public WriteTextTask(Path path, FileSystem fs, TensorBlock src, int rl, int ru) {
			_path = path;
			_fs = fs;
			_src = src;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws Exception {
			writeTextCellTensorToFile(_path, _fs, _src, _rl, _ru);
			return null;
		}
	}
}
