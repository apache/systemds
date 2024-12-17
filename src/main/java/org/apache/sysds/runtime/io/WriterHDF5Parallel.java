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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class WriterHDF5Parallel extends WriterHDF5 {

	public WriterHDF5Parallel(FileFormatPropertiesHDF5 _props) {
		super(_props);
	}

	@Override
	public void writeHDF5MatrixToHDFS(Path path, JobConf job, FileSystem fs, MatrixBlock src)
		throws IOException, DMLRuntimeException {

		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int) (OptimizerUtils
			.estimateSizeTextOutput(src.getNumRows(), src.getNumColumns(), src.getNonZeros(),
				Types.FileFormat.HDF5) / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);

		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelTextWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);

		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( !_forcedParallel && numThreads <= 1 ) {
			super.writeHDF5MatrixToHDFS(path, job, fs, src);
			return;
		}

		//create directory for concurrent tasks
		HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);

		//create and execute tasks
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<WriteHDF5Task> tasks = new ArrayList<>();
			int rlen = src.getNumRows();
			int blklen = (int) Math.ceil((double) rlen / numThreads);
			for(int i = 0; i < numThreads & i * blklen < rlen; i++) {
				Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
				tasks.add(new WriteHDF5Task(newPath, job, fs, src, i * blklen, Math.min((i + 1) * blklen, rlen)));
			}

			for(Future<Object> task : pool.invokeAll(tasks))
				task.get();
		}
		catch(Exception e) {
			throw new IOException("Failed parallel write of HDF5 output.", e);
		}
		finally{
			pool.shutdown();
		}
	}

	private static class WriteHDF5Task implements Callable<Object> {
		private final JobConf _job;
		private final FileSystem _fs;
		private final MatrixBlock _src;
		private final Path _path;
		private final int _rl, _ru;

		public WriteHDF5Task(Path path, JobConf job, FileSystem fs, MatrixBlock src, int rl, int ru) {
			_path = path;
			_job = job;
			_fs = fs;
			_src = src;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() throws IOException {
			writeHDF5MatrixToFile(_path, _job, _fs, _src, _rl, _ru);
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(_job, _path);
			return null;
		}
	}
}
