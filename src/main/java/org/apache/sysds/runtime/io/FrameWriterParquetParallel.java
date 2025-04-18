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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

/**
 * Multi-threaded frame parquet reader.
 * 
 */
public class FrameWriterParquetParallel extends FrameWriterParquet {

	/**
	 * Writes the FrameBlock data to HDFS in parallel. 
	 * The method estimates the number of output partitions by comparing the total number of cells in the FrameBlock with the
	 * HDFS block size. It then determines the number of threads to use based on the parallelism configuration and the
	 * number of partitions. In case of parallelism, it divides the FrameBlock into chunks and a thread pool is created to
	 * execute a write task for each partition concurrently.
	 *
	 * @param path The HDFS path where the Parquet files will be written.
	 * @param conf The Hadoop configuration.
	 * @param src  The FrameBlock containing the data to write.
	 */
	@Override
	protected void writeParquetFrameToHDFS(Path path, Configuration conf, FrameBlock src) 
		throws IOException, DMLRuntimeException 
	{
		// Estimate number of output partitions
		int numPartFiles = Math.max((int) (src.getNumRows() * src.getNumColumns() / InfrastructureAnalyzer.getHDFSBlockSize()), 1);
		
		// Determine parallelism
		int numThreads = Math.min(OptimizerUtils.getParallelBinaryWriteParallelism(), numPartFiles);

		// Fall back to sequential write if numThreads <= 1
		if (numThreads <= 1) {
			super.writeParquetFrameToHDFS(path, conf, src);
			return;
		}

		// Create directory for concurrent tasks
		HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);

		// Create and execute write tasks
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			List<WriteFileTask> tasks = new ArrayList<>();
			int chunkSize = (int) Math.ceil((double) src.getNumRows() / numThreads);

			for (int i = 0; i < numThreads; i++) {
				int startRow = i * chunkSize;
				int endRow = Math.min((i + 1) * chunkSize, (int) src.getNumRows());
				if (startRow < endRow) {
					Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
					tasks.add(new WriteFileTask(newPath, conf, src.slice(startRow, endRow - 1)));
				}
			}

			for (Future<Object> task : pool.invokeAll(tasks))
				task.get();
		} catch (Exception e) {
			throw new IOException("Failed parallel write of Parquet frame.", e);
		} finally {
			pool.shutdown();
		}
	}
	
	protected void writeSingleParquetFile(Path path, Configuration conf, FrameBlock src)
		throws IOException, DMLRuntimeException 
	{
		super.writeParquetFrameToHDFS(path, conf, src);
	}
	
	private class WriteFileTask implements Callable<Object> {
		private Path path;
		private Configuration conf;
		private FrameBlock src;

		public WriteFileTask(Path path, Configuration conf, FrameBlock src) {
			this.path = path;
			this.conf = conf;
			this.src = src;
		}

		@Override
		public Object call() throws Exception {
			writeSingleParquetFile(path, conf, src);
			return null;
		}
	}
}
