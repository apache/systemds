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
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Multi-threaded frame parquet reader.
 * 
 */
public class FrameReaderParquetParallel extends FrameReaderParquet {
	
	/**
	 * Reads a Parquet frame in parallel and populates the provided FrameBlock with the data.
	 * The method retrieves all file paths from the sequence files at that location, it then determines 
	 * the number of threads to use based on the available files and a configured parallelism setting.
	 * A thread pool is created to run a reading task for each file concurrently.
	 *
	 * @param path   The HDFS path to the Parquet file or the directory containing sequence files.
	 * @param conf   The Hadoop configuration.
	 * @param dest   The FrameBlock to be updated with the data read from the files.
	 * @param schema The expected value types for the frame columns.
	 * @param rlen   The expected number of rows.
	 * @param clen   The expected number of columns.
	 */
	@Override
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, FrameBlock dest, ValueType[] schema, long rlen, long clen) throws IOException, DMLRuntimeException {
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		Path[] files = IOUtilFunctions.getSequenceFilePaths(fs, path);
		Arrays.sort(files, Comparator.comparing(Path::getName));
		int numThreads = Math.min(OptimizerUtils.getParallelBinaryReadParallelism(), files.length);
		
		long[] offsets = new long[files.length];
		long cumulative = 0;
		
		for (int i = 0; i < files.length; i++) {
			long fileRows = 0;
		
			try (ParquetFileReader reader =
					ParquetFileReader.open(HadoopInputFile.fromPath(files[i], conf))) {
		
				ParquetMetadata meta = reader.getFooter();
		
				for (BlockMetaData block : meta.getBlocks()) {
					fileRows += block.getRowCount();
				}
			}
		
			offsets[i] = cumulative;
			cumulative += fileRows;
		}

		// Create and execute read tasks
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			List<ReadFileTask> tasks = new ArrayList<>();
			for (int i = 0; i < files.length; i++) {
				tasks.add(new ReadFileTask(files[i], conf, dest, clen, offsets[i]));
			}

			for (Future<Object> task : pool.invokeAll(tasks)) {
				task.get();
			}
		} catch (Exception e) {
			throw new IOException("Failed parallel read of Parquet frame.", e);
		} finally {
			pool.shutdown();
		}
	}

	private class ReadFileTask implements Callable<Object> {
		private Path path;
		private Configuration conf;
		private FrameBlock dest;
		private long clen;
		private long rowOffset;

		public ReadFileTask(Path path, Configuration conf, FrameBlock dest, long clen, long rowOffset) {
			this.path = path;
			this.conf = conf;
			this.dest = dest;
			this.clen = clen;
			this.rowOffset = rowOffset;
		}

		// When executed, a ParquetReader for the assigned file opens and iterates over each row processing every column.
		@Override
		public Object call() throws Exception {
			FrameReaderParquetParallel.super.readSingleParquetFile(path, conf, dest, clen, rowOffset);
			return null;
		}
	}
}
