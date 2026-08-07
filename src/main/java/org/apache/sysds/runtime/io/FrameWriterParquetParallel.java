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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

/**
 * Multi-threaded frame parquet writer. Multi-threaded frame parquet writer.
 *
 */
public class FrameWriterParquetParallel extends FrameWriterParquet {

	private static final String PARQUET_WRITER_TARGET_PART_SIZE_MB = "sysds.io.parquet.writer.target.part.size.mb";

	private static final String PARQUET_WRITER_THREADS = "sysds.io.parquet.writer.threads";

	private static final long DEFAULT_TARGET_PART_SIZE_BYTES = 128L * 1024 * 1024;

	private long getTargetPartSizeBytes() throws IOException {
		String value = System.getProperty(PARQUET_WRITER_TARGET_PART_SIZE_MB);

		if(value != null && !value.trim().isEmpty()) {
			try {
				long mb = Long.parseLong(value.trim());
				if(mb <= 0)
					throw new IOException("Invalid Parquet writer target part size: " + value);
				if(mb > Long.MAX_VALUE / (1024L * 1024L))
					throw new IOException("Parquet writer target part size is too large: " + value);
				return mb * 1024L * 1024L;
			}
			catch(NumberFormatException e) {
				throw new IOException("Invalid value for " + PARQUET_WRITER_TARGET_PART_SIZE_MB + ": " + value, e);
			}
		}

		long hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
		return hdfsBlockSize > 0 ? hdfsBlockSize : DEFAULT_TARGET_PART_SIZE_BYTES;
	}

	private int getMaxWriterThreads() throws IOException {
		int configuredParallelism = OptimizerUtils.getParallelBinaryWriteParallelism();
		String value = System.getProperty(PARQUET_WRITER_THREADS);

		if(value != null && !value.trim().isEmpty()) {
			try {
				int requestedThreads = Integer.parseInt(value.trim());
				if(requestedThreads <= 0)
					throw new IOException("Invalid Parquet writer thread count: " + value);

				return Math.min(requestedThreads, configuredParallelism);
			}
			catch(NumberFormatException e) {
				throw new IOException("Invalid value for " + PARQUET_WRITER_THREADS + ": " + value, e);
			}
		}
		return configuredParallelism;
	}

	private long estimateFrameSizeBytes(FrameBlock src) {
		long rows = src.getNumRows();
		long estimatedRowSize = 0;

		for(ValueType type : src.getSchema()) {
			switch(type) {
				case FP64:
				case INT64:
					estimatedRowSize += 8;
					break;
				case FP32:
				case INT32:
					estimatedRowSize += 4;
					break;
				case BOOLEAN:
					estimatedRowSize += 1;
					break;
				case STRING:
					estimatedRowSize += 16;
					break;
				default:
					estimatedRowSize += 8;
			}
		}

		return Math.max(rows * estimatedRowSize, 1);
	}

	/**
	 * Writes the FrameBlock data to HDFS in parallel. The number of output part files is derived from an estimated byte
	 * size and a target part size. The number of active writer threads is limited independently from the number of part
	 * files.
	 *
	 * @param path The HDFS path where the Parquet files will be written.
	 * @param conf The Hadoop configuration.
	 * @param src  The FrameBlock containing the data to write.
	 */
	@Override
	protected void writeParquetFrameToHDFS(Path path, Configuration conf, FrameBlock src) throws IOException {
		// Estimate number of output partitions
		long estimatedSizeBytes = estimateFrameSizeBytes(src);
		long targetPartSizeBytes = getTargetPartSizeBytes();

		int numPartFiles = (int) Math.max(1, (long) Math.ceil((double) estimatedSizeBytes / targetPartSizeBytes));

		numPartFiles = Math.min(numPartFiles, src.getNumRows());

		int maxWriterThreads = getMaxWriterThreads();
		int numThreads = Math.min(maxWriterThreads, numPartFiles);

		if(!_forcedParallel && numThreads <= 1) {
			super.writeParquetFrameToHDFS(path, conf, src);
			return;
		}

		// Create directory for concurrent tasks
		HDFSTool.createDirIfNotExistOnHDFS(path, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);

		// Materialize default column names before parallel tasks to avoid lazy initialization in workers.
		src.getColumnNames();
		// Create and execute write tasks
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			List<WriteFileTask> tasks = new ArrayList<>();
			int chunkSize = (int) Math.ceil((double) src.getNumRows() / numPartFiles);

			for(int i = 0; i < numPartFiles; i++) {
				int startRow = i * chunkSize;
				int endRow = Math.min((i + 1) * chunkSize, (int) src.getNumRows());
				if (startRow < endRow) {
					Path newPath = new Path(path, IOUtilFunctions.getPartFileName(i));
					tasks.add(new WriteFileTask(newPath, conf, src, startRow, endRow));
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
	
	protected void writeSingleParquetFile(Path path, Configuration conf, FrameBlock src, int startRow, int endRow)
		throws IOException {
		super.writeParquetFrameToHDFS(path, conf, src, startRow, endRow);
	}
	
	private class WriteFileTask implements Callable<Object> {
		private Path path;
		private Configuration conf;
		private FrameBlock src;
		private final int startRow;
		private final int endRow;

		public WriteFileTask(Path path, Configuration conf, FrameBlock src, int startRow, int endRow) {
			this.path = path;
			this.conf = conf;
			this.src = src;
			this.startRow = startRow;
			this.endRow = endRow;
		}

		@Override
		public Object call() throws Exception {
			writeSingleParquetFile(path, conf, src, startRow, endRow);
			return null;
		}
	}
}
