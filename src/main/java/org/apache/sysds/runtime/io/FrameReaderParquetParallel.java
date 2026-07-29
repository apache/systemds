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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
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

	private Path[] getParquetDataFilePaths(FileSystem fs, Path path) throws IOException {
		FileStatus status = fs.getFileStatus(path);

		if (status.isFile())
			return new Path[] {path};

		List<Path> files = new ArrayList<>();
		for (FileStatus child : fs.listStatus(path)) {
			if(child.isFile() && isParquetDataFile(child.getPath()))
				files.add(child.getPath());
		}

		return files.toArray(new Path[0]);
	}

	private boolean isParquetDataFile(Path path) {
		String name = path.getName();

		return !name.startsWith("_")
			&& !name.startsWith(".")
			&& !name.endsWith(".crc");
	}

	private long getParquetRowCount(Path path, Configuration conf) throws IOException {
		long rowCount = 0;
		try (ParquetFileReader fileReader = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
			for (BlockMetaData block : fileReader.getFooter().getBlocks()) {
				rowCount += block.getRowCount();
			}
		}
		return rowCount;
	}
	
	/**
	 * Reads a Parquet frame in parallel and populates the provided FrameBlock with the data.
	 * The method retrieves all Parquet data file paths at the given location, it then determines 
	 * the number of threads to use based on the available files and a configured parallelism setting.
	 * A thread pool is created to run a reading task for each file concurrently.
	 *
	 * @param path   The HDFS path to the Parquet file or the directory containing part files.
	 * @param conf   The Hadoop configuration.
	 * @param dest   The FrameBlock to be updated with the data read from the files.
	 * @param rlen   The expected number of rows.
	 * @param clen   The expected number of columns.
	 */
	@Override
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, FrameBlock dest, long rlen, long clen) throws IOException, DMLRuntimeException {
		FileSystem fs = IOUtilFunctions.getFileSystem(path, conf);
		Path[] files = getParquetDataFilePaths(fs, path);
		
		if (files.length == 0)
			throw new IOException("No Parquet data files found at path: " + path);

		Arrays.sort(files);
		long[] rowCounts = new long[files.length];
		long totalRows = 0;

		for (int i = 0; i < files.length; i++) {
			rowCounts[i] = getParquetRowCount(files[i], conf);
			totalRows += rowCounts[i];
		}

		if (rlen >= 0 && totalRows != rlen)
			throw new IOException("Mismatch in row count: expected " + rlen + ", but got " + totalRows);

		int numThreads = Math.min(OptimizerUtils.getParallelBinaryReadParallelism(), files.length);
		// Create and execute read tasks
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			List<ReadFileTask> tasks = new ArrayList<>();
			long rowOffset = 0;

			for (int i = 0; i < files.length; i++) {
				tasks.add(new ReadFileTask(files[i], conf, dest, clen, rowOffset, rowCounts[i]));
				rowOffset += rowCounts[i];
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
		private long expectedRows;

		public ReadFileTask(Path path, Configuration conf, FrameBlock dest, long clen, long rowOffset, long expectedRows) {
			this.path = path;
			this.conf = conf;
			this.dest = dest;
			this.clen = clen;
			this.rowOffset = rowOffset;
			this.expectedRows = expectedRows;
		}

		// When executed, a ParquetReader for the assigned file opens and iterates over each row processing every column.
		@Override
		public Object call() throws Exception {
			MessageType parquetSchema;
			try (ParquetFileReader fileReader = ParquetFileReader.open(HadoopInputFile.fromPath(path, conf))) {
				parquetSchema = fileReader.getFooter().getFileMetaData().getSchema();
			}
			String[] columnNames = dest.getColumnNames();
			int[] columnIndices = getParquetColumnIndices(parquetSchema, columnNames);
			PrimitiveType.PrimitiveTypeName[] columnTypes = getParquetColumnTypes(parquetSchema, columnIndices);
			try (ParquetReader<Group> reader = ParquetReader.builder(new GroupReadSupport(), path).withConf(conf).build()) {
				Group group;
				long localRow = 0;

				while ((group = reader.read()) != null) {
					if(localRow >= expectedRows)
						throw new IOException("Mismatch in row count for file " + path + ": expected " + expectedRows + ", but got more rows.");
					int outRow = Math.toIntExact(rowOffset + localRow);
					for (int col = 0; col < clen; col++) {
						int colIndex = columnIndices[col];
						dest.set(outRow, col, readTypedParquetValue(group, columnTypes[col], colIndex));
					}
					localRow++;
				}

				if (localRow != expectedRows)
					throw new IOException("Mismatch in row count for file " + path + ": expected " + expectedRows + ", but got " + localRow);
			}
			return null;
		}
	}
}