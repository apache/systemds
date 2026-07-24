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
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Multi-threaded frame parquet reader: reads one file per task, each decoding into its own row range of the shared
 * column backing arrays. Per-file row offsets are derived from the row counts in the file footers.
 */
public class FrameReaderParquetParallel extends FrameReaderParquet {

	@Override
	protected void readParquetFrameFromHDFS(Path path, Configuration conf, Object[] dest, ValueType[] schema,
		String[] names, long rlen) throws IOException {
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		Path[] files = IOUtilFunctions.getSequenceFilePaths(fs, path);
		Arrays.sort(files, Comparator.comparing(Path::getName));
		int numThreads = Math.min(OptimizerUtils.getParallelBinaryReadParallelism(), files.length);

		long[] offsets = new long[files.length];
		long cumulative = 0;
		for(int i = 0; i < files.length; i++) {
			offsets[i] = cumulative;
			try(ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(files[i], conf))) {
				for(BlockMetaData block : reader.getFooter().getBlocks())
					cumulative += block.getRowCount();
			}
		}
		if(cumulative != rlen)
			throw new IOException("Mismatch in row count: expected " + rlen + ", but got " + cumulative);

		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			List<ReadFileTask> tasks = new ArrayList<>();
			for(int i = 0; i < files.length; i++)
				tasks.add(new ReadFileTask(files[i], conf, dest, schema, names, rlen, (int) offsets[i]));

			for(Future<Object> task : pool.invokeAll(tasks))
				task.get();
		}
		catch(Exception e) {
			throw new IOException("Failed parallel read of Parquet frame.", e);
		}
		finally {
			pool.shutdown();
		}
	}

	private class ReadFileTask implements Callable<Object> {
		private final Path path;
		private final Configuration conf;
		private final Object[] dest;
		private final ValueType[] schema;
		private final String[] names;
		private final long rlen;
		private final int rowOffset;

		public ReadFileTask(Path path, Configuration conf, Object[] dest, ValueType[] schema, String[] names, long rlen,
			int rowOffset) {
			this.path = path;
			this.conf = conf;
			this.dest = dest;
			this.schema = schema;
			this.names = names;
			this.rlen = rlen;
			this.rowOffset = rowOffset;
		}

		@Override
		public Object call() throws Exception {
			readSingleParquetFile(path, conf, dest, schema, names, rlen, rowOffset);
			return null;
		}
	}
}
