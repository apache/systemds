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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class TensorReaderBinaryBlockParallel extends TensorReaderBinaryBlock {
	private final int _numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	
	@Override
	protected TensorBlock readBinaryBlockTensorFromHDFS(Path path, JobConf job, FileSystem fs, long[] dims, int blen,
			ValueType[] schema) throws IOException {
		int[] idims = Arrays.stream(dims).mapToInt(i -> (int) i).toArray();
		TensorBlock ret;
		if( schema.length == 1 )
			ret = new TensorBlock(schema[0], idims).allocateBlock();
		else
			ret = new TensorBlock(schema, idims).allocateBlock();
		try {
			//create read tasks for all files
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			ArrayList<TensorReaderBinaryBlockParallel.ReadFileTask> tasks = new ArrayList<>();
			for (Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path)) {
				TensorReaderBinaryBlockParallel.ReadFileTask t = new TensorReaderBinaryBlockParallel.ReadFileTask(lpath, job, ret, dims, blen);
				tasks.add(t);
			}
			
			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			pool.shutdown();
			
			//check for exceptions
			for (Future<Object> task : rt)
				task.get();
		}
		catch (Exception e) {
			throw new IOException("Failed parallel read of binary block input.", e);
		}
		return ret;
	}
	
	private static class ReadFileTask implements Callable<Object> {
		private final Path _path;
		private final JobConf _job;
		private final TensorBlock _dest;
		private final long[] _dims;
		private final int _blen;
		
		public ReadFileTask(Path path, JobConf job, TensorBlock dest, long[] dims, int blen) {
			_path = path;
			_job = job;
			_dest = dest;
			_dims = dims;
			_blen = blen;
		}
		
		@Override
		public Object call() throws Exception {
			TensorBlock value = new TensorBlock();
			TensorIndexes key = new TensorIndexes();
			//directly read from sequence files (individual partfiles)
			try(SequenceFile.Reader reader = new SequenceFile.Reader(_job, SequenceFile.Reader.file(_path))) {
				//note: next(key, value) does not yet exploit the given serialization classes, 
				//record reader does but is generally slower.
				while (reader.next(key, value)) {
					if( value.isEmpty(false) )
						continue;
					int[] lower = new int[_dims.length];
					int[] upper = new int[lower.length];
					UtilFunctions.getBlockBounds(key, value.getLongDims(), _blen, lower, upper);
					_dest.copy(lower, upper, value);
				}
			}
			
			return null;
		}
	}
}
