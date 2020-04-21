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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.FastStringTokenizer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class TensorReaderTextCellParallel extends TensorReaderTextCell {
	private int _numThreads = OptimizerUtils.getParallelTextReadParallelism();
	
	@Override
	protected TensorBlock readTextCellTensorFromHDFS(Path path, JobConf job, long[] dims,
			Types.ValueType[] schema) throws IOException {
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		int[] idims = Arrays.stream(dims).mapToInt(i -> (int) i).toArray();
		TensorBlock ret;
		if( schema.length == 1 )
			ret = new TensorBlock(schema[0], idims).allocateBlock();
		else
			ret = new TensorBlock(schema, idims).allocateBlock();
		try {
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			InputSplit[] splits = informat.getSplits(job, _numThreads);
			
			//create and execute read tasks for all splits
			List<TensorReaderTextCellParallel.ReadTask> tasks = Arrays.stream(splits)
					.map(s -> new TensorReaderTextCellParallel.ReadTask(s, informat, job, ret))
					.collect(Collectors.toList());
			List<Future<Object>> rt = pool.invokeAll(tasks);
			
			//check for exceptions
			for (Future<Object> task : rt)
				task.get();
			
			pool.shutdown();
		}
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
		return ret;
	}
	
	private static class ReadTask implements Callable<Object> {
		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final JobConf _job;
		private final TensorBlock _dest;
		
		public ReadTask(InputSplit split, TextInputFormat informat, JobConf job, TensorBlock dest) {
			_split = split;
			_informat = informat;
			_job = job;
			_dest = dest;
		}
		
		@Override
		public Object call() throws Exception {
			LongWritable key = new LongWritable();
			Text value = new Text();
			try {
				FastStringTokenizer st = new FastStringTokenizer(' ');
				
				int[] ix = new int[_dest.getNumDims()];
				RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
				try {
					while (reader.next(key, value)) {
						st.reset(value.toString());
						for (int i = 0; i < ix.length; i++) {
							ix[i] = st.nextInt() - 1;
						}
						_dest.set(ix, st.nextToken());
					}
				}
				finally {
					IOUtilFunctions.closeSilently(reader);
				}
			}
			catch (Exception ex) {
				throw new IOException("Unable to read tensor in text cell format.", ex);
			}
			return null;
		}
	}
}
