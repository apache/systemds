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
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions.CountRowsTask;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class FrameReaderJSONLParallel extends FrameReaderJSONL
{
	@Override
	protected void readJSONLFrameFromHDFS(Path path, JobConf jobConf, FileSystem fileSystem,
			FrameBlock dest, Types.ValueType[] schema, Map<String, Integer> schemaMap) 
		throws IOException
	{
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();

		TextInputFormat inputFormat = new TextInputFormat();
		inputFormat.configure(jobConf);
		InputSplit[] splits = inputFormat.getSplits(jobConf, numThreads);
		splits = IOUtilFunctions.sortInputSplits(splits);

		try{
			ExecutorService executorPool = CommonThreadPool.get(Math.min(numThreads, splits.length));

			//compute num rows per split
			ArrayList<CountRowsTask> countRowsTasks = new ArrayList<>();
			for (InputSplit split : splits){
				countRowsTasks.add(new CountRowsTask(split, inputFormat, jobConf));
			}
			List<Future<Long>> ret = executorPool.invokeAll(countRowsTasks);

			//compute row offset per split via cumsum on row counts
			long offset = 0;
			List<Long> offsets = new ArrayList<>();
			for( Future<Long> rc : ret ) {
				offsets.add(offset);
				offset += rc.get();
			}

			//read individual splits
			ArrayList<ReadRowsTask> readRowsTasks = new ArrayList<>();
			for( int i=0; i<splits.length; i++ )
				readRowsTasks.add(new ReadRowsTask(splits[i], inputFormat,
					jobConf, dest, schemaMap, offsets.get(i).intValue()));
			CommonThreadPool.invokeAndShutdown(executorPool, readRowsTasks);
		}
		catch (Exception e) {
			throw new IOException("Failed parallel read of JSONL input.", e);
		}
	}

	private class ReadRowsTask implements Callable<Object>{
		private InputSplit _split;
		private TextInputFormat _inputFormat;
		private JobConf _jobConf;
		private FrameBlock _dest;
		Map<String, Integer> _schemaMap;
		private int _offset;

		public ReadRowsTask(InputSplit split, TextInputFormat inputFormat, JobConf jobConf,
			FrameBlock dest, Map<String, Integer> schemaMap, int offset)
		{
			_split = split;
			_inputFormat = inputFormat;
			_jobConf = jobConf;
			_dest = dest;
			_schemaMap = schemaMap;
			_offset = offset;
		}

		@Override
		public Object call() throws Exception {
			readJSONLFrameFromInputSplit(_split, _inputFormat, _jobConf, _dest.getSchema(), _schemaMap, _dest, _offset);
			return null;
		}
	}
}
