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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.stats.Timing;

/**
 * Multi-threaded frame text csv reader.
 * 
 */
public class FrameReaderTextCSVParallel extends FrameReaderTextCSV
{
	public FrameReaderTextCSVParallel(FileFormatPropertiesCSV props) {
		super(props);
	}

	@Override
	protected void readCSVFrameFromHDFS( Path path, JobConf job, FileSystem fs, 
			FrameBlock dest, ValueType[] schema, String[] names, long rlen, long clen) 
		throws IOException
	{
		Timing time = new Timing(true);
		final int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, numThreads);
		if(HDFSTool.isDirectory(fs, path))
			splits = IOUtilFunctions.sortInputSplits(splits);

			
		final ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			if(splits.length == 1){
				new ReadRowsTask(splits[0], informat, job, dest, 0, true).call();
				return;
			}

			//compute num rows per split
			ArrayList<Future<Long>> cret = new ArrayList<>();
			for( int i=0; i<splits.length - 1; i++ ) // all but last split
				cret.add(pool.submit(new CountRowsTask(splits[i], informat, job, _props.hasHeader() && i==0)));
		
			LOG.debug("Spawned all row counting tasks CSV : " + time.stop());
			//compute row offset per split via cumsum on row counts
			long offset = 0;
			ArrayList<Future<Object>> tasks2 = new ArrayList<>();
			for( int i=0; i<splits.length -1; i++ ){
				long tmp = cret.get(i).get(); // ensure the subsequent task has a thread to use.
				tasks2.add(pool.submit(new ReadRowsTask(splits[i], informat, job, dest, (int) offset, i==0)));
				offset += tmp;
			}
			tasks2.add(pool.submit(new ReadRowsTask(splits[splits.length-1], informat, job, dest, (int) offset, splits.length==1)));

			LOG.debug("Spawned all reading tasks CSV : " + time.stop());
			//read individual splits
			for(Future<Object> a : tasks2)
				a.get();
			LOG.debug("Finished Reading CSV : " + time.stop());
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of text csv input.", e);
		}
		finally{
			pool.shutdown();
		}
	}

	@Override
	protected Pair<Integer,Integer> computeCSVSize( Path path, JobConf job, FileSystem fs) 
		throws IOException 
	{
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, numThreads);
		
		//compute number of columns
		int ncol = IOUtilFunctions.countNumColumnsCSV(splits, informat, job, _props.getDelim());
		
		//compute number of rows
		long nrow = 0;
		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			ArrayList<CountRowsTask> tasks = new ArrayList<>();
			for( int i=0; i<splits.length; i++ )
				tasks.add(new CountRowsTask(splits[i], informat, job, _props.hasHeader()&& i==0));
			List<Future<Long>> cret = pool.invokeAll(tasks);
			for( Future<Long> count : cret ) 
				nrow += count.get().longValue();
			
			if(nrow > Integer.MAX_VALUE)
				throw new DMLRuntimeException("invalid read with over Integer number of rows");
				
			return new Pair<>((int)nrow, ncol);
		}
		catch (Exception e) {
			throw new IOException("Failed parallel read of text csv input.", e);
		}
		finally {
			pool.shutdown();
		}
	}

	private static class CountRowsTask implements Callable<Long> {
		private InputSplit _split;
		private TextInputFormat _informat;
		private JobConf _job;
		private boolean _hasHeader;


		public CountRowsTask(InputSplit split, TextInputFormat informat, JobConf job, boolean hasHeader) {
			_split = split;
			_informat = informat;
			_job = job;
			_hasHeader = hasHeader;
		}

		@Override
		public Long call() throws Exception {
			long count =  countLinesInSplit(_split, _informat, _job, _hasHeader);
			return count;
		}
	}

	private class ReadRowsTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private FrameBlock _dest = null;
		private int _offset = -1;
		private boolean _isFirstSplit = false;
		
		
		public ReadRowsTask(InputSplit split, TextInputFormat informat, JobConf job, 
			FrameBlock dest, int offset, boolean first) 
		{
			_split = split;
			_informat = informat;
			_job = job;
			_dest = dest;
			_offset = offset;
			_isFirstSplit = first;
		}

		@Override
		public Object call() throws Exception {
			readCSVFrameFromInputSplit(_split, _informat, _job, _dest, _dest.getSchema(), 
				_dest.getColumnNames(), _dest.getNumRows(), _dest.getNumColumns(), _offset, _isFirstSplit);
			return null;
		}
	}
}
