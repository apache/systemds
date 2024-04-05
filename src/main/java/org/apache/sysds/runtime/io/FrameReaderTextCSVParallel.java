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
		int numThreads = OptimizerUtils.getParallelTextReadParallelism();
		
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		InputSplit[] splits = informat.getSplits(job, numThreads); 
		splits = IOUtilFunctions.sortInputSplits(splits);

		ExecutorService pool = CommonThreadPool.get(numThreads);
		try {
			// get number of threads pool to use the common thread pool.
			
			//compute num rows per split
			ArrayList<CountRowsTask> tasks = new ArrayList<>();
			for( int i=0; i<splits.length; i++ )
				tasks.add(new CountRowsTask(splits[i], informat, job, _props.hasHeader() && i==0, clen));
			List<Future<Integer>> cret = pool.invokeAll(tasks);

			//compute row offset per split via cumsum on row counts
			long offset = 0;
			List<Long> offsets = new ArrayList<>();
			for( Future<Integer> count : cret ) {
				offsets.add(offset);
				offset += count.get();
			}
			
			//read individual splits
			ArrayList<ReadRowsTask> tasks2 = new ArrayList<>();
			for( int i=0; i<splits.length; i++ )
				tasks2.add( new ReadRowsTask(splits[i], informat, job, dest, offsets.get(i).intValue(), i==0));
			CommonThreadPool.invokeAndShutdown(pool, tasks2);
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
				tasks.add(new CountRowsTask(splits[i], informat, job, _props.hasHeader()&& i==0, ncol));
			List<Future<Integer>> cret = pool.invokeAll(tasks);
			for( Future<Integer> count : cret ) 
				nrow += count.get().intValue();
			
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

	private static class CountRowsTask implements Callable<Integer> {
		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final JobConf _job;
		private final boolean _hasHeader;
		private final long _nCol;

		public CountRowsTask(InputSplit split, TextInputFormat informat, JobConf job, boolean hasHeader, long nCol) {
			_split = split;
			_informat = informat;
			_job = job;
			_hasHeader = hasHeader;
			_nCol = nCol;
		}

		@Override
		public Integer call() throws Exception {
			return countLinesInReader(_split, _informat, _job, _nCol, _hasHeader);

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
		public Object call() 
			throws Exception 
		{
			try{

				readCSVFrameFromInputSplit(_split, _informat, _job, _dest, _dest.getSchema(), 
						_dest.getColumnNames(), _dest.getNumRows(), _dest.getNumColumns(), _offset, _isFirstSplit);
				return null;
			}
			catch(Exception e){
				e.printStackTrace();
				throw e;
			}
		}
	}
}
