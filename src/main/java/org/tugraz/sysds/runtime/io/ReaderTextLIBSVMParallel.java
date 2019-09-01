/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.io;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.SparseRowVector;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.CommonThreadPool;

public class ReaderTextLIBSVMParallel extends MatrixReader
{
	private int _numThreads = 1;
	private SplitOffsetInfos _offsets = null;

	public ReaderTextLIBSVMParallel() {
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen,
			int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		// prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		InputSplit[] splits = informat.getSplits(job, _numThreads);
		splits = IOUtilFunctions.sortInputSplits(splits);

		// check existence and non-empty file
		checkValidInputFile(fs, path);

		// allocate output matrix block
		// First Read Pass (count rows/cols, determine offsets, allocate matrix block)
		MatrixBlock ret = computeLIBSVMSizeAndCreateOutputMatrixBlock(splits, path, job, rlen, clen, estnnz);
		rlen = ret.getNumRows();
		clen = ret.getNumColumns();

		// Second Read Pass (read, parse strings, append to matrix block)
		readLIBSVMMatrixFromHDFS(splits, path, job, ret, rlen, clen, blen);
		
		//post-processing (representation-specific, change of sparse/dense block representation)
		// - nnz explicitly maintained in parallel for the individual splits
		ret.examSparsity();

		// sanity check for parallel row count (since determined internally)
		if (rlen >= 0 && rlen != ret.getNumRows())
			throw new DMLRuntimeException("Read matrix inconsistent with given meta data: "
					+ "expected nrow="+ rlen + ", real nrow=" + ret.getNumRows());

		return ret;
	}
	
	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//not implemented yet, fallback to sequential reader
		return new ReaderTextLIBSVM()
			.readMatrixFromInputStream(is, rlen, clen, blen, estnnz);
	}
	
	private void readLIBSVMMatrixFromHDFS(InputSplit[] splits, Path path, JobConf job, 
			MatrixBlock dest, long rlen, long clen, int blen) 
		throws IOException 
	{
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		ExecutorService pool = CommonThreadPool.get(_numThreads);

		try 
		{
			// create read tasks for all splits
			ArrayList<LIBSVMReadTask> tasks = new ArrayList<>();
			int splitCount = 0;
			for (InputSplit split : splits) {
				tasks.add( new LIBSVMReadTask(split, _offsets, informat, job, dest, rlen, clen, splitCount++) );
			}
			pool.invokeAll(tasks);
			pool.shutdown();

			// check return codes and aggregate nnz
			long lnnz = 0;
			for (LIBSVMReadTask rt : tasks) {
				lnnz += rt.getPartialNnz();
				if (!rt.getReturnCode()) {
					Exception err = rt.getException();
					throw new IOException("Read task for libsvm input failed: "+ err.toString(), err);
				}
			}
			dest.setNonZeros(lnnz);
		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}
	
	private MatrixBlock computeLIBSVMSizeAndCreateOutputMatrixBlock(InputSplit[] splits, Path path,
			JobConf job, long rlen, long clen, long estnnz)
		throws IOException, DMLRuntimeException 
	{
		int nrow = 0;
		int ncol = (int) clen;
		
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);

		// count rows in parallel per split
		try 
		{
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			ArrayList<CountRowsTask> tasks = new ArrayList<>();
			for (InputSplit split : splits) {
				tasks.add(new CountRowsTask(split, informat, job));
			}
			pool.invokeAll(tasks);
			pool.shutdown();

			// collect row counts for offset computation
			// early error notify in case not all tasks successful
			_offsets = new SplitOffsetInfos(tasks.size());
			for (CountRowsTask rt : tasks) {
				if (!rt.getReturnCode())
					throw new IOException("Count task for libsvm input failed: "+ rt.getErrMsg());
				_offsets.setOffsetPerSplit(tasks.indexOf(rt), nrow);
				_offsets.setLenghtPerSplit(tasks.indexOf(rt), rt.getRowCount());
				nrow = nrow + rt.getRowCount();
			}
		} 
		catch (Exception e) {
			throw new IOException("Threadpool Error " + e.getMessage(), e);
		}
		
		//robustness for wrong dimensions which are already compiled into the plan
		if( (rlen != -1 && nrow != rlen) || (clen != -1 && ncol != clen) ) {
			String msg = "Read matrix dimensions differ from meta data: ["+nrow+"x"+ncol+"] vs. ["+rlen+"x"+clen+"].";
			if( rlen < nrow || clen < ncol ) {
				//a) specified matrix dimensions too small
				throw new DMLRuntimeException(msg);
			}
			else {
				//b) specified matrix dimensions too large -> padding and warning
				LOG.warn(msg);
				nrow = (int) rlen;
				ncol = (int) clen;
			}
		}
		
		// allocate target matrix block based on given size; 
		// need to allocate sparse as well since lock-free insert into target
		long estnnz2 = (estnnz < 0) ? (long)nrow * ncol : estnnz;
		return createOutputMatrixBlock(nrow, ncol, nrow, estnnz2, true, true);
	}
	
	private static class SplitOffsetInfos {
		// offset & length info per split
		private int[] offsetPerSplit = null;
		private int[] lenghtPerSplit = null;

		public SplitOffsetInfos(int numSplits) {
			lenghtPerSplit = new int[numSplits];
			offsetPerSplit = new int[numSplits];
		}

		public int getLenghtPerSplit(int split) {
			return lenghtPerSplit[split];
		}

		public void setLenghtPerSplit(int split, int r) {
			lenghtPerSplit[split] = r;
		}

		public int getOffsetPerSplit(int split) {
			return offsetPerSplit[split];
		}

		public void setOffsetPerSplit(int split, int o) {
			offsetPerSplit[split] = o;
		}
	}
	
	private static class CountRowsTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private boolean _rc = true;
		private String _errMsg = null;
		private int _nrows = -1;

		public CountRowsTask(InputSplit split, TextInputFormat informat, JobConf job) {
			_split = split;
			_informat = informat;
			_job = job;
			_nrows = 0;
		}

		public boolean getReturnCode() {
			return _rc;
		}

		public int getRowCount() {
			return _nrows;
		}
		
		public String getErrMsg() {
			return _errMsg;
		}

		@Override
		public Object call() 
			throws Exception 
		{
			RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			LongWritable key = new LongWritable();
			Text oneLine = new Text();

			try {
				// count rows from the first row
				while (reader.next(key, oneLine)) {
					_nrows++;
				}
			} 
			catch (Exception e) {
				_rc = false;
				_errMsg = "RecordReader error libsvm format. split: "+ _split.toString() + e.getMessage();
				throw new IOException(_errMsg);
			} 
			finally {
				IOUtilFunctions.closeSilently(reader);
			}

			return null;
		}
	}
	
	private static class LIBSVMReadTask implements Callable<Object> 
	{
		private InputSplit _split = null;
		private SplitOffsetInfos _splitoffsets = null;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private MatrixBlock _dest = null;
		private long _clen = -1;
		private int _splitCount = 0;
		
		private boolean _rc = true;
		private Exception _exception = null;
		private long _nnz;
		
		public LIBSVMReadTask(InputSplit split, SplitOffsetInfos offsets,
				TextInputFormat informat, JobConf job, MatrixBlock dest,
				long rlen, long clen, int splitCount) 
		{
			_split = split;
			_splitoffsets = offsets; // new SplitOffsetInfos(offsets);
			_informat = informat;
			_job = job;
			_dest = dest;
			_clen = clen;
			_rc = true;
			_splitCount = splitCount;
		}

		public boolean getReturnCode() {
			return _rc;
		}

		public Exception getException() {
			return _exception;
		}
		
		public long getPartialNnz() {
			return _nnz;
		}
		
		@Override
		public Object call() 
			throws Exception 
		{
			long lnnz = 0;
			
			try 
			{
				RecordReader<LongWritable, Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
				LongWritable key = new LongWritable();
				Text value = new Text();
				SparseRowVector vect = new SparseRowVector(1024);
				
				int row = _splitoffsets.getOffsetPerSplit(_splitCount);

				try {
					while (reader.next(key, value)) { // foreach line
						String rowStr = value.toString().trim();
						lnnz += ReaderTextLIBSVM.parseLibsvmRow(rowStr, vect, (int)_clen);
						_dest.appendRow(row, vect);
						row++;
					}

					// sanity checks (number of rows)
					if (row != (_splitoffsets.getOffsetPerSplit(_splitCount) + _splitoffsets.getLenghtPerSplit(_splitCount)) ) {
						throw new IOException("Incorrect number of rows ("+ row+ ") found in delimited file ("
							+ (_splitoffsets.getOffsetPerSplit(_splitCount) 
							+ _splitoffsets.getLenghtPerSplit(_splitCount))+ "): " + value);
					}
				} 
				finally {
					IOUtilFunctions.closeSilently(reader);
				}
			} 
			catch (Exception ex) {
				// central error handling (return code, message)
				_rc = false;
				_exception = ex;
			}

			//post processing
			_nnz = lnnz;
			return null;
		}
	}
}
