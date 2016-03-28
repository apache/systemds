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

package org.apache.sysml.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * Parallel version of ReaderTextCell.java. To summarize, we create read tasks per split
 * and use a fixed-size thread pool, to executed these tasks. If the target matrix is dense,
 * the inserts are done lock-free. If the matrix is sparse, we use a buffer to collect
 * unordered input cells, lock the the target sparse matrix once, and append all buffered values.
 * 
 * Note MatrixMarket:
 * 1) For matrix market files each read task probes for comments until it finds data because
 *    for very small tasks or large comments, any split might encounter % or %%. Hence,
 *    the parallel reader does not do the validity check for.
 * 2) In extreme scenarios, the last comment might be in one split, and the following meta data
 *    in the subsequent split. This would create incorrect results or errors. However, this
 *    scenario is extremely unlikely (num threads > num lines if 1 comment line) and hence ignored 
 *    similar to our parallel MR setting (but there we have a 128MB guarantee).     
 * 3) However, we use MIN_FILESIZE_MM (8KB) to give guarantees for the common case of small headers
 *    in order the issue described in (2).
 * 
 */
public class ReaderTextCellParallel extends MatrixReader
{
	private static final long MIN_FILESIZE_MM = 8L * 1024; //8KB
	
	private boolean _isMMFile = false;
	private int _numThreads = 1;
	
	public ReaderTextCellParallel(InputInfo info)
	{
		_isMMFile = (info == InputInfo.MatrixMarketInputInfo);
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path);
		
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, (int)rlen, (int)clen, estnnz, true, false);
	
		//core read 
		readTextCellMatrixFromHDFS(path, job, ret, rlen, clen, brlen, bclen, _isMMFile);

		//finally check if change of sparse/dense block representation required
		if( !AGGREGATE_BLOCK_NNZ )
			ret.recomputeNonZeros();			
		ret.examSparsity();
		
		return ret;
	}


	/**
	 * 
	 * @param path
	 * @param job
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 */
	private void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int brlen, int bclen, boolean matrixMarket )
		throws IOException
	{
		int par = _numThreads;
		
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		//check for min file size for matrix market (adjust num splits if necessary)
		if( _isMMFile ){
			long len = MapReduceTool.getFilesizeOnHDFS(path);
			par = ( len < MIN_FILESIZE_MM ) ? 1: par; 
		}	
		
		try 
		{
			//create read tasks for all splits
			ExecutorService pool = Executors.newFixedThreadPool(par);
			InputSplit[] splits = informat.getSplits(job, par);
			ArrayList<ReadTask> tasks = new ArrayList<ReadTask>();
			for( InputSplit split : splits ){
				ReadTask t = new ReadTask(split, informat, job, dest, rlen, clen, matrixMarket);
				tasks.add(t);
			}
			
			//wait until all tasks have been executed
			List<Future<Long>> rt = pool.invokeAll(tasks);	
			
			//check for exceptions and aggregate nnz
			long lnnz = 0;
			for( Future<Long> task : rt )
				lnnz += task.get();
				
			//post-processing
			dest.setNonZeros( lnnz );
			if( dest.isInSparseFormat() ) {
				ArrayList<SortRowsTask> tasks2 = new ArrayList<SortRowsTask>();
				int blklen = (int)(Math.ceil((double)rlen/_numThreads));
				for( int i=0; i<_numThreads & i*blklen<rlen; i++ )
					tasks2.add(new SortRowsTask(dest, i*blklen, Math.min((i+1)*blklen, (int)rlen)));
				pool.invokeAll(tasks2);
			}
			
			pool.shutdown();
		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}
	
	/**
	 * 
	 * 
	 */
	public static class ReadTask implements Callable<Long> 
	{
		private InputSplit _split = null;
		private boolean _sparse = false;
		private TextInputFormat _informat = null;
		private JobConf _job = null;
		private MatrixBlock _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		private boolean _matrixMarket = false;
		
		public ReadTask( InputSplit split, TextInputFormat informat, JobConf job, MatrixBlock dest, long rlen, long clen, boolean matrixMarket )
		{
			_split = split;
			_sparse = dest.isInSparseFormat();
			_informat = informat;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_matrixMarket = matrixMarket;
		}

		@Override
		public Long call() throws Exception 
		{
			long lnnz = 0; //aggregate block nnz
			
			//writables for reuse during read
			LongWritable key = new LongWritable();
			Text value = new Text();
			
			//required for error handling
			int row = -1; 
			int col = -1; 
			
			FastStringTokenizer st = new FastStringTokenizer(' ');
			RecordReader<LongWritable,Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			
			try
			{			
				
				// Read the header lines, if reading from a matrixMarket file
				if ( _matrixMarket ) {					
					// skip until end-of-comments (%% or %)
					boolean foundComment = false;
					while( reader.next(key, value) && value.toString().charAt(0) == '%'  ) {
						//do nothing just skip comments
						foundComment = true;
					}
					
					//process current value (otherwise ignore following meta data)
					if( !foundComment ) {
						st.reset( value.toString() ); //reinit tokenizer
						row = st.nextInt()-1;
						col = st.nextInt()-1;
						double lvalue = st.nextDoubleForParallel();
						synchronized( _dest ){ //sparse requires lock	
							_dest.appendValue(row, col, lvalue);
							lnnz++;
						}
					}
				}

				if( _sparse ) //SPARSE<-value
				{
					CellBuffer buff = new CellBuffer();
					
					while( reader.next(key, value) ) {
						st.reset( value.toString() ); //reinit tokenizer
						row = st.nextInt() - 1;
						col = st.nextInt() - 1;
						double lvalue = st.nextDoubleForParallel();
						
						buff.addCell(row, col, lvalue);
						//capacity buffer flush on demand
						if( buff.size()>=CellBuffer.CAPACITY ) 
							synchronized( _dest ){ //sparse requires lock
								lnnz += buff.size();
								buff.flushCellBufferToMatrixBlock(_dest);
							}
					}
					
					//final buffer flush 
					synchronized( _dest ){ //sparse requires lock
						lnnz += buff.size();
						buff.flushCellBufferToMatrixBlock(_dest);
					}
				} 
				else //DENSE<-value
				{
					while( reader.next(key, value) ) {
						st.reset( value.toString() ); //reinit tokenizer
						row = st.nextInt()-1;
						col = st.nextInt()-1;
						double lvalue = st.nextDoubleForParallel();
						_dest.setValueDenseUnsafe( row, col, lvalue );
						lnnz += (lvalue!=0) ? 1 : 0;
					}
				}
			}
			catch(Exception ex)	{
				//post-mortem error handling and bounds checking
				if( row < 0 || row + 1 > _rlen || col < 0 || col + 1 > _clen )
					throw new RuntimeException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
							  "out of overall matrix range [1:"+_rlen+",1:"+_clen+"]. ", ex);
				else
					throw new RuntimeException("Unable to read matrix in text cell format. ", ex);
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			
			return lnnz;
		}
	}
	
	/**
	 * Useful class for buffering unordered cells before locking target onces and
	 * appending all buffered cells.
	 * 
	 */
	public static class CellBuffer
	{
		public static final int CAPACITY = 100*1024; //100K elements 
		
		private int[] _rlen;
		private int[] _clen;
		private double[] _vals;
		private int _pos;
		
		public CellBuffer( ) {
			_rlen = new int[CAPACITY];
			_clen = new int[CAPACITY];
			_vals = new double[CAPACITY];
			_pos = -1;
		}
		
		public void addCell(int rlen, int clen, double val) {
			if( val==0 ) return;
			_pos++;
			_rlen[_pos] = rlen;
			_clen[_pos] = clen;
			_vals[_pos] = val;
		}
		
		public void flushCellBufferToMatrixBlock( MatrixBlock dest ) {
			for( int i=0; i<=_pos; i++ )
				dest.appendValue(_rlen[i], _clen[i], _vals[i]);
			reset();
		}
		
		public int size() {
			return _pos+1;
		}
		
		public void reset() {
			_pos = -1;
		}
	}
}
