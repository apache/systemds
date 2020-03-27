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

package org.tugraz.sysds.runtime.io;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.data.DenseBlock;
import org.tugraz.sysds.runtime.data.SparseBlock;
import org.tugraz.sysds.runtime.matrix.data.IJV;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.CommonThreadPool;
import org.tugraz.sysds.runtime.util.FastStringTokenizer;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.runtime.util.UtilFunctions;

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
 *    scenario is extremely unlikely (num threads &gt; num lines if 1 comment line) and hence ignored 
 *    similar to our parallel MR setting (but there we have a 128MB guarantee).     
 * 3) However, we use MIN_FILESIZE_MM (8KB) to give guarantees for the common case of small headers
 *    in order the issue described in (2).
 * 
 */
public class ReaderTextCellParallel extends ReaderTextCell
{
	private static final long MIN_FILESIZE_MM = 8L * 1024; //8KB
	
	private int _numThreads = 1;
	
	public ReaderTextCellParallel(InputInfo info) {
		super(info, false);
		_numThreads = OptimizerUtils.getParallelTextReadParallelism();
	}

	@Override
	protected void readTextCellMatrixFromHDFS( Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int blen )
		throws IOException
	{
		int par = _numThreads;
		
		FileInputFormat.addInputPath(job, path);
		TextInputFormat informat = new TextInputFormat();
		informat.configure(job);
		
		//check for min file size for matrix market (adjust num splits if necessary)
		if( _isMMFile ){
			long len = HDFSTool.getFilesizeOnHDFS(path);
			par = ( len < MIN_FILESIZE_MM ) ? 1: par; 
		}
		
		try 
		{
			ExecutorService pool = CommonThreadPool.get(par);
			InputSplit[] splits = informat.getSplits(job, par);
			
			//count nnz per row for sparse preallocation
			if( dest.isInSparseFormat() ) {
				int[] rNnz = new int[(int)rlen];
				boolean isSymmetric = _isMMFile && _mmProps.isSymmetric();
				List<CountNnzTask> tasks = Arrays.stream(splits)
					.map(s ->new CountNnzTask(s, informat, job, rNnz, isSymmetric))
					.collect(Collectors.toList());
				List<Future<Void>> rt1 = pool.invokeAll(tasks);
				for( Future<Void> task : rt1 )
					task.get();
				SparseBlock sblock = dest.allocateBlock().getSparseBlock();
				for( int i=0; i<rlen; i++ )
					if( rNnz[i] > 0 )
						sblock.allocate(i, UtilFunctions.roundToNext(rNnz[i], 4));
			}
			
			//create and execute read tasks for all splits
			List<ReadTask> tasks = Arrays.stream(splits)
				.map(s ->new ReadTask(s, informat, job, dest, rlen, clen, _isMMFile, _mmProps))
				.collect(Collectors.toList());
			List<Future<Long>> rt2 = pool.invokeAll(tasks);
			
			//check for exceptions and aggregate nnz
			long lnnz = 0;
			for( Future<Long> task : rt2 )
				lnnz += task.get();
			
			//post-processing
			dest.setNonZeros( lnnz );
			if( dest.isInSparseFormat() ) 
				sortSparseRowsParallel(dest, rlen, _numThreads, pool);
			
			pool.shutdown();
		} 
		catch (Exception e) {
			throw new IOException("Threadpool issue, while parallel read.", e);
		}
	}

	public static class ReadTask implements Callable<Long> 
	{
		private final InputSplit _split;
		private final boolean _sparse;
		private final TextInputFormat _informat;
		private final JobConf _job;
		private final MatrixBlock _dest;
		private final long _rlen;
		private final long _clen;
		private final boolean _matrixMarket;
		private final FileFormatPropertiesMM _mmProps;
		
		public ReadTask( InputSplit split, TextInputFormat informat, JobConf job, MatrixBlock dest, long rlen, long clen, boolean mm, FileFormatPropertiesMM mmProps ) {
			_split = split;
			_sparse = dest.isInSparseFormat();
			_informat = informat;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_matrixMarket = mm;
			_mmProps = mmProps;
		}

		@Override
		public Long call() throws Exception 
		{
			long lnnz = 0; //aggregate block nnz
			
			//writables for reuse during read
			LongWritable key = new LongWritable();
			Text value = new Text();
			IJV cell = new IJV();
			
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
						cell = parseCell(value.toString(), st, cell, _mmProps);
						synchronized( _dest ){ //sparse requires lock
							lnnz += appendCell(cell, _dest, _mmProps);
						}
					}
				}

				if( _sparse ) { //SPARSE<-value
					CellBuffer buff = new CellBuffer();
					while( reader.next(key, value) ) {
						cell = parseCell(value.toString(), st, cell, _mmProps);
						buff.addCell(cell.getI(), cell.getJ(), cell.getV());
						if( _mmProps != null && _mmProps.isSymmetric() && !cell.onDiag() )
							buff.addCell(cell.getJ(), cell.getI(), cell.getV());
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
				else { //DENSE<-value
					DenseBlock a = _dest.getDenseBlock();
					while( reader.next(key, value) ) {
						cell = parseCell(value.toString(), st, cell, _mmProps);
						lnnz += appendCell(cell, a, _mmProps);
					}
				}
			}
			catch(Exception ex) {
				//post-mortem error handling and bounds checking
				if( cell.getI() < 0 || cell.getI() + 1 > _rlen || cell.getJ() < 0 || cell.getJ() + 1 > _clen )
					throw new RuntimeException("Matrix cell ["+(cell.getI()+1)+","+(cell.getJ()+1)+"] " +
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
	
	public static class CountNnzTask implements Callable<Void> {
		private final InputSplit _split;
		private final TextInputFormat _informat;
		private final JobConf _job;
		private final int[] _rNnz;
		private final boolean _isSymmetric;
		
		public CountNnzTask( InputSplit split, TextInputFormat informat, JobConf job, int[] rNnz, boolean isSymmetric ) {
			_split = split;
			_informat = informat;
			_job = job;
			_rNnz = rNnz;
			_isSymmetric = isSymmetric;
		}

		@Override
		public Void call() throws Exception {
			LongWritable key = new LongWritable();
			Text value = new Text();
			FastStringTokenizer st = new FastStringTokenizer(' ');
			
			RecordReader<LongWritable,Text> reader = _informat.getRecordReader(_split, _job, Reporter.NULL);
			try {
				//counting without locking as conflicts unlikely
				while( reader.next(key, value) ) {
					if( value.toString().charAt(0) == '%' )
						continue;
					st.reset( value.toString() );
					_rNnz[(int)st.nextLong()-1] ++;
					if( _isSymmetric )
						_rNnz[(int)st.nextLong()-1] ++;
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			return null;
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
