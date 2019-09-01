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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.data.SparseBlock;
import org.tugraz.sysds.runtime.data.SparseBlockMCSR;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.mapred.MRJobConfiguration;
import org.tugraz.sysds.runtime.util.CommonThreadPool;


public class ReaderBinaryBlockParallel extends ReaderBinaryBlock 
{	
	private static int _numThreads = 1;
	
	public ReaderBinaryBlockParallel( boolean localFS )
	{
		super(localFS);
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//early abort for known empty matrices (e.g., remote parfor result vars)
		if( RETURN_EMPTY_NNZ0 && estnnz == 0 )
			return new MatrixBlock((int)rlen, (int)clen, true);
		
		//allocate output matrix block (incl block allocation for parallel)
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, blen, estnnz, true, true);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		Path path = new Path( (_localFS ? "file:///" : "") + fname); 
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
		
		//core read
		int numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
		long numBlocks = (long)Math.ceil((double)rlen / blen);
		readBinaryBlockMatrixFromHDFS(path, job, fs, ret,
			rlen, clen, blen, numThreads<=numBlocks);
		
		//finally check if change of sparse/dense block representation required
		if( !AGGREGATE_BLOCK_NNZ )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest,
			long rlen, long clen, int blen, boolean syncBlock )
		throws IOException, DMLRuntimeException
	{
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		try 
		{
			//create read tasks for all files
			ExecutorService pool = CommonThreadPool.get(_numThreads);
			ArrayList<ReadFileTask> tasks = new ArrayList<>();
			for( Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path) ){
				ReadFileTask t = new ReadFileTask(lpath, job, dest, rlen, clen, blen, syncBlock);
				tasks.add(t);
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);
			
			//check for exceptions and aggregate nnz
			long lnnz = 0;
			for( Future<Object> task : rt )
				lnnz += (Long)task.get();
			
			//post-processing
			dest.setNonZeros( lnnz );
			if( dest.isInSparseFormat() && clen>blen ) 
				sortSparseRowsParallel(dest, rlen, _numThreads, pool);
			
			pool.shutdown();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of binary block input.", e);
		}
	}

	private static class ReadFileTask implements Callable<Object> 
	{
		private final Path _path;
		private final JobConf _job;
		private final MatrixBlock _dest;
		private final long _rlen, _clen;
		private final int _blen;
		private final boolean _syncBlocks;
		
		public ReadFileTask(Path path, JobConf job, MatrixBlock dest, long rlen, long clen, int blen, boolean syncBlocks) {
			_path = path;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_blen = blen;
			_syncBlocks = syncBlocks;
		}

		@Override
		public Object call() throws Exception 
		{
			boolean sparse = _dest.isInSparseFormat();
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = getReuseBlock(_blen, sparse);
			long lnnz = 0; //aggregate block nnz
			
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile
				.Reader(_job, SequenceFile.Reader.file(_path));
			
			try
			{
				//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
				while( reader.next(key, value) )
				{	
					//empty block filter (skip entire block)
					if( value.isEmptyBlock(false) )
						continue;
					
					int row_offset = (int)(key.getRowIndex()-1)*_blen;
					int col_offset = (int)(key.getColumnIndex()-1)*_blen;
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > _rlen 
						|| col_offset + cols<0 || col_offset + cols > _clen ) {
						throw new IOException("Matrix block ["+(row_offset+1)+":"
							+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
							"out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
					}
			
					//copy block to result
					if( sparse )
					{
						//note: append requires final sort
						if (cols < _clen ) {
							//sparse requires lock, when matrix is wider than one block
							//(fine-grained locking of block rows instead of the entire matrix)
							//NOTE: fine-grained locking depends on MCSR SparseRow objects 
							SparseBlock sblock = _dest.getSparseBlock();
							if( sblock instanceof SparseBlockMCSR && sblock.get(row_offset) != null ) {
								if( _syncBlocks ) {
									synchronized( sblock.get(row_offset) ){ 
										_dest.appendToSparse(value, row_offset, col_offset);
									}
								}
								else {
									for( int i=0; i<rows; i++ ) 
										synchronized( sblock.get(row_offset+i) ) {
											_dest.appendRowToSparse(sblock, value, i, row_offset, col_offset, true);
										}
								}
							}
							else {
								synchronized( _dest ){ 
									_dest.appendToSparse(value, row_offset, col_offset);
								}
							}
						}
						else { //quickpath (no synchronization)
							_dest.appendToSparse(value, row_offset, col_offset);
						}
					} 
					else {
						_dest.copy( row_offset, row_offset+rows-1, 
							col_offset, col_offset+cols-1, value, false );
					}
					
					//aggregate nnz
					lnnz += value.getNonZeros();
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
			
			return lnnz;
		}
	}
}
