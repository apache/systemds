/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;


public class ReaderBinaryBlockParallel extends ReaderBinaryBlock 
{	
	private static int _numThreads = 1;
	
	public ReaderBinaryBlockParallel( boolean localFS )
	{
		super(localFS);
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{	
		//allocate output matrix block (incl block allocation for parallel)
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true, true);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = _localFS ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( (_localFS ? "file:///" : "") + fname); 
				
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryBlockMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
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
	 * @param fs 
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws DMLRuntimeException 
	 */
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, DMLRuntimeException
	{			
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		try 
		{
			//create read tasks for all files
			ExecutorService pool = Executors.newFixedThreadPool(_numThreads);
			ArrayList<ReadFileTask> tasks = new ArrayList<ReadFileTask>();
			for( Path lpath : getSequenceFilePaths(fs, path) ){
				ReadFileTask t = new ReadFileTask(lpath, job, fs, dest, rlen, clen, brlen, bclen);
				tasks.add(t);
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
			
			//check for exceptions and aggregate nnz
			long lnnz = 0;
			for( Future<Object> task : rt )
				lnnz += (Long)task.get();
			
			//post-processing
			dest.setNonZeros( lnnz );
			if( dest.isInSparseFormat() && clen>bclen ){
				//no need to sort if 1 column block since always sorted
				dest.sortSparseRows();
			}			
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel read of binary block input.", e);
		}
	}

	/**
	 * 
	 */
	private static class ReadFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private MatrixBlock _dest = null;
		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		
		public ReadFileTask(Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen)
		{
			_path = path;
			_fs = fs;
			_job = job;
			_dest = dest;
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
		}

		@Override
		@SuppressWarnings({ "deprecation", "resource" })
		public Object call() throws Exception 
		{
			boolean sparse = _dest.isInSparseFormat();
			MatrixIndexes key = new MatrixIndexes(); 
			MatrixBlock value = new MatrixBlock();
			long lnnz = 0; //aggregate block nnz
			
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(_fs,_path,_job);
			
			try
			{
				//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
				while( reader.next(key, value) )
				{	
					//empty block filter (skip entire block)
					if( value.isEmptyBlock(false) )
						continue;
					
					int row_offset = (int)(key.getRowIndex()-1)*_brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*_bclen;
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > _rlen || col_offset + cols<0 || col_offset + cols > _clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+_rlen+",1:"+_clen+"].");
					}
			
					//copy block to result
					if( sparse )
					{
						//note: append requires final sort
						if (cols < _clen ) {
							synchronized( _dest ){ //sparse requires lock, when matrix is wider than one block
								_dest.appendToSparse(value, row_offset, col_offset);
							}
						}
						else
							_dest.appendToSparse(value, row_offset, col_offset);
					} 
					else
					{
						_dest.copy( row_offset, row_offset+rows-1, 
								   col_offset, col_offset+cols-1,
								   value, false );
					}
					
					//aggregate nnz
					lnnz += value.getNonZeros();
				}
			}
			finally
			{
				IOUtilFunctions.closeSilently(reader);
			}
			
			return lnnz;
		}
	}
}
