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

package com.ibm.bi.dml.runtime.io;

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

import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterBinaryBlockParallel extends WriterBinaryBlock
{
	public WriterBinaryBlockParallel( int replication )
	{
		super(replication);
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	@Override
	protected void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen, int replication )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException
	{
		//estimate output size and number of output blocks (min 1)
		int numPartFiles = (int)(OptimizerUtils.estimatePartitionedSizeExactSparsity(rlen, clen, brlen, bclen, src.getNonZeros()) 
						   / InfrastructureAnalyzer.getHDFSBlockSize());
		numPartFiles = Math.max(numPartFiles, 1);
		
		//determine degree of parallelism
		int numThreads = OptimizerUtils.getParallelBinaryWriteParallelism();
		numThreads = Math.min(numThreads, numPartFiles);
		
		//fall back to sequential write if dop is 1 (e.g., <128MB) in order to create single file
		if( numThreads <= 1 ) {
			super.writeBinaryBlockMatrixToHDFS(path, job, src, rlen, clen, brlen, bclen, replication);
			return;
		}
			
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );

		//create directory for concurrent tasks
		MapReduceTool.createDirIfNotExistOnHDFS(path.toString(), DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		FileSystem fs = FileSystem.get(job);
		
		//create and execute write tasks
		try 
		{
			ExecutorService pool = Executors.newFixedThreadPool(numThreads);
			ArrayList<WriteFileTask> tasks = new ArrayList<WriteFileTask>();
			int blklen = (int)Math.ceil((double)rlen / brlen / numThreads) * brlen;
			for(int i=0; i<numThreads & i*blklen<rlen; i++) {
				Path newPath = new Path(path, String.format("0-m-%05d",i));
				tasks.add(new WriteFileTask(newPath, job, fs, src, i*blklen, Math.min((i+1)*blklen, rlen), brlen, bclen, _replication));
			}

			//wait until all tasks have been executed
			List<Future<Object>> rt = pool.invokeAll(tasks);	
			pool.shutdown();
			
			//check for exceptions 
			for( Future<Object> task : rt )
				task.get();
		} 
		catch (Exception e) {
			throw new IOException("Failed parallel write of binary block input.", e);
		}
	}

	/**
	 * 
	 */
	private static class WriteFileTask implements Callable<Object> 
	{
		private Path _path = null;
		private JobConf _job = null;
		private FileSystem _fs = null;
		private MatrixBlock _src = null;
		private long _rl = -1;
		private long _ru = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		private int _replication = 1;
		
		public WriteFileTask(Path path, JobConf job, FileSystem fs, MatrixBlock src, long rl, long ru, int brlen, int bclen, int rep)
		{
			_path = path;
			_fs = fs;
			_job = job;
			_src = src;
			_rl = rl;
			_ru = ru;
			_brlen = brlen;
			_bclen = bclen;
			_replication = rep;
		}
	
		@Override
		@SuppressWarnings("deprecation")
		public Object call() throws Exception 
		{
			// 1) create sequence file writer, with right replication factor 
			// (config via 'dfs.replication' not possible since sequence file internally calls fs.getDefaultReplication())
			SequenceFile.Writer writer = null;
			if( _replication > 0 ) //if replication specified (otherwise default)
			{
				//copy of SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class), except for replication
				writer = new SequenceFile.Writer(_fs, _job, _path, MatrixIndexes.class, MatrixBlock.class, _job.getInt("io.file.buffer.size", 4096),  
						                         (short)_replication, _fs.getDefaultBlockSize(), null, new SequenceFile.Metadata());	
			}
			else	
			{
				writer = new SequenceFile.Writer(_fs, _job, _path, MatrixIndexes.class, MatrixBlock.class);
			}
			
			try
			{
				//3) reblock and write
				MatrixIndexes indexes = new MatrixIndexes();

				//initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(_src.getNumRows(), _src.getNumColumns(),
						_brlen, _bclen, _src.isInSparseFormat(), _src.getNonZeros());  
					
				//create and write subblocks of matrix
				for(int blockRow = (int)_rl/_brlen; blockRow < (int)Math.ceil(_ru/(double)_brlen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(_src.getNumColumns()/(double)_bclen); blockCol++)
					{
						int maxRow = (blockRow*_brlen + _brlen < _src.getNumRows()) ? _brlen : _src.getNumRows() - blockRow*_brlen;
						int maxCol = (blockCol*_bclen + _bclen < _src.getNumColumns()) ? _bclen : _src.getNumColumns() - blockCol*_bclen;
				
						int row_offset = blockRow*_brlen;
						int col_offset = blockCol*_bclen;
						
						//get reuse matrix block
						MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, _brlen, _bclen);
	
						//copy submatrix to block
						_src.sliceOperations( row_offset, row_offset+maxRow-1, 
								             col_offset, col_offset+maxCol-1, block );
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
							
						//reset block for later reuse
						block.reset();
					}
			}
			finally
			{
				IOUtilFunctions.closeSilently(writer);
			}	
			
			return null;
		}
	}
}
