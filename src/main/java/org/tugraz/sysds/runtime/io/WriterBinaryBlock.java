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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.mapred.MRConfigurationNames;
import org.tugraz.sysds.runtime.matrix.mapred.MRJobConfiguration;
import org.tugraz.sysds.runtime.util.HDFSTool;

public class WriterBinaryBlock extends MatrixWriter
{
	protected int _replication = -1;
	
	public WriterBinaryBlock( int replication ) {
		_replication  = replication;
	}

	@Override
	public final void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag) 
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS( fname );

		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		//core write sequential/parallel
		if( diag )
			writeDiagBinaryBlockMatrixToHDFS(path, job, fs, src, rlen, clen, blen);
		else
			writeBinaryBlockMatrixToHDFS(path, job, fs, src, rlen, clen, blen);

		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	@Override
	@SuppressWarnings("deprecation")
	public final void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen) 
		throws IOException, DMLRuntimeException 
	{
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		SequenceFile.Writer writer = null;
		try {
			writer = new SequenceFile.Writer(fs, job, path,
				MatrixIndexes.class, MatrixBlock.class);
			MatrixIndexes index = new MatrixIndexes(1, 1);
			MatrixBlock block = new MatrixBlock(
				(int)Math.max(Math.min(rlen, blen),1),
				(int)Math.max(Math.min(clen, blen),1), true);
			writer.append(index, block);
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
		
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	protected void writeBinaryBlockMatrixToHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock src, long rlen, long clen, int blen )
		throws IOException, DMLRuntimeException
	{
		//sequential write 
		writeBinaryBlockMatrixToSequenceFile(path, job, fs, src, blen, 0, (int)rlen);
	}

	@SuppressWarnings("deprecation")
	protected final void writeBinaryBlockMatrixToSequenceFile( Path path, JobConf job, FileSystem fs, MatrixBlock src, int blen, int rl, int ru ) 
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		int rlen = src.getNumRows();
		int clen = src.getNumColumns();
		
		// 1) create sequence file writer, with right replication factor 
		// (config via MRConfigurationNames.DFS_REPLICATION not possible since sequence file internally calls fs.getDefaultReplication())
		SequenceFile.Writer writer = null;
		if( _replication > 0 ) //if replication specified (otherwise default)
		{
			//copy of SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class), except for replication
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class, job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096),
					                         (short)_replication, fs.getDefaultBlockSize(), null, new SequenceFile.Metadata());	
		}
		else	
		{
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
		}
		
		try
		{
			// 2) bound check for src block
			if( src.getNumRows() > rlen || src.getNumColumns() > clen )
			{
				throw new IOException("Matrix block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
						              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
		
			//3) reblock and write
			MatrixIndexes indexes = new MatrixIndexes();

			if( rlen <= blen && clen <= blen && rl == 0 ) //opt for single block
			{
				//directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, blen, sparse, src.getNonZeros());
				
				//create and write subblocks of matrix
				for(int blockRow = rl/blen; blockRow < (int)Math.ceil(ru/(double)blen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)blen); blockCol++)
					{
						int maxRow = (blockRow*blen + blen < src.getNumRows()) ? blen : src.getNumRows() - blockRow*blen;
						int maxCol = (blockCol*blen + blen < src.getNumColumns()) ? blen : src.getNumColumns() - blockCol*blen;
				
						int row_offset = blockRow*blen;
						int col_offset = blockCol*blen;
						
						//get reuse matrix block
						MatrixBlock block = getMatrixBlockForReuse(blocks, maxRow, maxCol, blen);
	
						//copy submatrix to block
						src.slice( row_offset, row_offset+maxRow-1, 
								             col_offset, col_offset+maxCol-1, block );
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
							
						//reset block for later reuse
						block.reset();
					}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}

	@SuppressWarnings("deprecation")
	protected final void writeDiagBinaryBlockMatrixToHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock src, long rlen, long clen, int blen ) 
		throws IOException, DMLRuntimeException
	{
		boolean sparse = src.isInSparseFormat();
		
		// 1) create sequence file writer, with right replication factor 
		// (config via MRConfigurationNames.DFS_REPLICATION not possible since sequence file internally calls fs.getDefaultReplication())
		SequenceFile.Writer writer = null;
		if( _replication > 0 ) //if replication specified (otherwise default)
		{
			//copy of SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class), except for replication
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class, job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096),
					                         (short)_replication, fs.getDefaultBlockSize(), null, new SequenceFile.Metadata());	
		}
		else	
		{
			writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixBlock.class);
		}
		
		try
		{
			// 2) bound check for src block
			if( src.getNumRows() > rlen || src.getNumColumns() > clen )
			{
				throw new IOException("Matrix block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
						              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
		
			//3) reblock and write
			MatrixIndexes indexes = new MatrixIndexes();

			if( rlen <= blen && clen <= blen ) //opt for single block
			{
				//directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				MatrixBlock[] blocks = createMatrixBlocksForReuse(rlen, clen, blen, sparse, src.getNonZeros());
				MatrixBlock emptyBlock = new MatrixBlock();
					
				//create and write subblocks of matrix
				for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)blen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)blen); blockCol++)
					{
						int maxRow = (blockRow*blen + blen < src.getNumRows()) ? blen : src.getNumRows() - blockRow*blen;
						int maxCol = (blockCol*blen + blen < src.getNumColumns()) ? blen : src.getNumColumns() - blockCol*blen;
						MatrixBlock block = null;
						
						if( blockRow==blockCol ) //block on diagonal
						{	
							int row_offset = blockRow*blen;
							int col_offset = blockCol*blen;
							
							//get reuse matrix block
							block = getMatrixBlockForReuse(blocks, maxRow, maxCol, blen);
		
							//copy submatrix to block
							src.slice( row_offset, row_offset+maxRow-1, 
								col_offset, col_offset+maxCol-1, block );
						}
						else //empty block (not on diagonal)
						{
							block = emptyBlock;
							block.reset(maxRow, maxCol);
						}
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
						
						//reset block for later reuse
						if( blockRow!=blockCol )
							block.reset();
					}
			}				
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}
	}
}
