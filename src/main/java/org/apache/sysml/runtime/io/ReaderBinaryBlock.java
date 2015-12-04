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
import java.util.Collection;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

public class ReaderBinaryBlock extends MatrixReader
{
	protected boolean _localFS = false;
	
	public ReaderBinaryBlock( boolean localFS )
	{
		_localFS = localFS;
	}
	
	public void setLocalFS(boolean flag) {
		_localFS = flag;
	}
	
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, false, false);
		
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
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param estnnz
	 * @return
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	public ArrayList<IndexedMatrixValue> readIndexedMatrixBlocksFromHDFS(String fname, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block collection
		ArrayList<IndexedMatrixValue> ret = new ArrayList<IndexedMatrixValue>();
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = _localFS ? FileSystem.getLocal(job) : FileSystem.get(job);
		Path path = new Path( (_localFS ? "file:///" : "") + fname); 
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryBlockMatrixBlocksFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
		return ret;
	}


	
	/**
	 * Note: For efficiency, we directly use SequenceFile.Reader instead of SequenceFileInputFormat-
	 * InputSplits-RecordReader (SequenceFileRecordReader). First, this has no drawbacks since the
	 * SequenceFileRecordReader internally uses SequenceFile.Reader as well. Second, it is 
	 * advantageous if the actual sequence files are larger than the file splits created by   
	 * informat.getSplits (which is usually aligned to the HDFS block size) because then there is 
	 * overhead for finding the actual split between our 1k-1k blocks. This case happens
	 * if the read matrix was create by CP or when jobs directly write to large output files 
	 * (e.g., parfor matrix partitioning).
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
	@SuppressWarnings("deprecation")
	private static void readBinaryBlockMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException, DMLRuntimeException
	{
		boolean sparse = dest.isInSparseFormat();
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
		long lnnz = 0; //aggregate block nnz
		
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
			try
			{
				//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
				while( reader.next(key, value) )
				{	
					//empty block filter (skip entire block)
					if( value.isEmptyBlock(false) )
						continue;
					
					int row_offset = (int)(key.getRowIndex()-1)*brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*bclen;
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen || col_offset + cols<0 || col_offset + cols > clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
					}
			
					//copy block to result
					if( sparse )
					{
						//note: append requires final sort (but prevents repeated shifting)
						dest.appendToSparse(value, row_offset, col_offset);
					} 
					else
					{
						dest.copy( row_offset, row_offset+rows-1, 
								   col_offset, col_offset+cols-1,
								   value, false );
					}
					
					//maintain nnz as aggregate of block nnz
					lnnz += value.getNonZeros();
				}
			}
			finally
			{
				IOUtilFunctions.closeSilently(reader);
			}
		}
		
		//post-processing
		dest.setNonZeros( lnnz );
		if( sparse && clen>bclen ){
			//no need to sort if 1 column block since always sorted
			dest.sortSparseRows();
		}
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
	 */
	@SuppressWarnings("deprecation")
	private void readBinaryBlockMatrixBlocksFromHDFS( Path path, JobConf job, FileSystem fs, Collection<IndexedMatrixValue> dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		MatrixIndexes key = new MatrixIndexes(); 
		MatrixBlock value = new MatrixBlock();
			
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
			try
			{
				while( reader.next(key, value) )
				{	
					int row_offset = (int)(key.getRowIndex()-1)*brlen;
					int col_offset = (int)(key.getColumnIndex()-1)*bclen;
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen || col_offset + cols<0 || col_offset + cols > clen )
					{
						throw new IOException("Matrix block ["+(row_offset+1)+":"+(row_offset+rows)+","+(col_offset+1)+":"+(col_offset+cols)+"] " +
								              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
					}
			
					//copy block to result
					dest.add(new IndexedMatrixValue(new MatrixIndexes(key), new MatrixBlock(value)));
				}
			}
			finally
			{
				IOUtilFunctions.closeSilently(reader);
			}
		}
	}
}
