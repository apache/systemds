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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.MapReduceTool;


/*
 * This write uses fixed size blocks with block-encoded keys.
 * 
 * 
 */

public class FrameWriterBinaryBlock extends FrameWriter
{
	/**
	 * @param src
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	@Override
	public void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeBinaryBlockFrameToHDFS(path, job, src, rlen, clen);
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @return 
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	@SuppressWarnings("deprecation")
	protected void writeBinaryBlockFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		FileSystem fs = FileSystem.get(job);
		int brlen = ConfigurationManager.getBlocksize();
		int bclen = ConfigurationManager.getBlocksize();

		// 1) create sequence file writer 
		SequenceFile.Writer writer = null;
		writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, FrameBlock.class);
		
		try
		{
			// 2) bound check for src block
			if( src.getNumRows() > rlen || src.getNumColumns() > clen )
			{
				throw new IOException("Frame block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
						              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
			}
		
			//3) reblock and write
			MatrixIndexes indexes = new MatrixIndexes();

			if( rlen <= brlen && clen <= bclen ) //opt for single block
			{
				//directly write single block
				indexes.setIndexes(1, 1);
				writer.append(indexes, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				FrameBlock[] blocks = createFrameBlocksForReuse(src.getSchema(), src.getColumnNames(), rlen);  
				
				//create and write subblocks of frame
				for(int blockRow = 0; blockRow < (int)Math.ceil(src.getNumRows()/(double)brlen); blockRow++)
					for(int blockCol = 0; blockCol < (int)Math.ceil(src.getNumColumns()/(double)bclen); blockCol++)
					{
						int maxRow = (blockRow*brlen + brlen < src.getNumRows()) ? brlen : src.getNumRows() - blockRow*brlen;
						int maxCol = (blockCol*bclen + bclen < src.getNumColumns()) ? bclen : src.getNumColumns() - blockCol*bclen;
				
						int row_offset = blockRow*brlen;
						int col_offset = blockCol*bclen;
						
						//get reuse frame block
						FrameBlock block = getFrameBlockForReuse(blocks);
	
						//copy subpart to block
						src.sliceOperations( row_offset, row_offset+maxRow-1, 
								             col_offset, col_offset+maxCol-1, block );
						
						//append block to sequence file
						indexes.setIndexes(blockRow+1, blockCol+1);
						writer.append(indexes, block);
					}
			}
		
		}
		finally
		{
			IOUtilFunctions.closeSilently(writer);
		}
	}
}
