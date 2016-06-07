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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;


/**
 * Single-threaded frame binary block writer.
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
	public final void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );

		//bound check for src block
		if( src.getNumRows() > rlen || src.getNumColumns() > clen ) {
			throw new IOException("Frame block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
					              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
		}
		
		//write binary block to hdfs (sequential/parallel)
		writeBinaryBlockFrameToHDFS( path, job, src, rlen, clen );
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	protected void writeBinaryBlockFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen )
			throws IOException, DMLRuntimeException
	{
		FileSystem fs = FileSystem.get(job);
		int blen = ConfigurationManager.getBlocksize();
		
		//sequential write to single file
		writeBinaryBlockFrameToSequenceFile(path, job, fs, src, blen, 0, (int)rlen);		
	}

	/**
	 * Internal primitive to write a block-aligned row range of a frame to a single sequence file, 
	 * which is used for both single- and multi-threaded writers (for consistency). 
	 *  
	 * @param path
	 * @param job
	 * @param fs
	 * @param src
	 * @param blen
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	@SuppressWarnings("deprecation")
	protected final void writeBinaryBlockFrameToSequenceFile( Path path, JobConf job, FileSystem fs, FrameBlock src, int blen, int rl, int ru ) 
		throws DMLRuntimeException, IOException
	{
		//1) create sequence file writer 
		SequenceFile.Writer writer = null;
		writer = new SequenceFile.Writer(fs, job, path, LongWritable.class, FrameBlock.class);
		
		try
		{
			//2) reblock and write
			LongWritable index = new LongWritable();

			if( src.getNumRows() <= blen ) //opt for single block
			{
				//directly write single block
				index.set(1);
				writer.append(index, src);
			}
			else //general case
			{
				//initialize blocks for reuse (at most 4 different blocks required)
				FrameBlock[] blocks = createFrameBlocksForReuse(src.getSchema(), src.getColumnNames(), src.getNumRows());  
				
				//create and write subblocks of frame
				for(int bi = rl; bi < ru; bi += blen) {
					int len = Math.min(blen,  src.getNumRows()-bi);
					
					//get reuse frame block and copy subpart to block
					FrameBlock block = getFrameBlockForReuse(blocks);
					src.sliceOperations( bi, bi+len-1, 0, src.getNumColumns()-1, block );
					
					//append block to sequence file
					index.set(bi+1);
					writer.append(index, block);
				}
			}
		}
		finally {
			IOUtilFunctions.closeSilently(writer);
		}		
	}
}
