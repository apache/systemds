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

package org.apache.sysds.runtime.io;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.HDFSTool;


/**
 * Single-threaded frame binary block writer.
 * 
 */
public class FrameWriterBinaryBlock extends FrameWriter {
	// protected static final Log LOG = LogFactory.getLog(FrameWriterBinaryBlock.class.getName());

	@Override
	public final void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException {
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		HDFSTool.deleteFileIfExistOnHDFS( fname );

		//bound check for src block
		if( src.getNumRows() > rlen || src.getNumColumns() > clen ) {
			throw new IOException("Frame block [1:"+src.getNumRows()+",1:"+src.getNumColumns()+"] " +
					              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
		}
		
		//write binary block to hdfs (sequential/parallel)
		writeBinaryBlockFrameToHDFS( path, job, src, rlen, clen );
	}

	protected void writeBinaryBlockFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen )
			throws IOException, DMLRuntimeException
	{
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		int blen = ConfigurationManager.getBlocksize();
		
		//sequential write to single file
		writeBinaryBlockFrameToSequenceFile(path, job, fs, src, blen, 0, (int)rlen);
		IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
	}

	/**
	 * Internal primitive to write a block-aligned row range of a frame to a single sequence file, 
	 * which is used for both single- and multi-threaded writers (for consistency). 
	 * 
	 * @param path file path
	 * @param job job configuration
	 * @param fs file system
	 * @param src frame block
	 * @param blen block length
	 * @param rl lower row
	 * @param ru upper row
	 * @throws IOException if IOException occurs
	 */
	@SuppressWarnings("deprecation")
	protected static void writeBinaryBlockFrameToSequenceFile( Path path, JobConf job, FileSystem fs, FrameBlock src, int blen, int rl, int ru ) 
		throws IOException
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
					
					//get reuse frame block and copy subpart to block (incl meta on first)
					FrameBlock block = getFrameBlockForReuse(blocks);
					src.slice( bi, bi+len-1, 0, src.getNumColumns()-1, block );
					if( bi==0 ) //first block
						block.setColumnMetadata(src.getColumnMetadata());
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
