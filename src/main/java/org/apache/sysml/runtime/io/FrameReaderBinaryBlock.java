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
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;



public class FrameReaderBinaryBlock extends FrameReader
{
	/**
	 * 
	 * @param fname
	 * @param schema
	 * @param names
	 * @param rlen
	 * @param clen
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	@Override
	public FrameBlock readFrameFromHDFS(String fname, List<ValueType> schema, List<String> names,
			long rlen, long clen) 
			throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		FrameBlock ret = createOutputFrameBlock(schema, names, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname ); 
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryBlockFrameFromHDFS(path, job, fs, ret, rlen, clen);
		
		return ret;
	}
	
	/**
	 * Note: For efficiency, we directly use SequenceFile.Reader instead of SequenceFileInputFormat-
	 * InputSplits-RecordReader (SequenceFileRecordReader). First, this has no drawbacks since the
	 * SequenceFileRecordReader internally uses SequenceFile.Reader as well. Second, it is 
	 * advantageous if the actual sequence files are larger than the file splits created by   
	 * informat.getSplits (which is usually aligned to the HDFS block size) because then there is 
	 * overhead for finding the actual split between our 1k-1k blocks. This case happens
	 * if the read frame was create by CP or when jobs directly write to large output files 
	 * (e.g., parfor matrix partitioning).
	 * 
	 * @param path
	 * @param job
	 * @param fs 
	 * @param dest
	 * @param rlen
	 * @param clen
	 * 
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	@SuppressWarnings("deprecation")
	private static void readBinaryBlockFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		LongWritable key = new LongWritable();
		FrameBlock value = new FrameBlock();
		
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
		{
			//directly read from sequence files (individual partfiles)
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
			
			try
			{
				//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
				while( reader.next(key, value) ) {	
					int row_offset = (int)(key.get()-1);
					
					int rows = value.getNumRows();
					int cols = value.getNumColumns();
					
					//bound check per block
					if( row_offset + rows < 0 || row_offset + rows > rlen ) {
						throw new IOException("Frame block ["+(row_offset+1)+":"+(row_offset+rows)+","+":"+"] " +
								              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
					}
			
					dest.copy( row_offset, row_offset+rows-1, 
							0, cols-1, value);
				}
			}
			finally {
				IOUtilFunctions.closeSilently(reader);
			}
		}
	}

	/**
	 * Specific functionality of FrameReaderBinaryBlock, mostly used for testing.
	 * 
	 * @param fname
	 * @return
	 * @throws IOException 
	 */
	@SuppressWarnings("deprecation")
	public FrameBlock readFirstBlock(String fname) throws IOException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname ); 
		
		LongWritable key = new LongWritable();
		FrameBlock value = new FrameBlock();
		
		//read first block from first file
		Path lpath = getSequenceFilePaths(fs, path)[0];  
		SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);		
		try {
			reader.next(key, value);
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
		
		return value;
	}
}
