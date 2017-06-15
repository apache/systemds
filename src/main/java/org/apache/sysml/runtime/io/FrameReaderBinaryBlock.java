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
import java.io.InputStream;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;


/**
 * Single-threaded frame binary block reader.
 * 
 */
public class FrameReaderBinaryBlock extends FrameReader
{
	@Override
	public final FrameBlock readFrameFromHDFS(String fname, ValueType[] schema, String[] names, long rlen, long clen) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		ValueType[] lschema = createOutputSchema(schema, clen);
		String[] lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		Path path = new Path( fname ); 
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read (sequential/parallel)
		readBinaryBlockFrameFromHDFS(path, job, fs, ret, rlen, clen);
		
		return ret;
	}
	
	@Override
	public FrameBlock readFrameFromInputStream(InputStream is, ValueType[] schema, String[] names, long rlen, long clen)
		throws IOException, DMLRuntimeException 
	{
		throw new DMLRuntimeException("Not implemented yet.");
	}

	protected void readBinaryBlockFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		//sequential read from sequence files
		for( Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path) ) //1..N files 
			readBinaryBlockFrameFromSequenceFile(lpath, job, fs, dest);
	}

	@SuppressWarnings({ "deprecation" })
	protected final void readBinaryBlockFrameFromSequenceFile( Path path, JobConf job, FileSystem fs, FrameBlock dest )
		throws IOException, DMLRuntimeException
	{
		int rlen = dest.getNumRows();
		int clen = dest.getNumColumns();
		
		//directly read from sequence files (individual partfiles)
		SequenceFile.Reader reader = new SequenceFile.Reader(fs,path,job);
		LongWritable key = new LongWritable(-1L);
		FrameBlock value = new FrameBlock();
		
		try
		{
			while( reader.next(key, value) ) {	
				int row_offset = (int)(key.get()-1);
				int rows = value.getNumRows();
				int cols = value.getNumColumns();
				
				if(rows == 0 || cols == 0)	//Empty block, ignore it.
					continue;
				
				//bound check per block
				if( row_offset + rows < 0 || row_offset + rows > rlen ) {
					throw new IOException("Frame block ["+(row_offset+1)+":"+(row_offset+rows)+","+":"+"] " +
							              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
				}
		
				//copy block into target frame, incl meta on first
				dest.copy( row_offset, row_offset+rows-1, 0, cols-1, value);
				if( row_offset==0 )
					dest.setColumnMetadata(value.getColumnMetadata());
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
		}
	}	
	
	/**
	 * Specific functionality of FrameReaderBinaryBlock, mostly used for testing.
	 * 
	 * @param fname file name
	 * @return frame block
	 * @throws IOException if IOException occurs
	 */
	@SuppressWarnings("deprecation")
	public FrameBlock readFirstBlock(String fname) throws IOException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		Path path = new Path( fname ); 
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		LongWritable key = new LongWritable();
		FrameBlock value = new FrameBlock();
		
		//read first block from first file
		Path lpath = IOUtilFunctions.getSequenceFilePaths(fs, path)[0];  
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
