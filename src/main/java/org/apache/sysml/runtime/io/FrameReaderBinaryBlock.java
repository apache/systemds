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


/**
 * Single-threaded frame binary block reader.
 * 
 */
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
	public final FrameBlock readFrameFromHDFS(String fname, List<ValueType> schema, List<String> names, long rlen, long clen) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output frame block
		List<ValueType> lschema = createOutputSchema(schema, clen);
		List<String> lnames = createOutputNames(names, clen);
		FrameBlock ret = createOutputFrameBlock(lschema, lnames, rlen);
		
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname ); 
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read (sequential/parallel)
		readBinaryBlockFrameFromHDFS(path, job, fs, ret, rlen, clen);
		
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
	 * 
	 * @throws IOException
	 * @throws DMLRuntimeException 
	 */
	protected void readBinaryBlockFrameFromHDFS( Path path, JobConf job, FileSystem fs, FrameBlock dest, long rlen, long clen )
		throws IOException, DMLRuntimeException
	{
		//sequential read from sequence files
		for( Path lpath : getSequenceFilePaths(fs, path) ) //1..N files 
			readBinaryBlockFrameFromSequenceFile(lpath, job, fs, dest);
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param fs
	 * @param dest
	 * @throws IOException
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings({ "deprecation", "resource" })
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
			//note: next(key, value) does not yet exploit the given serialization classes, record reader does but is generally slower.
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
		
				dest.copy( row_offset, row_offset+rows-1, 
						0, cols-1, value);
			}
		}
		finally {
			IOUtilFunctions.closeSilently(reader);
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
