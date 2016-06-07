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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * Single-threaded frame text csv writer.
 * 
 */
public class FrameWriterTextCSV extends FrameWriter
{
	//blocksize for string concatenation in order to prevent write OOM 
	//(can be set to very large value to disable blocking)
	public static final int BLOCKSIZE_J = 32; //32 cells (typically ~512B, should be less than write buffer of 1KB)
	
	protected CSVFileFormatProperties _props = null;
	
	public FrameWriterTextCSV( CSVFileFormatProperties props ) {
		_props = props;
	}
	
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
	public final void writeFrameToHDFS(FrameBlock src, String fname, long rlen, long clen) 
		throws IOException, DMLRuntimeException 
	{
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
	
		//validity check frame dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Frame dimensions mismatch with metadata: " + 
					src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
		
		//core write (sequential/parallel)
		writeCSVFrameToHDFS(path, job, src, rlen, clen, _props);
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param csvprops
	 * @throws IOException
	 */
	protected void writeCSVFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen, CSVFileFormatProperties csvprops ) 
		throws IOException
	{
		FileSystem fs = FileSystem.get(job);
        
		//sequential write to single text file
		writeCSVFrameToFile(path, job, fs, src, 0, (int)rlen, csvprops);	
	}
	
	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param props
	 * @return
	 * @throws IOException
	 */
	protected final void writeCSVFrameToFile( Path path, JobConf job, FileSystem fs, FrameBlock src, int rl, int ru, CSVFileFormatProperties props )
		throws IOException
	{
    	//create buffered writer
		BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
    	int cols = src.getNumColumns();
	
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			props = (props==null)? new CSVFileFormatProperties() : props;
			String delim = props.getDelim();
			
			// Write header line, if needed
			if( props.hasHeader() && rl==0 ) 
			{
				//write row chunk-wise to prevent OOM on large number of columns
				for( int bj=0; bj<cols; bj+=BLOCKSIZE_J ) {
					for( int j=bj; j < Math.min(cols,bj+BLOCKSIZE_J); j++) {
						sb.append("C"+ (j+1));
						if ( j < cols-1 )
							sb.append(delim);
					}
					br.write( sb.toString() );
		            sb.setLength(0);	
				}
				sb.append('\n');
				br.write( sb.toString() );
	            sb.setLength(0);
			}
			
			// Write data lines
			Iterator<String[]> iter = src.getStringRowIterator(rl, ru);
			while( iter.hasNext() ) {
				//write row chunk-wise to prevent OOM on large number of columns
				String[] row = iter.next();
				for( int bj=0; bj<cols; bj+=BLOCKSIZE_J ) {
					for( int j=bj; j<Math.min(cols,bj+BLOCKSIZE_J); j++ ) {
						if(row[j] != null)
							sb.append(row[j]);					
						if( j != cols-1 )
							sb.append(delim);
					}
					br.write( sb.toString() );
		            sb.setLength(0);
				}
				
				sb.append('\n');
				br.write( sb.toString() );
				sb.setLength(0); 
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}
	}
}
