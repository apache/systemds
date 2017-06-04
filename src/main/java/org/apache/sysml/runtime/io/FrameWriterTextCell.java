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
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * Single-threaded frame text cell writer.
 * 
 */
public class FrameWriterTextCell extends FrameWriter
{

	@Override
	public final void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
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
		writeTextCellFrameToHDFS(path, job, src, src.getNumRows(), src.getNumColumns());
	}

	protected void writeTextCellFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen )
		throws IOException
	{
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
		
		//sequential write to single text file
		writeTextCellFrameToFile(path, job, fs, src, 0, (int)rlen);	
	}	
	
	/**
	 * Internal primitive to write a row range of a frame to a single text file, 
	 * which is used for both single- and multi-threaded writers (for consistency). 
	 * 
	 * @param path file path
	 * @param job job configuration
	 * @param fs file system
	 * @param src frame block
	 * @param rl lower row
	 * @param ru upper row
	 * @throws IOException if IOException occurs
	 */
	protected final void writeTextCellFrameToFile( Path path, JobConf job, FileSystem fs, FrameBlock src, int rl, int ru ) 
		throws IOException
	{
		boolean entriesWritten = false;
    	int cols = src.getNumColumns();

    	//create buffered writer 
    	BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		

		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			//write frame meta data
			if( rl == 0 ) {
				for( int j=0; j<cols; j++ )
					if( !src.isColumnMetadataDefault(j) ) {
						sb.append("-1 " + (j+1) + " " + src.getColumnMetadata(j).getNumDistinct() + "\n");
						sb.append("-2 " + (j+1) + " " + src.getColumnMetadata(j).getMvValue() + "\n");
						br.write( sb.toString() );
						sb.setLength(0);
					}
			}
			
			//write frame row range to output
			Iterator<String[]> iter = src.getStringRowIterator(rl, ru);
			for( int i=rl; iter.hasNext(); i++ ) { //for all rows
				String rowIndex = Integer.toString(i+1);
				String[] row = iter.next();
				for( int j=0; j<cols; j++ ) {
					if( row[j] != null ) {
						sb.append( rowIndex );
						sb.append(' ');
						sb.append( j+1 );
						sb.append(' ');
						sb.append( row[j] );
						sb.append('\n');
						br.write( sb.toString() );
						sb.setLength(0); 
						entriesWritten = true;
					}
				}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				br.write("1 1 0\n");
			}
		}
		finally {
			IOUtilFunctions.closeSilently(br);
		}		
	}
}
