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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

public class FrameWriterTextCell extends FrameWriter
{
	/**
	 * @param src
	 * @param fname
	 * @return
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	@Override
	public void writeFrameToHDFS( FrameBlock src, String fname, long rlen, long clen )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//validity check frame dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Frame dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
				
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeTextCellFrameToHDFS(path, job, src, src.getNumRows(), src.getNumColumns());
	}

	/**
	 * 
	 * @param path
	 * @param job
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	protected void writeTextCellFrameToHDFS( Path path, JobConf job, FrameBlock src, long rlen, long clen )
		throws IOException
	{
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Frame block [1:"+rows+",1:"+cols+"] " +
					              "out of overall frame range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			for( int i=0; i<rows; i++ )
			{
				String rowIndex = Integer.toString(i+1);					
				for( int j=0; j<cols; j++ )
				{
					String lvalue = (String) src.get(i, j);
					sb.append(rowIndex);
					sb.append(' ');
					sb.append( j+1 );
					sb.append(' ');
					sb.append( lvalue );
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
					entriesWritten = true;
				}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				br.write("1 1 0\n");
			}
		}
		finally
		{
			IOUtilFunctions.closeSilently(br);
		}
	}	

}
