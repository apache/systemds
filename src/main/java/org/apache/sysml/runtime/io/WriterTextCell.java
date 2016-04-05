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

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

public class WriterTextCell extends MatrixWriter
{
	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException 
	{
		//validity check matrix dimensions
		if( src.getNumRows() != rlen || src.getNumColumns() != clen ) {
			throw new IOException("Matrix dimensions mismatch with metadata: "+src.getNumRows()+"x"+src.getNumColumns()+" vs "+rlen+"x"+clen+".");
		}
				
		//prepare file access
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeTextCellMatrixToHDFS(path, job, src, rlen, clen);
	}

	@Override
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int brlen, int bclen) 
		throws IOException, DMLRuntimeException 
	{
		Path path = new Path( fname );
		FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
		
		FSDataOutputStream writer = fs.create(path);
		writer.writeBytes("1 1 0");
		writer.close();
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
	protected void writeTextCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
		
    	int rows = src.getNumRows();
		int cols = src.getNumColumns();

		//bound check per block
		if( rows > rlen || cols > clen )
		{
			throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
					              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
		}
		
		try
		{
			//for obj reuse and preventing repeated buffer re-allocations
			StringBuilder sb = new StringBuilder();
			
			if( sparse ) //SPARSE
			{			   
				Iterator<IJV> iter = src.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.getI()+1);
					sb.append(' ');
					sb.append(cell.getJ()+1);
					sb.append(' ');
					sb.append(cell.getV());
					sb.append('\n');
					br.write( sb.toString() ); //same as append
					sb.setLength(0); 
					entriesWritten = true;					
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
				{
					String rowIndex = Integer.toString(i+1);					
					for( int j=0; j<cols; j++ )
					{
						double lvalue = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
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
