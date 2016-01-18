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
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * 
 */
public class WriterMatrixMarket extends MatrixWriter
{
	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
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
		writeMatrixMarketMatrixToHDFS(path, job, src, rlen, clen, nnz);
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
	 * @param fileName
	 * @param src
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	protected void writeMatrixMarketMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
        BufferedWriter br=new BufferedWriter(new OutputStreamWriter(fs.create(path,true)));		
        
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
			
			// First output MM header
			sb.append ("%%MatrixMarket matrix coordinate real general\n");
		
			// output number of rows, number of columns and number of nnz
			sb.append (rlen + " " + clen + " " + nnz + "\n");
            br.write( sb.toString());
            sb.setLength(0);
            
            // output matrix cell
			if( sparse ) //SPARSE
			{			   
				Iterator<IJV> iter = src.getSparseBlockIterator();
				while( iter.hasNext() )
				{
					IJV cell = iter.next();

					sb.append(cell.i+1);
					sb.append(' ');
					sb.append(cell.j+1);
					sb.append(' ');
					sb.append(cell.v);
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
							sb.append(j+1);
							sb.append(' ');
							sb.append(lvalue);
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
	
	/**
	 * 
	 * @param srcFileName
	 * @param fileName
	 * @param rlen
	 * @param clen
	 * @param nnz
	 * @throws IOException
	 */
	public void mergeTextcellToMatrixMarket( String srcFileName, String fileName, long rlen, long clen, long nnz )
		throws IOException
	{
		  Configuration conf = new Configuration(ConfigurationManager.getCachedJobConf());
		
		  Path src = new Path (srcFileName);
	      Path merge = new Path (fileName);
	      FileSystem hdfs = FileSystem.get(conf);
	    
	      if (hdfs.exists (merge)) {
	    	hdfs.delete(merge, true);
	      }
        
	      OutputStream out = hdfs.create(merge, true);

	      // write out the header first 
	      StringBuilder  sb = new StringBuilder();
	      sb.append ("%%MatrixMarket matrix coordinate real general\n");
	    
	      // output number of rows, number of columns and number of nnz
	 	  sb.append (rlen + " " + clen + " " + nnz + "\n");
	      out.write (sb.toString().getBytes());

	      // if the source is a directory
	      if (hdfs.getFileStatus(src).isDirectory()) {
	        try {
	          FileStatus[] contents = hdfs.listStatus(src);
	          for (int i = 0; i < contents.length; i++) {
	            if (!contents[i].isDirectory()) {
	               InputStream in = hdfs.open (contents[i].getPath());
	               try {
	                 IOUtils.copyBytes (in, out, conf, false);
	               }  finally {
	                  IOUtilFunctions.closeSilently(in);
	               }
	             }
	           }
	         } finally {
	        	 IOUtilFunctions.closeSilently(out);
	         }
	      } else if (hdfs.isFile(src))  {
	        InputStream in = null;
	        try {
   	          in = hdfs.open (src);
	          IOUtils.copyBytes (in, out, conf, true);
	        } 
	        finally {
	        	IOUtilFunctions.closeSilently(in);
	        	IOUtilFunctions.closeSilently(out);
	        }
	      } else {
	        throw new IOException(src.toString() + ": No such file or directory");
	      }
	}
}
