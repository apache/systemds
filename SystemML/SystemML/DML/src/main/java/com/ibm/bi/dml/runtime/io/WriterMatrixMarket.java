/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterMatrixMarket extends MatrixWriter
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public void writeMatrixFromHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//prepare file access
		JobConf job = new JobConf();
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeMatrixMarketMatrixToHDFS(path, job, src, rlen, clen, nnz);
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
	private static void writeMatrixMarketMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, long nnz )
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
				SparseRowsIterator iter = src.getSparseRowsIterator();
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
			if( br != null )
				br.close();
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
		  Configuration conf = new Configuration();
		
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
	          FileStatus contents[] = hdfs.listStatus(src);
	          for (int i = 0; i < contents.length; i++) {
	            if (!contents[i].isDirectory()) {
	               InputStream in = hdfs.open (contents[i].getPath());
	               try {
	                 IOUtils.copyBytes (in, out, conf, false);
	               }  finally {
	                 in.close();
	               }
	             }
	           }
	         } finally {
	            out.close();
	         }
	      } else if (hdfs.isFile(src))  {
	        InputStream in = null;
	        try {
   	          in = hdfs.open (src);
	          IOUtils.copyBytes (in, out, conf, true);
	        } finally {
	          in.close();
	          out.close();
	        }
	      } else {
	        throw new IOException(src.toString() + ": No such file or directory");
	      }
	}
}
