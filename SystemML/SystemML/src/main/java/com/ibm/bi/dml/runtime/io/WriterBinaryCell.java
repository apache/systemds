/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriterBinaryCell extends MatrixWriter
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz) 
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//prepare file access
		JobConf job = new JobConf();
		Path path = new Path( fname );

		//if the file already exists on HDFS, remove it.
		MapReduceTool.deleteFileIfExistOnHDFS( fname );
			
		//core write
		writeBinaryCellMatrixToHDFS(path, job, src, rlen, clen, brlen, bclen);
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
	@SuppressWarnings("deprecation")
	private static void writeBinaryCellMatrixToHDFS( Path path, JobConf job, MatrixBlock src, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = src.isInSparseFormat();
		boolean entriesWritten = false;
		FileSystem fs = FileSystem.get(job);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, MatrixIndexes.class, MatrixCell.class);
		
		MatrixIndexes indexes = new MatrixIndexes();
		MatrixCell cell = new MatrixCell();

		int rows = src.getNumRows(); 
		int cols = src.getNumColumns();
        
		try
		{
			//bound check per block
			if( rows > rlen || cols > clen )
			{
				throw new IOException("Matrix block [1:"+rows+",1:"+cols+"] " +
						              "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
		
			if( sparse ) //SPARSE
			{
				
				SparseRowsIterator iter = src.getSparseRowsIterator();
				while( iter.hasNext() )
				{
					IJV lcell = iter.next();
					indexes.setIndexes(lcell.i+1, lcell.j+1);
					cell.setValue(lcell.v);
					writer.append(indexes, cell);
					entriesWritten = true;
				}
			}
			else //DENSE
			{
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
					{
						double lvalue  = src.getValueDenseUnsafe(i, j);
						if( lvalue != 0 ) //for nnz
						{
							indexes.setIndexes(i+1, j+1);
							cell.setValue(lvalue);
							writer.append(indexes, cell);
							entriesWritten = true;
						}
					}
			}
	
			//handle empty result
			if ( !entriesWritten ) {
				writer.append(new MatrixIndexes(1, 1), new MatrixCell(0));
			}
		}
		finally
		{
			if( writer != null )
				writer.close();
		}
	}
}
