/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

/**
 * 
 * 
 */
public class ReaderBinaryCell extends MatrixReader
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int brlen, int bclen, long estnnz) 
		throws IOException, DMLRuntimeException 
	{
		//allocate output matrix block
		MatrixBlock ret = createOutputMatrixBlock(rlen, clen, estnnz, true, false);
		
		//prepare file access
		JobConf job = new JobConf();	
		FileSystem fs = FileSystem.get(job);
		Path path = new Path( fname );
		
		//check existence and non-empty file
		checkValidInputFile(fs, path); 
	
		//core read 
		readBinaryCellMatrixFromHDFS(path, job, fs, ret, rlen, clen, brlen, bclen);
		
		//finally check if change of sparse/dense block representation required
		if( !ret.isInSparseFormat() )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
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
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	@SuppressWarnings("deprecation")
	private void readBinaryCellMatrixFromHDFS( Path path, JobConf job, FileSystem fs, MatrixBlock dest, long rlen, long clen, int brlen, int bclen )
		throws IOException
	{
		boolean sparse = dest.isInSparseFormat();		
		MatrixIndexes key = new MatrixIndexes();
		MatrixCell value = new MatrixCell();
		int row = -1;
		int col = -1;
		
		try
		{
			for( Path lpath : getSequenceFilePaths(fs,path) ) //1..N files 
			{
				//directly read from sequence files (individual partfiles)
				SequenceFile.Reader reader = new SequenceFile.Reader(fs,lpath,job);
				
				try
				{
					if( sparse )
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							//dest.quickSetValue( row, col, lvalue );
							dest.appendValue(row, col, lvalue);
						}
					}
					else
					{
						while(reader.next(key, value))
						{
							row = (int)key.getRowIndex()-1;
							col = (int)key.getColumnIndex()-1;
							double lvalue = value.getValue();
							dest.setValueDenseUnsafe( row, col, lvalue );
						}
					}
				}
				finally
				{
					IOUtilFunctions.closeSilently(reader);
				}
			}
			
			if( sparse )
				dest.sortSparseRows();
		}
		catch(Exception ex)
		{
			//post-mortem error handling and bounds checking
			if( row < 0 || row + 1 > rlen || col < 0 || col + 1 > clen )
			{
				throw new IOException("Matrix cell ["+(row+1)+","+(col+1)+"] " +
									  "out of overall matrix range [1:"+rlen+",1:"+clen+"].");
			}
			else
			{
				throw new IOException( "Unable to read matrix in binary cell format.", ex );
			}
		}
	}
	
}
