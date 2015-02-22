/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.EOFException;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * Base class for all format-specific matrix readers. Every reader is required to implement the basic 
 * read functionality but might provide additional custom functionality. Any non-default parameters
 * (e.g., CSV read properties) should be passed into custom constructors. There is also a factory
 * for creating format-specific readers. 
 * 
 */
public abstract class MatrixReader 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	
	/**
	 * 
	 * @param fname
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expNnz
	 * @return
	 */
	public abstract MatrixBlock readMatrixFromHDFS( String fname, long rlen, long clen, int brlen, int bclen, long estnnz )
		throws IOException, DMLRuntimeException;
	
	/**
	 * 
	 * @param file
	 * @return
	 * @throws IOException
	 */
	public static Path[] getSequenceFilePaths( FileSystem fs, Path file ) 
		throws IOException
	{
		Path[] ret = null;
		
		if( fs.isDirectory(file) )
		{
			LinkedList<Path> tmp = new LinkedList<Path>();
			FileStatus[] dStatus = fs.listStatus(file);
			for( FileStatus fdStatus : dStatus )
				if( !fdStatus.getPath().getName().startsWith("_") ) //skip internal files
					tmp.add(fdStatus.getPath());
			ret = tmp.toArray(new Path[0]);
		}
		else
		{
			ret = new Path[]{ file };
		}
		
		return ret;
	}
	
	/**
	 * NOTE: mallocDense controls if the output matrix blocks is fully allocated, this can be redundant
	 * if binary block read and single block. 
	 * 
	 * @param rlen
	 * @param clen
	 * @param estnnz
	 * @param mallocDense
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	protected static MatrixBlock createOutputMatrixBlock( long rlen, long clen, long estnnz, boolean mallocDense, boolean mallocSparse ) 
		throws IOException, DMLRuntimeException
	{
		//check input dimension
		if( !OptimizerUtils.isValidCPDimensions(rlen, clen) )
			throw new DMLRuntimeException("Matrix dimensions too large for CP runtime: "+rlen+" x "+clen);
		
		//determine target representation (sparse/dense)
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, estnnz); 
		
		//prepare result matrix block
		MatrixBlock ret = new MatrixBlock((int)rlen, (int)clen, sparse, (int)estnnz);
		if( !sparse && mallocDense ){
			ret.allocateDenseBlockUnsafe((int)rlen, (int)clen);
			Arrays.fill(ret.getDenseArray(),0);
		}
		else if( sparse && mallocSparse  )
			ret.allocateSparseRowsBlock();
		
		return ret;
	}
	
	/**
	 * 
	 * @param fs
	 * @param path
	 * @throws IOException 
	 */
	protected static void checkValidInputFile(FileSystem fs, Path path) 
		throws IOException
	{
		//check non-existing file
		if( !fs.exists(path) )	
			throw new IOException("File "+path.toString()+" does not exist on HDFS/LFS.");
	
		//check for empty file
		if( MapReduceTool.isFileEmpty( fs, path.toString() ) )
			throw new EOFException("Empty input file "+ path.toString() +".");
		
	}
}
