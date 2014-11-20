/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.io.IOException;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.io.MatrixReader;
import com.ibm.bi.dml.runtime.io.MatrixReaderFactory;
import com.ibm.bi.dml.runtime.io.MatrixWriter;
import com.ibm.bi.dml.runtime.io.MatrixWriterFactory;
import com.ibm.bi.dml.runtime.io.ReadProperties;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.FileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;


/**
 * This class provides methods to read and write matrix blocks from to HDFS using different data formats.
 * Those functionalities are used especially for CP read/write and exporting in-memory matrices to HDFS
 * (before executing MR jobs).
 * 
 */
public class DataConverter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	
	//////////////
	// READING and WRITING of matrix blocks to/from HDFS
	// (textcell, binarycell, binaryblock)
	///////
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo,  MatrixCharacteristics mc )
		throws IOException
	{
		writeMatrixToHDFS(mat, dir, outputinfo, mc, -1, null);
	}
	
	/**
	 * 
	 * @param mat
	 * @param dir
	 * @param outputinfo
	 * @param mc
	 * @param replication
	 * @param formatProperties
	 * @throws IOException
	 */
	public static void writeMatrixToHDFS(MatrixBlock mat, String dir, OutputInfo outputinfo, MatrixCharacteristics mc, int replication, FileFormatProperties formatProperties)
		throws IOException
	{
		try {
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter( outputinfo, replication, formatProperties );
			writer.writeMatrixFromHDFS(mat, dir, mc.get_rows(), mc.get_cols(), mc.get_rows_per_block(), mc.get_cols_per_block(), mc.getNonZeros());
		}
		catch(Exception e)
		{
			throw new IOException(e);
		}
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, boolean localFS) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.localFS = localFS;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		
		//expected matrix is sparse (default SystemML usecase)
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, int brlen, int bclen, double expectedSparsity) 
		throws IOException
	{	
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		
		return readMatrixFromHDFS(prop);
	}

	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, boolean localFS) 
		throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.localFS = localFS;
		
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @param localFS
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(String dir, InputInfo inputinfo, long rlen, long clen, 
			int brlen, int bclen, double expectedSparsity, FileFormatProperties formatProperties) 
	throws IOException
	{
		ReadProperties prop = new ReadProperties();
		
		prop.path = dir;
		prop.inputInfo = inputinfo;
		prop.rlen = rlen;
		prop.clen = clen;
		prop.brlen = brlen;
		prop.bclen = bclen;
		prop.expectedSparsity = expectedSparsity;
		prop.formatProperties = formatProperties;
		
		//prop.printMe();
		return readMatrixFromHDFS(prop);
	}
	
	/**
	 * Core method for reading matrices in format textcell, matrixmarket, binarycell, or binaryblock 
	 * from HDFS into main memory. For expected dense matrices we directly copy value- or block-at-a-time 
	 * into the target matrix. In contrast, for sparse matrices, we append (column-value)-pairs and do a 
	 * final sort if required in order to prevent large reorg overheads and increased memory consumption 
	 * in case of unordered inputs.  
	 * 
	 * DENSE MxN input:
	 *  * best/average/worst: O(M*N)
	 * SPARSE MxN input
	 *  * best (ordered, or binary block w/ clen<=bclen): O(M*N)
	 *  * average (unordered): O(M*N*log(N))
	 *  * worst (descending order per row): O(M * N^2)
	 * 
	 * NOTE: providing an exact estimate of 'expected sparsity' can prevent a full copy of the result
	 * matrix block (required for changing sparse->dense, or vice versa)
	 * 
	 * @param dir
	 * @param inputinfo
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param expectedSparsity
	 * @return
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixFromHDFS(ReadProperties prop) 
		throws IOException
	{	
		//Timing time = new Timing(true);
		
		long estnnz = (long)(prop.expectedSparsity*prop.rlen*prop.clen);
	
		//core matrix reading 
		MatrixBlock ret = null;
		try {
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(prop);
			ret = reader.readMatrixFromHDFS(prop.path, prop.rlen, prop.clen, prop.brlen, prop.bclen, estnnz);
		}
		catch(DMLRuntimeException rex)
		{
			throw new IOException(rex);
		}	
		
		//System.out.println("read matrix ("+rlen+","+clen+","+ret.getNonZeros()+") in "+time.stop());
				
		return ret;
	}

	
	//////////////
	// Utils for CREATING and COPYING matrix blocks 
	///////
	
	/**
	 * Creates a two-dimensional double matrix of the input matrix block. 
	 * 
	 * @param mb
	 * @return
	 */
	public static double[][] convertToDoubleMatrix( MatrixBlock mb )
	{
		int rows = mb.getNumRows();
		int cols = mb.getNumColumns();
		double[][] ret = new double[rows][cols];
		
		if( mb.isInSparseFormat() )
		{
			SparseRowsIterator iter = mb.getSparseRowsIterator();
			while( iter.hasNext() )
			{
				IJV cell = iter.next();
				ret[cell.i][cell.j] = cell.v;
			}
		}
		else
		{
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					ret[i][j] = mb.getValueDenseUnsafe(i, j);
		}
				
		return ret;
	}
	
	/**
	 * Creates a dense Matrix Block and copies the given double matrix into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[][] data ) 
		throws DMLRuntimeException
	{
		int rows = data.length;
		int cols = (rows > 0)? data[0].length : 0;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * Creates a dense Matrix Block and copies the given double vector into it.
	 * 
	 * @param data
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock convertToMatrixBlock( double[] data, boolean columnVector ) 
		throws DMLRuntimeException
	{
		int rows = columnVector ? data.length : 1;
		int cols = columnVector ? 1 : data.length;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		
		try
		{ 
			//copy data to mb (can be used because we create a dense matrix)
			mb.init( data, rows, cols );
		} 
		catch (Exception e){} //can never happen
		
		//check and convert internal representation
		mb.examSparsity();
		
		return mb;
	}

	/**
	 * 
	 * @param map
	 * @return
	 */
	public static MatrixBlock convertToMatrixBlock( HashMap<MatrixIndexes,Double> map )
	{
		// compute dimensions from the map
		long nrows=0, ncols=0;
		for (MatrixIndexes index : map.keySet()) {
			nrows = Math.max( nrows, index.getRowIndex() );
			ncols = Math.max( ncols, index.getColumnIndex() );
		}
		
		int rlen = (int)nrows;
		int clen = (int)ncols;
		int nnz = map.size();
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz); 		
		MatrixBlock mb = new MatrixBlock(rlen, clen, sparse, nnz);
		
		// copy map values into new block
		for (MatrixIndexes index : map.keySet()) {
			double value  = map.get(index).doubleValue();
			if ( value != 0 )
			{
				mb.quickSetValue( (int)index.getRowIndex()-1, 
						          (int)index.getColumnIndex()-1, 
						          value );
			}
		}
		
		return mb;
	}
	
}
