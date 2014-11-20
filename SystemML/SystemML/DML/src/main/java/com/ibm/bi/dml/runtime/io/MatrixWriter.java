/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.io;

import java.io.IOException;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

/**
 * Base class for all format-specific matrix writers. Every writer is required to implement the basic 
 * write functionality but might provide additional custom functionality. Any non-default parameters
 * (e.g., CSV read properties) should be passed into custom constructors. There is also a factory
 * for creating format-specific writers. 
 * 
 */
public abstract class MatrixWriter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
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
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public abstract void writeMatrixFromHDFS( MatrixBlock src, String fname, long rlen, long clen, int brlen, int bclen, long nnz )
		throws IOException, DMLRuntimeException, DMLUnsupportedOperationException;
	
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param sparse
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock[] createMatrixBlocksForReuse( long rlen, long clen, int brlen, int bclen, boolean sparse, long nonZeros ) 
		throws DMLRuntimeException
	{
		MatrixBlock[] blocks = new MatrixBlock[4];
		double sparsity = ((double)nonZeros)/(rlen*clen);
		long estNNZ = -1;
		
		//full block 
		if( rlen >= brlen && clen >= bclen )
		{
			estNNZ = (long) (brlen*bclen*sparsity);
			blocks[0] = new MatrixBlock( brlen, bclen, sparse, (int)estNNZ );
		}
		//partial col block
		if( rlen >= brlen && clen%bclen!=0 )
		{
			estNNZ = (long) (brlen*(clen%bclen)*sparsity);
			blocks[1] = new MatrixBlock( brlen, (int)(clen%bclen), sparse, (int)estNNZ );
		}
		//partial row block
		if( rlen%brlen!=0 && clen>=bclen )
		{
			estNNZ = (long) ((rlen%brlen)*bclen*sparsity);
			blocks[2] = new MatrixBlock( (int)(rlen%brlen), bclen, sparse, (int)estNNZ );
		}
		//partial row/col block
		if( rlen%brlen!=0 && clen%bclen!=0 )
		{
			estNNZ = (long) ((rlen%brlen)*(clen%bclen)*sparsity);
			blocks[3] = new MatrixBlock( (int)(rlen%brlen), (int)(clen%bclen), sparse, (int)estNNZ );
		}
		
		//space allocation
		for( MatrixBlock b : blocks )
			if( b != null )
				if( !sparse )
					b.allocateDenseBlockUnsafe(b.getNumRows(), b.getNumColumns());		
		//NOTE: no preallocation for sparse (preallocate sparserows with estnnz) in order to reduce memory footprint
		
		return blocks;
	}
	
	/**
	 * 
	 * @param blocks
	 * @param rows
	 * @param cols
	 * @param brlen
	 * @param bclen
	 * @return
	 */
	public static MatrixBlock getMatrixBlockForReuse( MatrixBlock[] blocks, int rows, int cols, int brlen, int bclen )
	{
		int index = -1;
		
		if( rows==brlen && cols==bclen )
			index = 0;
		else if( rows==brlen && cols<bclen )
			index = 1;
		else if( rows<brlen && cols==bclen )
			index = 2;
		else //if( rows<brlen && cols<bclen )
			index = 3;

		return blocks[ index ];
	}
}
