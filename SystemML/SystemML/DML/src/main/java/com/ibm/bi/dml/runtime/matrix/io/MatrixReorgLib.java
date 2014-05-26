/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.MaxIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

/**
 * MB:
 * Library for selected matrix reorg operations including special cases
 * and all combinations of dense and sparse representations.
 * 
 * Current list of supported operations:
 *  - reshape, r' (transpose)
 * 
 */
public class MatrixReorgLib 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final boolean ALLOW_BLOCK_REUSE = false;
	
	private enum ReorgType {
		TRANSPOSE,
		DIAG,
		RESHAPE,
		INVALID,
	}
	
	/////////////////////////
	// public interface    //
	/////////////////////////
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isSupportedReorgOperator( ReorgOperator op )
	{
		return (getReorgType(op) != ReorgType.INVALID);
	}

	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlockDSM reorg( MatrixBlockDSM in, MatrixBlockDSM out, ReorgOperator op ) 
		throws DMLRuntimeException
	{
		ReorgType type = getReorgType(op);
		
		switch( type )
		{
			case TRANSPOSE: return transpose(in, out);
			case DIAG:      return diag(in, out); 
			
			default:        
				throw new DMLRuntimeException("Unsupported reorg operator: "+op.fn);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlockDSM transpose( MatrixBlockDSM in, MatrixBlockDSM out ) 
		throws DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		if( !in.sparse && !out.sparse )
			transposeDenseToDense( in, out );
		else if( in.sparse && out.sparse )
			transposeSparseToSparse( in, out );
		else if( in.sparse )
			transposeSparseToDense( in, out );
		else
			transposeDenseToSparse( in, out );
		
		//System.out.println("r' ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}

	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlockDSM diag( MatrixBlockDSM in, MatrixBlockDSM out ) 
			throws DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		int rlen = in.rlen;
		int clen = in.clen;
		
		if( clen == 1 ) //diagV2M
			diagV2M( in, out );
		else if ( rlen == clen ) //diagM2V
			diagM2V( in, out );
		else
			throw new DMLRuntimeException("Reorg diagM2V requires squared block input. ("+rlen+", "+clen+")");
		
		//System.out.println("rdiag ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}
	
	/**
	 * CP reshape operation (single input, single output matrix) 
	 *
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlockDSM reshape( MatrixBlockDSM in, MatrixBlockDSM out, int rows, int cols, boolean rowwise ) 
		throws DMLRuntimeException
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//check validity
		if( ((long)rlen)*clen != ((long)rows)*cols )
			throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells ("+rlen+":"+clen+", "+rows+":"+cols+").");
		
		//check for same dimensions
		if( rlen==rows && clen == cols ) {
			out.copy(in); //incl dims, nnz
			return out;
		}
	
		//determine output representation
	    out.sparse = MatrixBlockDSM.evalSparseFormatInMemory(rows, cols, in.nonZeros);
		
		//core reshape (sparse or dense)	
		if(!in.sparse && !out.sparse)
			reshapeDense(in, out, rows, cols, rowwise);		
		else if(in.sparse && out.sparse)
			reshapeSparse(in, out, rows, cols, rowwise);
		else if(in.sparse)
			reshapeSparseToDense(in, out, rows, cols, rowwise);
		else
			reshapeDenseToSparse(in, out, rows, cols, rowwise);
		
		//finally set output dimensions
		out.rlen = rows;
		out.clen = cols;
		out.nonZeros = in.nonZeros;
		
		//out.print();
		
		
		return out;
	}


	/**
	 * MR interface - for reshape we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 *  
	 * @param in
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param out
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ArrayList<IndexedMatrixValue> reshape( IndexedMatrixValue in, long rows1, long cols1, int brlen1, int bclen1, 
			                      ArrayList<IndexedMatrixValue> out, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise ) 	
		throws DMLRuntimeException
	{
		//prepare inputs
		MatrixIndexes ixIn = in.getIndexes();
		MatrixBlockDSM mbIn = (MatrixBlockDSM) in.getValue();
		
		//prepare result blocks (no reuse in order to guarantee mem constraints)
		Collection<MatrixIndexes> rix = computeAllResultBlockIndexes(ixIn, rows1, cols1, brlen1, bclen1, rows2, cols2, brlen2, bclen2, rowwise);
		HashMap<MatrixIndexes, MatrixBlockDSM> rblk = createAllResultBlocks(rix, mbIn.nonZeros, rows1, cols1, brlen1, bclen1, rows2, cols2, brlen2, bclen2, rowwise, out);
		
		//basic algorithm
		long row_offset = (ixIn.getRowIndex()-1)*brlen1;
		long col_offset = (ixIn.getColumnIndex()-1)*bclen1;
		if( mbIn.sparse )
			reshapeSparse(mbIn, row_offset, col_offset, rblk, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
		else //dense
			reshapeDense(mbIn, row_offset, col_offset, rblk, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);

		//prepare output
		out = new ArrayList<IndexedMatrixValue>();
		for( Entry<MatrixIndexes, MatrixBlockDSM> e : rblk.entrySet() )
			out.add(new IndexedMatrixValue(e.getKey(),e.getValue()));
		
		return out;
	}
	
	

	///////////////////////////////
	// private CP implementation //
	///////////////////////////////

	
	/**
	 * 
	 * @param op
	 * @return
	 */
	private static ReorgType getReorgType( ReorgOperator op )
	{
		if( op.fn.equals(SwapIndex.getSwapIndexFnObject()) )  //transpose
			return ReorgType.TRANSPOSE;
		
		if( op.fn.equals(MaxIndex.getMaxIndexFnObject()) ) //diag
			return ReorgType.DIAG;
				
		return ReorgType.INVALID;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeDenseToDense(MatrixBlockDSM in, MatrixBlockDSM out)
	{
		if( in.denseBlock == null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//allocate output arrays (if required)
		out.sparse = false;
		out.allocateDenseBlock();
		
		double[] a = in.getDenseArray();
		double[] c = out.getDenseArray();
		
		if( m==1 || n==1 ) //VECTOR TRANSPOSE
		{
			System.arraycopy(a, 0, c, 0, m2*n2);
		}
		else //MATRIX TRANSPOSE
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128; 
			
			//blocked execution
			for( int bi = 0; bi<m; bi+=blocksizeI )
				for( int bj = 0; bj<n; bj+=blocksizeJ )
				{
					int bimin = Math.min(bi+blocksizeI, m);
					int bjmin = Math.min(bj+blocksizeJ, n);
					//core transpose operation
					for( int i=bi; i<bimin; i++ )
					{
						int aix = i * n + bj;
						int cix = bj * n2 + i;
						transposeRow(a, c, aix, cix, n2, bjmin-bj);
					}
				}
		}
		
		out.nonZeros = in.nonZeros;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeDenseToSparse(MatrixBlockDSM in, MatrixBlockDSM out)
	{
		if( in.denseBlock == null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = in.nonZeros/m2; 
		
		//allocate output arrays (if required)
		out.reset(m2, n2, true); //always sparse
		out.adjustSparseRows(m2-1);
				
		double[] a = in.getDenseArray();
		SparseRow[] c = out.getSparseRows();
		
		//blocking according to typical L2 cache sizes 
		final int blocksizeI = 128;
		final int blocksizeJ = 128; 
		
		//blocked execution
		for( int bi = 0; bi<m; bi+=blocksizeI )
			for( int bj = 0; bj<n; bj+=blocksizeJ )
			{
				int bimin = Math.min(bi+blocksizeI, m);
				int bjmin = Math.min(bj+blocksizeJ, n);
				//core transpose operation
				for( int i=bi; i<bimin; i++ )				
					for( int j=bj, aix=i*n+bj; j<bjmin; j++, aix++ )
					{
						if( c[j] == null )
							 c[j] = new SparseRow(ennz2,n2);
						c[j].append(i, a[aix]);
					}
			}
		
		out.nonZeros = in.nonZeros;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeSparseToSparse(MatrixBlockDSM in, MatrixBlockDSM out)
	{
		if( in.sparseRows == null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = in.nonZeros/m2; 
		
		//allocate output arrays (if required)
		out.reset(m2, n2, true); //always sparse
		out.adjustSparseRows(m2-1);
		
		SparseRow[] a = in.getSparseRows();
		SparseRow[] c = out.getSparseRows();
		
		//blocking according to typical L2 cache sizes 
		final int blocksizeI = 128;
		final int blocksizeJ = 128; 
	
		//temporary array for block boundaries (for preventing binary search) 
		int[] ix = new int[blocksizeI];
		
		//blocked execution
		for( int bi = 0; bi<m; bi+=blocksizeI )
		{
			Arrays.fill(ix, 0);
			for( int bj = 0; bj<n; bj+=blocksizeJ )
			{
				int bimin = Math.min(bi+blocksizeI, m);
				int bjmin = Math.min(bj+blocksizeJ, n);

				//core transpose operation
				for( int i=bi, iix=0; i<bimin; i++, iix++ )
				{
					SparseRow arow = a[i];
					if( arow!=null && arow.size()>0 )
					{
						int alen = arow.size();
						double[] avals = arow.getValueContainer();
						int[] aix = arow.getIndexContainer();
						int j = ix[iix]; //last block boundary
						for( ; j<alen && aix[j]<bjmin; j++ )
						{
							if( c[aix[j]] == null )
								 c[aix[j]] = new SparseRow(ennz2,n2);
							c[aix[j]].append(i, avals[j]);
						}
						ix[iix] = j; //keep block boundary
					}
				}
			}
		}
		out.nonZeros = in.nonZeros;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeSparseToDense(MatrixBlockDSM in, MatrixBlockDSM out)
	{
		if( in.sparseRows == null )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//allocate output arrays (if required)
		out.reset(m2, n2, false); //always dense
		out.allocateDenseBlock();
		
		SparseRow[] a = in.getSparseRows();
		double[] c = out.getDenseArray();
		
		//blocking according to typical L2 cache sizes 
		final int blocksizeI = 128;
		final int blocksizeJ = 128; 
	
		//temporary array for block boundaries (for preventing binary search) 
		int[] ix = new int[blocksizeI];
		
		//blocked execution
		for( int bi = 0; bi<m; bi+=blocksizeI )
		{
			Arrays.fill(ix, 0);
			for( int bj = 0; bj<n; bj+=blocksizeJ )
			{
				int bimin = Math.min(bi+blocksizeI, m);
				int bjmin = Math.min(bj+blocksizeJ, n);

				//core transpose operation
				for( int i=bi, iix=0; i<bimin; i++, iix++ )
				{
					SparseRow arow = a[i];
					if( arow!=null && arow.size()>0 )
					{
						int alen = arow.size();
						double[] avals = arow.getValueContainer();
						int[] aix = arow.getIndexContainer();
						int j = ix[iix]; //last block boundary
						for( ; j<alen && aix[j]<bjmin; j++ )
							c[ aix[j]*n2+i ] = avals[ j ];
						ix[iix] = j; //keep block boundary						
					}
				}
			}
		}
		out.nonZeros = in.nonZeros;
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param aix
	 * @param cix
	 * @param n2
	 * @param len
	 */
	private static void transposeRow( double[] a, double[] c, int aix, int cix, int n2, int len )
	{
		final int bn = len%8;
		
		//compute rest (not aligned to 8-blocks)
		for( int j=0; j<bn; j++, aix++, cix+=n2 )
			c[ cix ] = a[ aix+0 ];	
		
		//unrolled 8-blocks
		for( int j=bn; j<len; j+=8, aix+=8, cix+=8*n2 )
		{
			c[ cix + 0*n2 ] = a[ aix+0 ];
			c[ cix + 1*n2 ] = a[ aix+1 ];
			c[ cix + 2*n2 ] = a[ aix+2 ];
			c[ cix + 3*n2 ] = a[ aix+3 ];
			c[ cix + 4*n2 ] = a[ aix+4 ];
			c[ cix + 5*n2 ] = a[ aix+5 ];
			c[ cix + 6*n2 ] = a[ aix+6 ];
			c[ cix + 7*n2 ] = a[ aix+7 ];	
		}
	}
	
	/**
	 * Generic implementation diagV2M (non-performance critical)
	 * (in most-likely DENSE, out most likely SPARSE)
	 * 
	 * @param in
	 * @param out
	 */
	private static void diagV2M( MatrixBlockDSM in, MatrixBlockDSM out )
	{
		int rlen = in.rlen;
		
		//CASE column vector
		for( int i=0; i<rlen; i++ )
		{
			double val = in.quickGetValue(i, 0);
			if( val != 0 )
				out.appendValue(i, i, val);
		}
	}
	
	/**
	 * Generic implementation diagM2V (non-performance critical)
	 * (in most-likely SPARSE, out most likely DENSE)
	 * 
	 * NOTE: squared block assumption (checked on entry diag)
	 * 
	 * @param in
	 * @param out
	 */
	private static void diagM2V( MatrixBlockDSM in, MatrixBlockDSM out )
	{
		int rlen = in.rlen;
		
		for( int i=0; i<rlen; i++ )
		{
			double val = in.quickGetValue(i, i);
			if( val != 0 )
				out.quickSetValue(i, 0, val);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
	private static void reshapeDense( MatrixBlockDSM in, MatrixBlockDSM out, int rows, int cols, boolean rowwise )
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.denseBlock == null )
			return;
		
		//allocate block if necessary
		if(out.denseBlock==null)
			out.denseBlock = new double[rows * cols];
		
		//dense reshape
		double[] a = in.denseBlock;
		double[] c = out.denseBlock;
		
		if( rowwise )
		{
			//VECTOR-MATRIX, MATRIX-VECTOR, GENERAL CASE
			//pure copy of rowwise internal representation
			System.arraycopy(a, 0, c, 0, c.length);
		}	
		else //colwise
		{
			if( rlen==1 || clen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not on c
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0, cix=0; i<rows; i++, cix+=cols )
						c[ cix + j ] = a[ aix++ ];
			}
			else if( rows==1 || cols==1 ) //MATRIX->VECTOR	
			{
				//note: cache-friendly on c but not on a
				for( int j=0, cix=0; j<clen; j++ )
					for( int i=0, aix=0; i<rlen; i++, aix+=clen )
						c[ cix++ ] = a[ aix + j ];
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on c but not an a
				for( int i=0, cix=0; i<rows; i++ )
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows )
					{
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						c[ cix++ ] = a[ ai*clen+aj ];				
					}			
				//index conversion c[i,j]<- a[k,l]: 
				// k = (rows*j+i)%rlen
				// l = (rows*j+i)/rlen
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
	private static void reshapeSparse( MatrixBlockDSM in, MatrixBlockDSM out, int rows, int cols, boolean rowwise )
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.sparseRows == null )
			return;
		
		//allocate block if necessary
		if(out.sparseRows==null)
			out.sparseRows=new SparseRow[rows];
		int estnnz = in.nonZeros/rows;
		
		//sparse reshape
		SparseRow[] aRows = in.sparseRows;
		SparseRow[] cRows = out.sparseRows;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix not really different from general
			// * clen=1 and cols=1 will never be sparse.
			
			if( rows==1 ) //MATRIX->VECTOR	
			{
				//note: cache-friendly on a and c; append-only
				if( cRows[0] == null )
					cRows[0] = new SparseRow(estnnz, cols);
				SparseRow crow = cRows[0];
				for( int i=0, cix=0; i<rlen; i++, cix+=clen ) 
				{
					SparseRow arow = aRows[i];
					if( arow!=null && arow.size()>0 ) {
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();	
						for( int j=0; j<alen; j++ )
							crow.append(cix+aix[j], avals[j]);
					}
				}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				for( int i=0, cix=0; i<rlen; i++, cix+=clen ) 
				{
					SparseRow arow = aRows[i];
					if( arow!=null && arow.size()>0 ){
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();	
						for( int j=0; j<alen; j++ )
						{
							int ci = (cix+aix[j])/cols;
							int cj = (cix+aix[j])%cols;       
							if( cRows[ci] == null )
								cRows[ci] = new SparseRow(estnnz, cols);
							cRows[ci].append(cj, avals[j]);
						}
					}	
				}
			}
		}	
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			// * clen=1 and cols=1 will never be sparse.
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				SparseRow arow = aRows[0];
				if( arow!=null && arow.size()>0 ){
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					for( int j=0; j<alen; j++ )
					{
						int ci = aix[j]%rows;
						int cj = aix[j]/rows;       
						if( cRows[ci] == null )
							cRows[ci] = new SparseRow(estnnz, cols);
						cRows[ci].append(cj, avals[j]);
					}
				}								
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c; append&sort, in-place w/o shifts
				for( int i=0; i<rlen; i++ ) 
				{
					SparseRow arow = aRows[i];
					if( arow!=null && arow.size()>0 ){
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();	
						for( int j=0; j<alen; j++ )
						{
							int tmpix = aix[j]*rlen+i;
							int ci = tmpix%rows;
							int cj = tmpix/rows;       
							if( cRows[ci] == null )
								cRows[ci] = new SparseRow(estnnz, cols);
							cRows[ci].append(cj, avals[j]);
						}
					}	
				}
				out.sortSparseRows();
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
	private static void reshapeDenseToSparse( MatrixBlockDSM in, MatrixBlockDSM out, int rows, int cols, boolean rowwise )
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.denseBlock == null )
			return;
		
		//allocate block if necessary
		if(out.sparseRows==null)
			out.sparseRows=new SparseRow[rows];
		int estnnz = in.nonZeros/rows;
		
		//sparse reshape
		double[] a = in.denseBlock;
		SparseRow[] cRows = out.sparseRows;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c; append-only
			for( int i=0, aix=0; i<rows; i++ ) 
				for( int j=0; j<cols; j++ )
				{
					double val = a[aix++];
					if( val != 0 ){
						if( cRows[i] == null )
							cRows[i] = new SparseRow(estnnz, cols);
						cRows[i].append(j, val);
					}
				}
		}	
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0; i<rows; i++ ) 
					{
						double val = a[aix++];
						if( val != 0 ){
							if( cRows[i] == null )
								cRows[i] = new SparseRow(estnnz, cols);
							cRows[i].append(j, val);
						}
					}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on c but not a; append-only
				for( int i=0; i<rows; i++ )
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows )
					{
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						double val = a[ ai*clen+aj ];
						if( val != 0 ){
							if( cRows[i] == null )
								cRows[i] = new SparseRow(estnnz, cols);
							cRows[i].append(j, val);
						}
					}			
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
	private static void reshapeSparseToDense( MatrixBlockDSM in, MatrixBlockDSM out, int rows, int cols, boolean rowwise )
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.sparseRows == null )
			return;
		
		//allocate block if necessary
		if(out.denseBlock==null)
			out.denseBlock = new double[rows * cols];
		
		//sparse/dense reshape
		SparseRow[] aRows = in.sparseRows;
		double[] c = out.denseBlock;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c
			for( int i=0, cix=0; i<rlen; i++, cix+=clen ) 
			{
				SparseRow arow = aRows[i];
				if( arow!=null && arow.size()>0 ){
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();	
					for( int j=0; j<alen; j++ )
						c[cix+aix[j]] = avals[j];
				}	
			}
		}	
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c
				SparseRow arow = aRows[0];
				if( arow!=null && arow.size()>0 ){
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();	
					for( int j=0; j<alen; j++ )
					{
						int ci = aix[j]%rows;
						int cj = aix[j]/rows;       
						c[ci*cols+cj] = avals[j];
					}
				}								
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c
				for( int i=0; i<rlen; i++ ) 
				{
					SparseRow arow = aRows[i];
					if( arow!=null && arow.size()>0 ){
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();	
						for( int j=0; j<alen; j++ )
						{
							int tmpix = aix[j]*rlen+i;
							int ci = tmpix%rows;
							int cj = tmpix/rows;   
							c[ci*cols+cj] = avals[j];
						}
					}	
				}
			}
		}
	}
	
	///////////////////////////////
	// private MR implementation //
	///////////////////////////////
	
	/**
	 * 
	 * @param ixin
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static Collection<MatrixIndexes> computeAllResultBlockIndexes( MatrixIndexes ixin,
            long rows1, long cols1, int brlen1, int bclen1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		HashSet<MatrixIndexes> ret = new HashSet<MatrixIndexes>();
		
		long nrblk2 = rows2/brlen2 + ((rows2%brlen2!=0)?1:0);
		long ncblk2 = cols2/bclen2 + ((cols2%bclen2!=0)?1:0);
		
		long row_offset = (ixin.getRowIndex()-1)*brlen1;
		long col_offset = (ixin.getColumnIndex()-1)*bclen1;
		
		if( rowwise ){
			for( long i=row_offset; i<Math.min(rows1,row_offset+brlen1); i++ )
			{
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), i, col_offset, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), i, Math.min(cols1,col_offset+bclen1)-1, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				
				
				//add first row block
				ret.add(first);
				//add blocks in between first and last
				for( int k1=(int)first.getRowIndex(); k1<=last.getRowIndex(); k1++ )
					for( int k2=1; k2<=ncblk2; k2++ )
					{
						if( (k1==first.getRowIndex() && k2<=first.getColumnIndex()) || 
							(k1==last.getRowIndex() && k2>=last.getColumnIndex() ) ){
							continue;
						}
						ret.add(new MatrixIndexes(k1,k2));
					}
				//add last row block
				ret.add(last);
				
			}
		}
		else{ //colwise
			for( long j=col_offset; j<Math.min(cols1,col_offset+bclen1); j++ )
			{
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), row_offset, j, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), Math.min(rows1,row_offset+brlen1)-1, j, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				
				if( first.getColumnIndex()==0 )
					throw new RuntimeException(first.toString());
				if( last.getColumnIndex()==0 )
					throw new RuntimeException(last.toString());
				
				//add first row block
				ret.add(first);
				//add blocks in between first and last
				for( int k1=(int)first.getColumnIndex(); k1<=last.getColumnIndex(); k1++ )
					for( int k2=1; k2<=nrblk2; k2++ )
					{
						if( (k1==first.getColumnIndex() && k2<=first.getRowIndex()) || 
							(k1==last.getColumnIndex() && k2>=last.getRowIndex() ) ){
							continue;
						}
						ret.add(new MatrixIndexes(k1,k2));
					}
				//add last row block
				ret.add(last);
				
			}
		}
		
		//System.out.println("created result block ix: "+ret.size());
		
		return ret;
	}
	
	/**
	 * 
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @param reuse 
	 * @return
	 */
	private static HashMap<MatrixIndexes, MatrixBlockDSM> createAllResultBlocks( Collection<MatrixIndexes> rix,
            long nnz, long rows1, long cols1, int brlen1, int bclen1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise, ArrayList<IndexedMatrixValue> reuse )
	{
		HashMap<MatrixIndexes, MatrixBlockDSM> ret = new HashMap<MatrixIndexes,MatrixBlockDSM>();
		long nBlocks = rix.size();
		int count = 0;
		
		//System.out.println("Reuse "+((reuse!=null)?reuse.size():0)+"/"+nBlocks);
		
		for( MatrixIndexes ix : rix )
		{
			//compute indexes
			long bi = ix.getRowIndex();
			long bj = ix.getColumnIndex();
			int lbrlen = (int) Math.min(brlen2, rows2-(bi-1)*brlen2);
			int lbclen = (int) Math.min(bclen2, cols2-(bj-1)*bclen2);
			
			//create result block
			int estnnz = (int) (nnz/nBlocks); //force initialcapacity per row to 1, for many blocks
			boolean sparse = MatrixBlockDSM.evalSparseFormatInMemory(lbrlen, lbclen, estnnz);
			MatrixBlockDSM block = null;
			if( ALLOW_BLOCK_REUSE && reuse!=null && reuse.size()>0) {
				block = (MatrixBlockDSM) reuse.get(count++).getValue();
				block.reset(lbrlen, lbclen, sparse, estnnz);
			}
			else
				block = new MatrixBlockDSM(lbrlen, lbclen, sparse, estnnz); 
			
			//System.out.println("create block ("+bi+","+bj+"): "+lbrlen+" "+lbclen);
			//if( lbrlen<1 || lbclen<1 )
			//	throw new RuntimeException("Computed block dimensions ("+bi+","+bj+" -> "+lbrlen+","+lbclen+") are invalid!");
			
			ret.put(ix, block);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param row_offset
	 * @param col_offset
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 */
	private static void reshapeDense( MatrixBlockDSM in, long row_offset, long col_offset, 
			HashMap<MatrixIndexes,MatrixBlockDSM> rix,
            long rows1, long cols1, 
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
    {
		int rlen = in.rlen;
		int clen = in.clen;
		double[] a = in.denseBlock;
		
		//append all values to right blocks
		MatrixIndexes ixtmp = new MatrixIndexes();
		for( int i=0, aix=0; i<rlen; i++, aix+=clen )
		{
			long ai = row_offset+i;
			for( int j=0; j<clen; j++ )
			{
				double val = a[ aix+j ];
				if( val !=0 ) {
					long aj = col_offset+j;
					computeResultBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					MatrixBlockDSM out = rix.get(ixtmp);
					computeInBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), val);
				}
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise )
		{
			for( MatrixBlockDSM block : rix.values() )
				if( block.sparse )
					block.sortSparseRows();
		}				
    }

	/**
	 * 
	 * @param in
	 * @param row_offset
	 * @param col_offset
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 */
	private static void reshapeSparse( MatrixBlockDSM in, long row_offset, long col_offset, 
			HashMap<MatrixIndexes,MatrixBlockDSM> rix,
            long rows1, long cols1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
    {
		int rlen = in.rlen;
		SparseRow[] aRows = in.sparseRows;
		
		//append all values to right blocks
		MatrixIndexes ixtmp = new MatrixIndexes();
		for( int i=0; i<rlen; i++ )
		{
			SparseRow arow = aRows[i];
			if( arow!=null && arow.size()>0 ) {
				long ai = row_offset+i;
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				for( int j=0; j<alen; j++ ) 
				{
					long aj = col_offset+aix[j];
					computeResultBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					MatrixBlockDSM out = rix.get(ixtmp);
					computeInBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), avals[j]);
				}
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise )
		{
			for( MatrixBlockDSM block : rix.values() )
				if( block.sparse )
					block.sortSparseRows();
		}				
    }
	
	/**
	 * Assumes internal (0-begin) indices ai, aj as input; computes external block indexes (1-begin) 
	 * 
	 * @param ixout
	 * @param ai
	 * @param aj
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static MatrixIndexes computeResultBlockIndex( MatrixIndexes ixout, long ai, long aj,
			            long rows1, long cols1, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		long ci, cj, tempc;
		int bci, bcj;
		
		if( rowwise ) {
			tempc = ai*cols1+aj;
			ci = tempc/cols2;
			cj = tempc%cols2;
			bci = (int) ci/brlen2 + 1;
			bcj = (int) cj/bclen2 + 1;
		}
		else { //colwise
			tempc = ai+rows1*aj;
			ci = tempc%rows2;
			cj = tempc/rows2;
			bci = (int) ci/brlen2 + 1;
			bcj = (int) cj/bclen2 + 1;			
		}
		
		//System.out.println("result block ix "+bci+" "+bcj);
		//if( bci<1 || bcj<1 )
		//	throw new RuntimeException("Computed block indexes ("+ai+","+aj+" -> "+bci+","+bcj+") are invalid!");
		//if( bci>Math.ceil(((double)rows2)/brlen2) || bcj>Math.ceil(((double)cols2)/bclen2) )
		//	throw new RuntimeException("Computed block indexes ("+ai+","+aj+" -> "+bci+","+bcj+") are invalid!");
		
		ixout.setIndexes(bci, bcj);	
		return ixout;
	}
	
	/**
	 * 
	 * @param ixout
	 * @param ai
	 * @param aj
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static MatrixIndexes computeInBlockIndex( MatrixIndexes ixout, long ai, long aj,
            long rows1, long cols1, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		long ci, cj, tempc;
		
		if( rowwise ) {
			tempc = ai*cols1+aj;
			ci = (tempc/cols2)%brlen2;
			cj = (tempc%cols2)%bclen2;
		}
		else { //colwise
			tempc = ai+rows1*aj; 
			ci = (tempc%rows2)%brlen2;
			cj = (tempc/rows2)%bclen2;
		}
		
		//System.out.println("ai/aj: "+ai+"/"+aj+"  ->  ci/cj: "+ci+"/"+cj);

		ixout.setIndexes(ci, cj);	
		return ixout;
	}
}
