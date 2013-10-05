/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

/**
 * MB:
 * Library for matrix multiplications including MM, MV, VV for all
 * combinations of dense and sparse representations and special
 * operations such as transpose-self matrix multiplication.
 * 
 * In general all implementations use internally dense outputs
 * for direct access, but change the final result to sparse if necessary.  
 * 
 * NOTES on BLAS:
 * * Experiments in 04/2013 showed that even on dense-dense this implementation 
 *   is 3x faster than f2j-BLAS-DGEMM, 2x faster than f2c-BLAS-DGEMM, and
 *   level (+10% after JIT) with a native C implementation. 
 * * Calling native BLAS would loose platform independence and would require 
 *   JNI calls incl data transfer (and changing the representation from row-
 *   to column-major representation). Furthermore, BLAS does not support sparse 
 *   matrices and would be an external dependency. 
 * 
 *  
 */
public class MatrixMultLib 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static boolean LOW_LEVEL_OPTIMIZATION = true;

	////////////////////////////////
	// public matrix mult interface
	////////////////////////////////
	
	/**
	 * Performs a matrix multiplication and stores the result in the resulting matrix.
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 *  
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @throws DMLRuntimeException 
	 */
	public static void matrixMult(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException
	{		
		//Timing time = new Timing();
		//time.start();
		
		if(!m1.sparse && !m2.sparse)
			matrixMultDenseDense(m1, m2, ret);
		else if(m1.sparse && m2.sparse)
			matrixMultSparseSparse(m1, m2, ret);
		else if(m1.sparse)
			matrixMultSparseDense(m1, m2, ret);
		else
			matrixMultDenseSparse(m1, m2, ret);
		
		//System.out.println("MM in "+time.stop());
	}


	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static void matrixMultTransposeSelf( MatrixBlockDSM m1, MatrixBlockDSM ret, boolean leftTranspose )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//Timing time = new Timing();
		//time.start();
		
		if( m1.sparse )
			matrixMultTransposeSelfSparse(m1, ret, leftTranspose);
		else 
			matrixMultTransposeSelfDense(m1, ret, leftTranspose);

		//System.out.println("TSMM in "+time.stop());
	}
	
	
	//////////////////////////////////////////
	// optimized matrix mult implementation //
	//////////////////////////////////////////
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultDenseDense(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException
	{	
		//check inputs / outputs
		if( m1.denseBlock==null || m2.denseBlock==null )
			return;
		ret.sparse=false;
		if( ret.denseBlock==null )
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		
		double[] a = m1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m1.clen;
					
		if( m==1 && n==1 ) //DOT PRODUCT
		{
			for( int i=0; i<cd; i++ )
				c[0] += a[i] * b[i];
			ret.nonZeros = (c[0]!=0) ? 1 : 0;
		}
		else if( n==1 ) //MATRIX-VECTOR
		{
			if( LOW_LEVEL_OPTIMIZATION )
			{
				int bcd = cd%8;
				for( int i = 0, aix=0, cix=0; i < m; i++, cix++) 
				{
					//compute rest
					for( int k = 0; k < bcd; k++, aix++)
						c[ cix ] += a[ aix ] * b[ k ];
					//unrolled 8-block 
					for( int k = bcd; k < cd; k+=8, aix+=8)
					{
						c[ cix ] += a[ aix ]   * b[ k   ];
						c[ cix ] += a[ aix+1 ] * b[ k+1 ];
						c[ cix ] += a[ aix+2 ] * b[ k+2 ];
						c[ cix ] += a[ aix+3 ] * b[ k+3 ];
						c[ cix ] += a[ aix+4 ] * b[ k+4 ];
						c[ cix ] += a[ aix+5 ] * b[ k+5 ];
						c[ cix ] += a[ aix+6 ] * b[ k+6 ];
						c[ cix ] += a[ aix+7 ] * b[ k+7 ];
					}				
				}
			}
			else
			{
				for( int i = 0, aix=0, cix=0; i < m; i++, cix++) 
					for( int k = 0; k < cd; k++, aix++)
						c[ cix ] += a[ aix ] * b[ k ];				
			}
			ret.recomputeNonZeros();
		}
		else //MATRIX-MATRIX
		{	
			double val; 
			if( LOW_LEVEL_OPTIMIZATION )
			{
				int bn = n%8;
				for( int i = 0, aix=0, cix=0; i < m; i++, cix+=n) 
					for( int k = 0, bix=0; k < cd; k++, aix++, bix+=n)
					{			
						val = a[ aix ];
						if( val != 0 )
						{
							//rest, not aligned to 8-blocks
							for( int j = 0; j < bn; j++)
								c[ cix+j ] += val * b[ bix+j ];
							//unrolled 8-block
							for( int j=bn, jix1=cix+bn, jix2=bix+bn; j < n; j+=8, jix1+=8, jix2+=8 ) 
							{
								c[ jix1   ] += val * b[ jix2   ];
								c[ jix1+1 ] += val * b[ jix2+1 ];
								c[ jix1+2 ] += val * b[ jix2+2 ];
								c[ jix1+3 ] += val * b[ jix2+3 ];
								c[ jix1+4 ] += val * b[ jix2+4 ];
								c[ jix1+5 ] += val * b[ jix2+5 ];
								c[ jix1+6 ] += val * b[ jix2+6 ];
								c[ jix1+7 ] += val * b[ jix2+7 ];
							}
						}
					}	
			}
			else
			{
				for( int i = 0, aix=0, cix=0; i < m; i++, cix+=n) 
					for( int k = 0, bix=0; k < cd; k++, aix++, bix+=n)
					{			
						val = a[ aix ];
						if( val != 0 )
							for( int j = 0; j < n; j++) 
								c[ cix+j ] += val * b[ bix+j ];
					}				
			}
			ret.recomputeNonZeros();
			
			
			//opt version with blocking 
			/*
			int blocksize1 = 32;
			int blocksize2 = 32;
			int blocksize3 = 500;
			double val = 0;	
			for( int bi = 0; bi < m; bi+=blocksize1) 
			{
				int maxI = Math.min(m,bi+blocksize1);
				for( int bk = 0; bk < cd; bk+=blocksize2)
				{
					int maxK = Math.min(cd,bk+blocksize2);
					for( int bj = 0; bj < n; bj+=blocksize3) 
					{	
						int maxJ = Math.min(n,bj+blocksize3);
						for( int i = bi, aix=bi*cd, cix=bi*n; i < maxI; i++, aix+=cd, cix+=n) 
							for( int k = bk, bix=bk*n; k < maxK; k++, bix+=n)
							{			
								val = a[ aix+k ];
								if( val != 0 )
									for( int j = bj; j < maxJ; j++) 
										c[ cix+j ] += val * b[ bix+j ];
							}	
					}
				}
			}
			*/		
		}		
		ret.examSparsity();
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultDenseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException 
	{
		//check inputs / outputs
		if( m1.denseBlock==null || m2.sparseRows==null  )
			return;
		ret.sparse=false;
		if( ret.denseBlock==null )
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		
		double[] a = m1.denseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m2.clen;
		int cd = m1.clen;
		
		double val;
		int blen;
		int[] bix;
		double[] bvals;
	
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if( LOW_LEVEL_OPTIMIZATION )
		{
			for( int i=0, aix=0, cix=0; i < m; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					val = a[aix];
					if( val != 0 )
					{
						SparseRow brow = m2.sparseRows[ k ];
						if( brow != null && brow.size() > 0 ) 
						{
							blen = brow.size();
							bix = brow.getIndexContainer();
							bvals = brow.getValueContainer();	
							int bblen = blen%8;
							//rest, not aligned to 8-blocks
							for( int j = 0; j < bblen; j++)
								c[cix+bix[j]] += val * bvals[j];
							//unrolled 8-block 
							for(int j = bblen; j < blen; j+=8)
							{
								c[cix+bix[j]]   += val * bvals[j];
								c[cix+bix[j+1]] += val * bvals[j+1];
								c[cix+bix[j+2]] += val * bvals[j+2];
								c[cix+bix[j+3]] += val * bvals[j+3];
								c[cix+bix[j+4]] += val * bvals[j+4];
								c[cix+bix[j+5]] += val * bvals[j+5];
								c[cix+bix[j+6]] += val * bvals[j+6];
								c[cix+bix[j+7]] += val * bvals[j+7];
							}
						}
					}
				}	
		}
		else
		{
			for( int i=0, aix=0, cix=0; i < m; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					val = a[aix];
					if( val!=0 )
					{
						SparseRow brow = m2.sparseRows[ k ];
						if( brow != null && brow.size() > 0 ) 
						{
							blen = brow.size();
							bix = brow.getIndexContainer();
							bvals = brow.getValueContainer();	
							for(int j = 0; j < blen; j++)
								c[cix+bix[j]] += val * bvals[j];								
						}
					}
				}		
		}
		ret.recomputeNonZeros();
		ret.examSparsity();	
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultSparseDense(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException
	{
		//check inputs / outputs
		if( m1.sparseRows==null || m2.denseBlock==null )
			return;	
		ret.sparse=false;
		if(ret.denseBlock==null)
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m2.clen;
		
		double val;
		int alen;
		int[] aix;
		double[] avals;
		
		if( m==1 && n==1 ) //DOT PRODUCT
		{
			SparseRow arow = m1.sparseRows[0];
			if( arow != null && arow.size() > 0 )
			{
				alen = arow.size();
				aix = arow.getIndexContainer();
				avals = arow.getValueContainer();
				
				for(int k = 0; k < alen; k++) 
					c[0] += avals[k] * b[aix[k]];
			}
			ret.nonZeros = (c[0]!=0) ? 1 : 0;
		}
		else if( n==1 ) //MATRIX-VECTOR
		{
			if( LOW_LEVEL_OPTIMIZATION )
			{
				int balen;
				for( int i=0; i<Math.min(m, m1.sparseRows.length); i++ )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						balen = alen%8;
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						//compute rest
						for( int k = 0; k < balen; k++)
							c[i] += avals[k] * b[aix[k]];	
						//unrolled 8-block 
						for(int k = balen; k < alen; k+=8) 
						{
							c[i] += avals[k]   * b[aix[k]];
							c[i] += avals[k+1] * b[aix[k+1]];
							c[i] += avals[k+2] * b[aix[k+2]];
							c[i] += avals[k+3] * b[aix[k+3]];
							c[i] += avals[k+4] * b[aix[k+4]];
							c[i] += avals[k+5] * b[aix[k+5]];
							c[i] += avals[k+6] * b[aix[k+6]];
							c[i] += avals[k+7] * b[aix[k+7]];
						}
					}
				}
			}
			else
			{
				for( int i=0; i<Math.min(m, m1.sparseRows.length); i++ )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						for(int k = 0; k < alen; k++) 
							c[i] += avals[k] * b[aix[k]];														
					}
				}
			}
			ret.recomputeNonZeros();			
		}
		else //MATRIX-MATRIX
		{
			if(LOW_LEVEL_OPTIMIZATION)
			{
				int bn=n%8;
				int bix;
				for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						for(int k = 0; k < alen; k++) 
						{
							val = avals[k];
							bix = aix[k]*n;
							//compute rest
							for(int j = 0; j < bn; j++)
								c[cix+j] += val * b[bix+j];
							//unrolled 8-block 
							for(int j=bn, jix1=cix+bn, jix2=bix+bn; j < n; j+=8, jix1+=8, jix2+=8 )
							{
								c[ jix1   ] += val * b[ jix2   ];
								c[ jix1+1 ] += val * b[ jix2+1 ];
								c[ jix1+2 ] += val * b[ jix2+2 ];
								c[ jix1+3 ] += val * b[ jix2+3 ];
								c[ jix1+4 ] += val * b[ jix2+4 ];
								c[ jix1+5 ] += val * b[ jix2+5 ];
								c[ jix1+6 ] += val * b[ jix2+6 ];
								c[ jix1+7 ] += val * b[ jix2+7 ];
							}
						}						
					}
				}				
			}
			else
			{
				for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						for(int k = 0; k < alen; k++) 
						{
							val = avals[k];
							for(int j = 0, bix=aix[k]*n; j < n; j++)
								c[cix+j] += val * b[bix+j];								
						}						
					}
				}
			}
			ret.recomputeNonZeros();
		}
		ret.examSparsity();
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultSparseSparse(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException
	{
		//check inputs / outputs
		if( m1.sparseRows==null || m2.sparseRows==null )
			return;	
		ret.sparse=false;
		if(ret.denseBlock==null)
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m2.clen;
		
		double val;
		int alen, blen;
		int[] aix, bix;
		double[] avals, bvals;
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if(LOW_LEVEL_OPTIMIZATION)
		{
			int bblen;
			for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && arow.size() > 0 ) 
				{
					alen = arow.size();
					aix = arow.getIndexContainer();
					avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && brow.size() > 0 ) 
						{
							blen = brow.size();
							bblen = blen%8;
							bix = brow.getIndexContainer();
							bvals = brow.getValueContainer();	
							//compute rest
							for(int j = 0; j < bblen; j++)
								c[cix+bix[j]] += val * bvals[j];
							//unrolled 8-block 
							for(int j = bblen; j < blen; j+=8)
							{
								c[cix+bix[j]]   += val * bvals[j];
								c[cix+bix[j+1]] += val * bvals[j+1];
								c[cix+bix[j+2]] += val * bvals[j+2];
								c[cix+bix[j+3]] += val * bvals[j+3];
								c[cix+bix[j+4]] += val * bvals[j+4];
								c[cix+bix[j+5]] += val * bvals[j+5];
								c[cix+bix[j+6]] += val * bvals[j+6];
								c[cix+bix[j+7]] += val * bvals[j+7];
							}
						}
					}						
				}
			}
		}
		else
		{
			for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && arow.size() > 0 ) 
				{
					alen = arow.size();
					aix = arow.getIndexContainer();
					avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && brow.size() > 0 ) 
						{
							blen = brow.size();
							bix = brow.getIndexContainer();
							bvals = brow.getValueContainer();	
							for(int j = 0; j < blen; j++)
								c[cix+bix[j]] += val * bvals[j];								
						}
					}						
				}
			}
		}
		ret.recomputeNonZeros();
		ret.examSparsity();
	}


	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultTransposeSelfDense( MatrixBlockDSM m1, MatrixBlockDSM ret, boolean leftTranspose ) 
		throws DMLRuntimeException
	{
		//1) allocate output block
		ret.rlen = leftTranspose ? m1.clen : m1.rlen;
		ret.clen = leftTranspose ? m1.clen : m1.rlen;
		ret.sparse = false;
		if(ret.denseBlock==null)
			ret.denseBlock = new double[ret.rlen * ret.clen]; 
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		if( m1.denseBlock == null )
			return;
		
		//2) transpose self matrix multiply dense
		// (compute only upper-triangular matrix due to symmetry)
		double[] a = m1.denseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m1.clen;
		
		double val;
		if( leftTranspose ) // t(X)%*%X
		{
			if( n==1 ) //VECTOR 
			{
				for( int i = 0; i < m; i++ )
					c[0] += a[i] * a[i];
			}
			else //MATRIX
			{	
				//algorithm: scan a once (t(a)), foreach val: scan row of a and row of c (KIJ)
				if( LOW_LEVEL_OPTIMIZATION )
				{
					//1) Unrolled inner loop, for better ILP
					//2) Blocked execution, for less cache trashing in parallel exec 					
					int blocksize = 64;
					int bn, bjmin, bjmax;
					for( int bi = 0; bi<n; bi+=blocksize )
						for( int bj = bi; bj<n; bj+=blocksize ) 
						{
							bjmin = Math.min(n, bj+blocksize);
							for(int k = 0, ix1 = 0; k < m; k++, ix1+=n)
								for(int i = bi, ix3 = bi*n; i < Math.min(n, bi+blocksize); i++, ix3+=n) 
								{
									val = a[ ix1+i ];
									if( val != 0 )
									{
										//from i due to symmetry
										bjmax = Math.max(i,bj);
										bn = (bjmin-bjmax)%8;
										//compute rest
										for(int j = bjmax; j < bjmax+bn; j++) 
											c[ ix3+j ]  += val * a[ ix1+j ];
										//unrolled 8-block
										for(int j=bjmax+bn, jix1=ix3+bjmax+bn, jix2=ix1+bjmax+bn; j < bjmin; j+=8, jix1+=8, jix2+=8) 
										{
											c[ jix1   ]  += val * a[ jix2   ];
											c[ jix1+1 ]  += val * a[ jix2+1 ];
											c[ jix1+2 ]  += val * a[ jix2+2 ];
											c[ jix1+3 ]  += val * a[ jix2+3 ];
											c[ jix1+4 ]  += val * a[ jix2+4 ];
											c[ jix1+5 ]  += val * a[ jix2+5 ];
											c[ jix1+6 ]  += val * a[ jix2+6 ];
											c[ jix1+7 ]  += val * a[ jix2+7 ];
										}
									}
								}
						}	
				}
				else
				{	
					for(int k = 0, ix1 = 0; k < m; k++, ix1+=n)
						for(int i = 0, ix3 = 0; i < n; i++, ix3+=n) 
						{
							val = a[ ix1+i ];
							if( val != 0 )
							{
								for(int j = i; j < n; j++) //from i due to symmetry
									c[ ix3+j ]  += val * a[ ix1+j ];
							}
						}
				}
			}
		}
		else // X%*%t(X)
		{
			if( m==1 ) //VECTOR
			{
				for( int i = 0; i < n; i++ )
					c[0] += a[i] * a[i];
			}
			else //MATRIX
			{
				//algorithm: scan c, foreach ci,j: scan row of a and t(a) (IJK)				
				if(LOW_LEVEL_OPTIMIZATION)
				{
					//1) Unrolled inner loop, for better ILP
					//2) Blocked execution, for less cache trashing in parallel exec 
					//   (smaller block sizes would be slightly better, but consistent as is)
					int blocksize = 64; 
					int bn = n%8;
					int bjmin, bjmax;
					for( int bi = 0; bi<m; bi+=blocksize )
						for( int bj = bi; bj<m; bj+=blocksize ) 
						{
							bjmin =  Math.min(m, bj+blocksize);							
							for(int i = bi, ix1 = bi*n, ix3 = bi*m; i < Math.min(m, bi+blocksize); i++, ix1+=n, ix3+=m)
							{
								bjmax = Math.max(i,bj);
								for(int j = bjmax, ix2 = bjmax*n; j <bjmin; j++, ix2+=n) //from i due to symmetry
								{
									val = 0;
									//compute rest
									for(int k = 0; k < bn; k++)
										val += a[ ix1+k ] * a[ix2+k];
									//unrolled 8-block
									for(int k=bn, kix1=ix1+bn, kix2=ix2+bn; k < n; k+=8, kix1+=8, kix2+=8)
									{
										val += a[ kix1   ] * a[ kix2   ];
										val += a[ kix1+1 ] * a[ kix2+1 ];
										val += a[ kix1+2 ] * a[ kix2+2 ];
										val += a[ kix1+3 ] * a[ kix2+3 ];
										val += a[ kix1+4 ] * a[ kix2+4 ];
										val += a[ kix1+5 ] * a[ kix2+5 ];
										val += a[ kix1+6 ] * a[ kix2+6 ];
										val += a[ kix1+7 ] * a[ kix2+7 ];
									}
									c[ ix3+j ] = val;	
								}
							}
						}
				}
				else
				{
					for(int i = 0, ix1 = 0, ix3 = 0; i < m; i++, ix1+=n, ix3+=m)
						for(int j = i, ix2 = i*n; j < m; j++, ix2+=n) //from i due to symmetry
						{
							val = 0;
							for(int k = 0; k < n; k++)
								val += a[ ix1+k ] * a[ix2+k];
							c[ ix3+j ] = val;	
						}
				}
			}
		}

		//3) copy symmetric values
		for( int i=0; i<ret.rlen; i++)
			for( int j=i+1; j<ret.clen; j++ )
			{
				val = c[i*ret.clen+j];
				if( val != 0 ) 
					c[ j*ret.clen+i ] = val;
			}
		ret.recomputeNonZeros();
		ret.examSparsity();	
	}
	
	/**
	 * 
	 * @param out
	 * @param leftTranspose
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultTransposeSelfSparse( MatrixBlockDSM m1, MatrixBlockDSM ret, boolean leftTranspose ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//1) allocate output block
		ret.rlen = leftTranspose ? m1.clen : m1.rlen;
		ret.clen = leftTranspose ? m1.clen : m1.rlen;
		ret.sparse = false;  //assumption dense output
		if(ret.denseBlock==null)
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		if( m1.sparseRows == null )
			return;
		
		//2) transpose self matrix multiply sparse
		// (compute only upper-triangular matrix due to symmetry)		
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m1.clen;
		
		double val;
		int alen;
		int[] aix;
		double[] avals;

		if( leftTranspose ) // t(X)%*%X 
		{
			//only general case (because vectors always dense)
			//algorithm: scan rows, foreach row self join (KIJ)
			if( LOW_LEVEL_OPTIMIZATION )
			{
				int balen;
				int ix2;
				for( SparseRow arow : m1.sparseRows )
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						for(int i = 0; i < alen; i++) 
						{
							val = avals[i];
							if( val != 0 )
							{
								balen = (alen-i)%8;
								ix2 = aix[i]*n;
								//compute rest
								for(int j = i; j < i+balen; j++)
									c[ix2+aix[j]] += val * avals[j];
								//unrolled 8-block
								for(int j = i+balen; j < alen; j+=8)
								{
									c[ix2+aix[j]]   += val * avals[j];
									c[ix2+aix[j+1]] += val * avals[j+1];
									c[ix2+aix[j+2]] += val * avals[j+2];
									c[ix2+aix[j+3]] += val * avals[j+3];
									c[ix2+aix[j+4]] += val * avals[j+4];
									c[ix2+aix[j+5]] += val * avals[j+5];
									c[ix2+aix[j+6]] += val * avals[j+6];
									c[ix2+aix[j+7]] += val * avals[j+7];
								}
							}
						}
					}	
			}
			else
			{
				for( SparseRow arow : m1.sparseRows )
					if( arow != null && arow.size() > 0 ) 
					{
						alen = arow.size();
						aix = arow.getIndexContainer();
						avals = arow.getValueContainer();					
						
						for(int i = 0; i < alen; i++) 
						{
							val = avals[i];
							if( val != 0 )
								for(int j = i, ix2 = aix[i]*n; j < alen; j++)
									c[ix2+aix[j]] += val * avals[j];
						}
					}
			}
		}
		else // X%*%t(X)
		{
			if( m==1 ) //VECTOR 
			{
				SparseRow arow = m1.sparseRows[0];
				if( arow !=null && arow.size()>0 )
				{
					alen = arow.size();
					avals = arow.getValueContainer();	
					for( int i = 0; i < alen; i++ )
						c[0] += avals[i] * avals[i];
				}
			}
			else //MATRIX
			{	
				//note: reorg to similar layout as t(X)%*%X because faster than 
				//direct computation with IJK (no dependencies/branches in inner loop)
				MatrixBlockDSM tmpBlock = new MatrixBlock(n,m,m1.sparse);
				m1.reorgOperations(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), 
						       tmpBlock, 0, 0, -1);
			
				if( tmpBlock.sparseRows == null )
					return;
				
				//algorithm: scan rows, foreach row self join (KIJ)
				if( LOW_LEVEL_OPTIMIZATION )
				{
					int balen;
					int ix2;
					for( SparseRow arow : tmpBlock.sparseRows )
						if( arow != null && arow.size() > 0 ) 
						{
							alen = arow.size();
							aix = arow.getIndexContainer();
							avals = arow.getValueContainer();					
							
							for(int i = 0; i < alen; i++) 
							{
								val = avals[i];
								if( val != 0 )
								{
									balen = (alen-i)%8;
									ix2 = aix[i]*m;
									//compute rest
									for(int j = i; j < i+balen; j++)
										c[ix2+aix[j]] += val * avals[j];
									//unrolled 8-block
									for(int j = i+balen; j < alen; j+=8)
									{
										c[ix2+aix[j]]   += val * avals[j];
										c[ix2+aix[j+1]] += val * avals[j+1];
										c[ix2+aix[j+2]] += val * avals[j+2];
										c[ix2+aix[j+3]] += val * avals[j+3];
										c[ix2+aix[j+4]] += val * avals[j+4];
										c[ix2+aix[j+5]] += val * avals[j+5];
										c[ix2+aix[j+6]] += val * avals[j+6];
										c[ix2+aix[j+7]] += val * avals[j+7];
									}
								}
							}
						}
				}
				else
				{
					for( SparseRow arow : tmpBlock.sparseRows )
						if( arow != null && arow.size() > 0 ) 
						{
							alen = arow.size();
							aix = arow.getIndexContainer();
							avals = arow.getValueContainer();					
							
							for(int i = 0; i < alen; i++) 
							{
								val = avals[i];
								if( val != 0 )
									for(int j = i, ix2 = aix[i]*m; j < alen; j++)
										c[ix2+aix[j]] += val * avals[j];
							}
						}
				}
			}
		}
	
		//3) copy symmetric values
		for( int i=0; i<ret.rlen; i++)
			for( int j=i+1; j<ret.clen; j++ )
			{
				val = c[i*ret.clen+j];
				if( val != 0 ) 
					c[ j*ret.clen+i ] = val;
			}
		ret.recomputeNonZeros(); 
		ret.examSparsity();	
	}
	
	
	
	/////////////////////////////////////////////////////
	// old matrix mult implementation (for comparison) //
	/////////////////////////////////////////////////////	
	
	/**
	 * 
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 */
	@Deprecated
	public static void matrixMultDenseDenseOld(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret)
	{	
		//check inputs / outputs
		if( m1.denseBlock==null || m2.denseBlock==null )
			return;
		ret.sparse=false;
		if( ret.denseBlock==null )
			ret.denseBlock = new double[ret.rlen * ret.clen];
		Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0);
		
		double[] a = m1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m2.clen;
		int k = m1.clen; 
		int l, i, j, aIndex, bIndex, cIndex; 
		double temp;
		
		int nnzs=0;
		for(l = 0; l < k; l++)
		{
			aIndex = l;
			cIndex = 0;
			for(i = 0; i < m; i++)
			{
				// aIndex = i * k + l => a[i, l]
				temp = a[aIndex];
				if(temp != 0)
				{
					bIndex = l * n;
					for(j = 0; j < n; j++)
					{
						// bIndex = l * n + j => b[l, j]
						// cIndex = i * n + j => c[i, j]
						if(c[cIndex]==0)
							nnzs++;
						c[cIndex] = c[cIndex] + temp * b[bIndex];
						if(c[cIndex]==0)
							nnzs--;
						cIndex++;
						bIndex++;
					}
				}else
					cIndex+=n;
				aIndex += k;
			}
		}
		ret.nonZeros=nnzs;
	}
	
}
