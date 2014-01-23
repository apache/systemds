/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final boolean LOW_LEVEL_OPTIMIZATION = true;

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
		//Timing time = new Timing(true);
		
		if(!m1.sparse && !m2.sparse)
			matrixMultDenseDense(m1, m2, ret);
		else if(m1.sparse && m2.sparse)
			matrixMultSparseSparse(m1, m2, ret);
		else if(m1.sparse)
			matrixMultSparseDense(m1, m2, ret);
		else
			matrixMultDenseSparse(m1, m2, ret);
		
		//System.out.println("MM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+")x" +
		//		              "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
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
		//Timing time = new Timing(true);
		
		if( m1.sparse )
			matrixMultTransposeSelfSparse(m1, ret, leftTranspose);
		else 
			matrixMultTransposeSelfDense(m1, ret, leftTranspose);

		//System.out.println("TSMM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());

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

		if( LOW_LEVEL_OPTIMIZATION )
		{
			if( m==1 && n==1 ) //DOT PRODUCT
			{
				c[0] = dotProduct(a, b, cd);
			}
			else if( n==1 ) //MATRIX-VECTOR
			{
				for( int i=0, aix=0; i < m; i++, aix+=cd) 
					c[ i ] = dotProduct(a, b, aix, 0, cd);	
			}
			else //MATRIX-MATRIX
			{	
				//1) Unrolled inner loop (for better instruction-level parallelism)
				//2) Blocked execution (for less cache trashing in parallel exec) 	
				//3) Asymmetric block sizes (for less misses in inner loop, yet blocks in L1/L2)
				
				final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 
				final int blocksizeK = 32; //64//256KB b block (typical L2 size per core), used while read 512B of a / read/write 4KB of c 
				final int blocksizeJ = 1024; //512//4KB (typical main-memory page size), for scan 

				//blocked execution
				for( int bi = 0; bi < m; bi+=blocksizeI )
					for( int bk = 0, bimin = Math.min(m, bi+blocksizeI); bk < cd; bk+=blocksizeK ) 
						for( int bj = 0, bkmin = Math.min(cd, bk+blocksizeK); bj < n; bj+=blocksizeJ ) 
						{
							int bjlen = Math.min(n, bj+blocksizeJ)-bj;
							//core sub block matrix multiplication
				    		for( int i = bi; i < bimin; i++) 
				    		{
				    			int cixj = i * n + bj; //re-init scan index on c
								for( int k = bk, aix=i*cd; k < bkmin; k++)
								{
									double val = a[ aix+k ]; 									
									if( val != 0 ) //skip row if applicable
									{
										int bixj = k * n + bj; //re-init scan index on b
										vectMultiplyAdd(val, b, c, bixj, cixj, bjlen);
									}
								}	
				    		}
						}
			}
		}
		else
		{
			double val;
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
	
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if( LOW_LEVEL_OPTIMIZATION )
		{
			for( int i=0, aix=0, cix=0; i < m; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					double val = a[aix];
					if( val != 0 )
					{
						SparseRow brow = m2.sparseRows[ k ];
						if( brow != null && brow.size() > 0 ) 
						{
							int blen = brow.size();
							int[] bix = brow.getIndexContainer();
							double[] bvals = brow.getValueContainer();								
							vectMultiplyAdd(val, bvals, c, bix, cix, blen);
						}
					}
				}	
		}
		else
		{
			for( int i=0, aix=0, cix=0; i < m; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					double val = a[aix];
					if( val!=0 )
					{
						SparseRow brow = m2.sparseRows[ k ];
						if( brow != null && brow.size() > 0 ) 
						{
							int blen = brow.size();
							int[] bix = brow.getIndexContainer();
							double[] bvals = brow.getValueContainer();	
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

		if( LOW_LEVEL_OPTIMIZATION )
		{
		
			if( m==1 && n==1 ) //DOT PRODUCT
			{
				SparseRow arow = m1.sparseRows[0];
				if( arow != null && arow.size() > 0 )
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					
					c[0] = dotProduct(avals, b, aix, 0, alen);
				}
			}
			else if( n==1 ) //MATRIX-VECTOR
			{
				for( int i=0; i<Math.min(m, m1.sparseRows.length); i++ )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
					
						c[i] = dotProduct(avals, b, aix, 0, alen);							
					}
				}
			}
			else //MATRIX-MATRIX
			{
				int bix;
				for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
						
						for(int k = 0; k < alen; k++) 
						{
							double val = avals[k];
							bix = aix[k]*n;
							
							vectMultiplyAdd(val, b, c, bix, cix, n);
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
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						double val = avals[k];
						for(int j = 0, bix=aix[k]*n; j < n; j++)
							c[cix+j] += val * b[bix+j];								
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
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if(LOW_LEVEL_OPTIMIZATION)
		{
			for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && arow.size() > 0 ) 
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						double val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && brow.size() > 0 ) 
						{
							int blen = brow.size();
							int[] bix = brow.getIndexContainer();
							double[] bvals = brow.getValueContainer();	
							
							vectMultiplyAdd(val, bvals, c, bix, cix, blen);
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
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						double val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && brow.size() > 0 ) 
						{
							int blen = brow.size();
							int[] bix = brow.getIndexContainer();
							double[] bvals = brow.getValueContainer();	
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
		
		if( leftTranspose ) // t(X)%*%X
		{
			if( LOW_LEVEL_OPTIMIZATION )
			{
				if( n==1 ) //VECTOR 
				{
					c[0] = dotProduct(a, a, m);
				}
				else //MATRIX
				{	
					//algorithm: scan a once (t(a)), foreach val: scan row of a and row of c (KIJ)
				
					//1) Unrolled inner loop, for better ILP
					//2) Blocked execution, for less cache trashing in parallel exec 					
					int blocksizeI = 16;
					int blocksizeJ = 256;

					for( int bi = 0; bi<n; bi+=blocksizeI )
						for( int bj = bi; bj<n; bj+=blocksizeJ )
						{
							final int bimin = Math.min(n, bi+blocksizeI);
							final int bjmin = Math.min(n, bj+blocksizeJ);
							
							for(int k = 0, ix1 = 0; k < m; k++, ix1+=n)
								for(int i = bi, ix3 = bi*n; i < bimin; i++, ix3+=n) 
								{
									double val = a[ ix1+i ];
									if( val != 0 )
									{
										//from i due to symmetry
										int bjmax = Math.max(i,bj);
										vectMultiplyAdd(val, a, c, ix1+bjmax, ix3+bjmax, bjmin-bjmax);
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
						double val = a[ ix1+i ];
						if( val != 0 )
						{
							for(int j = i; j < n; j++) //from i due to symmetry
								c[ ix3+j ]  += val * a[ ix1+j ];
						}
					}
			}
		}
		else // X%*%t(X)
		{
			if(LOW_LEVEL_OPTIMIZATION)
			{
				if( m==1 ) //VECTOR
				{
					c[0] = dotProduct(a, a, n);
				}
				else //MATRIX
				{
					//algorithm: scan c, foreach ci,j: scan row of a and t(a) (IJK)				
				
					//1) Unrolled inner loop, for better ILP
					//2) Blocked execution, for less cache trashing in parallel exec 
					//   (smaller block sizes would be slightly better, but consistent as is)
					//3) Single write in inner loop (transient intermediates)
					int blocksize = 64;
					for( int bi = 0; bi<m; bi+=blocksize )
						for( int bj = bi; bj<m; bj+=blocksize ) 
						{
							final int bimin = Math.min(m, bi+blocksize);
							final int bjmin = Math.min(m, bj+blocksize);	
							
							for(int i = bi, ix1 = bi*n, ix3 = bi*m; i < bimin; i++, ix1+=n, ix3+=m)
							{
								final int bjmax = Math.max(i,bj);
								for(int j = bjmax, ix2 = bjmax*n; j <bjmin; j++, ix2+=n) //from i due to symmetry
								{
									c[ ix3+j ] = dotProduct(a, a, ix1, ix2, n);	
								}
							}
						}
				}
			}
			else
			{
				for(int i = 0, ix1 = 0, ix3 = 0; i < m; i++, ix1+=n, ix3+=m)
					for(int j = i, ix2 = i*n; j < m; j++, ix2+=n) //from i due to symmetry
					{
						double val = 0;
						for(int k = 0; k < n; k++)
							val += a[ ix1+k ] * a[ix2+k];
						c[ ix3+j ] = val;	
					}
			}
		}

		//3) copy symmetric values
		copyUpperToLowerTriangle( ret );
		
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

		if( leftTranspose ) // t(X)%*%X 
		{
			//only general case (because vectors always dense)
			//algorithm: scan rows, foreach row self join (KIJ)
			if( LOW_LEVEL_OPTIMIZATION )
			{
				for( SparseRow arow : m1.sparseRows )
					if( arow != null && arow.size() > 0 ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
						
						for(int i = 0; i < alen; i++) 
						{
							double val = avals[i];
							if( val != 0 )
							{
								int ix2 = aix[i]*n;
								vectMultiplyAdd(val, avals, c, aix, ix2, alen);
							}
						}
					}	
			}
			else
			{
				for( SparseRow arow : m1.sparseRows )
					if( arow != null && arow.size() > 0 ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
						
						for(int i = 0; i < alen; i++) 
						{
							double val = avals[i];
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
					int alen = arow.size();
					double[] avals = arow.getValueContainer();	
					c[0] = dotProduct(avals, avals, alen);
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
					for( SparseRow arow : tmpBlock.sparseRows )
						if( arow != null && arow.size() > 0 ) 
						{
							int alen = arow.size();
							int[] aix = arow.getIndexContainer();
							double[] avals = arow.getValueContainer();					
							
							for(int i = 0; i < alen; i++) 
							{
								double val = avals[i];
								if( val != 0 )
								{
									int ix2 = aix[i]*m;
									vectMultiplyAdd(val, avals, c, aix, i, ix2, alen);
								}
							}
						}
				}
				else
				{
					for( SparseRow arow : tmpBlock.sparseRows )
						if( arow != null && arow.size() > 0 ) 
						{
							int alen = arow.size();
							int[] aix = arow.getIndexContainer();
							double[] avals = arow.getValueContainer();					
							
							for(int i = 0; i < alen; i++) 
							{
								double val = avals[i];
								if( val != 0 )
									for(int j = i, ix2 = aix[i]*m; j < alen; j++)
										c[ix2+aix[j]] += val * avals[j];
							}
						}
				}
			}
		}
	
		//3) copy symmetric values
		copyUpperToLowerTriangle( ret );
		
		ret.recomputeNonZeros(); 
		ret.examSparsity();	
	}
	
	////////////////////////////////////////////
	// performance-relevant utility functions //
	////////////////////////////////////////////
	
	/**
	 * Computes the dot-product of two vectors. Experiments (on long vectors of
	 * 10^7 values) showed that this generic function provides equivalent performance
	 * even for the specific case of dotProduct(a,a,len) as used for TSMM.  
	 * 
	 * @param a
	 * @param b
	 * @param len
	 * @return
	 */
	private static double dotProduct( double[] a, double[] b, int len )
	{
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = 0; i < bn; i++ )
			val += a[ i ] * b[ i ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=8 )
		{
			//read 64B cachelines of a and b
			//compute cval' = sum(a * b) + cval
			val += a[ i+0 ] * b[ i+0 ]
			     + a[ i+1 ] * b[ i+1 ]
			     + a[ i+2 ] * b[ i+2 ]
			     + a[ i+3 ] * b[ i+3 ]
			     + a[ i+4 ] * b[ i+4 ]
			     + a[ i+5 ] * b[ i+5 ]
			     + a[ i+6 ] * b[ i+6 ]
			     + a[ i+7 ] * b[ i+7 ];
		}
		
		//scalar result
		return val; 
	}
	
	/**
	 * 
	 * @param a
	 * @param b
	 * @param ai
	 * @param bi
	 * @param len
	 * @return
	 */
	private static double dotProduct( double[] a, double[] b, int ai, int bi, int len )
	{
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = 0; i < bn; i++, ai++, bi++ )
			val += a[ ai ] * b[ bi ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=8, ai+=8, bi+=8 )
		{
			//read 64B cachelines of a and b
			//compute cval' = sum(a * b) + cval
			val += a[ ai+0 ] * b[ bi+0 ]
			     + a[ ai+1 ] * b[ bi+1 ]
			     + a[ ai+2 ] * b[ bi+2 ]
			     + a[ ai+3 ] * b[ bi+3 ]
			     + a[ ai+4 ] * b[ bi+4 ]
			     + a[ ai+5 ] * b[ bi+5 ]
			     + a[ ai+6 ] * b[ bi+6 ]
			     + a[ ai+7 ] * b[ bi+7 ];
		}
		
		//scalar result
		return val; 
	}
	
	private static double dotProduct( double[] a, double[] b, int[] aix, int bi, int len )
	{
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = 0; i < bn; i++ )
			val += a[ i ] * b[ bi+aix[i] ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=8 )
		{
			//read 64B cacheline of a
			//read 64B of b via 'gather'
			//compute cval' = sum(a * b) + cval
			val += a[ i+0 ] * b[ bi+aix[i+0] ]
			     + a[ i+1 ] * b[ bi+aix[i+1] ]
			     + a[ i+2 ] * b[ bi+aix[i+2] ]
			     + a[ i+3 ] * b[ bi+aix[i+3] ]
			     + a[ i+4 ] * b[ bi+aix[i+4] ]
			     + a[ i+5 ] * b[ bi+aix[i+5] ]
			     + a[ i+6 ] * b[ bi+aix[i+6] ]
			     + a[ i+7 ] * b[ bi+aix[i+7] ];
		}
		
		//scalar result
		return val; 
	}

	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bi
	 * @param ci
	 * @param len
	 */
	private static void vectMultiplyAdd( double aval, double[] b, double[] c, int bi, int ci, int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi++, ci++)
			c[ ci ] += aval * b[ bi ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi+=8, ci+=8) 
		{
			//read 64B cachelines of b and c
			//compute c' = aval * b + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += aval * b[ bi+0 ];
			c[ ci+1 ] += aval * b[ bi+1 ];
			c[ ci+2 ] += aval * b[ bi+2 ];
			c[ ci+3 ] += aval * b[ bi+3 ];
			c[ ci+4 ] += aval * b[ bi+4 ];
			c[ ci+5 ] += aval * b[ bi+5 ];
			c[ ci+6 ] += aval * b[ bi+6 ];
			c[ ci+7 ] += aval * b[ bi+7 ];
		}
	}
	
	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bix
	 * @param ci
	 * @param len
	 */
	private static void vectMultiplyAdd( double aval, double[] b, double[] c, int[] bix, int ci, int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++ )
			c[ ci + bix[j] ] += aval * b[ j ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8 )
		{
			//read 64B cacheline of b
			//read 64B of c via 'gather'
			//compute c' = aval * b + c
			//write back 64B of c = c' via 'scatter'
			c[ ci+bix[j+0] ] += aval * b[ j+0 ];
			c[ ci+bix[j+1] ] += aval * b[ j+1 ];
			c[ ci+bix[j+2] ] += aval * b[ j+2 ];
			c[ ci+bix[j+3] ] += aval * b[ j+3 ];
			c[ ci+bix[j+4] ] += aval * b[ j+4 ];
			c[ ci+bix[j+5] ] += aval * b[ j+5 ];
			c[ ci+bix[j+6] ] += aval * b[ j+6 ];
			c[ ci+bix[j+7] ] += aval * b[ j+7 ];
		}
	}
	
	private static void vectMultiplyAdd( double aval, double[] b, double[] c, int[] bix, int bi, int ci, int len )
	{
		final int bn = (len-bi)%8;
		
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			c[ ci + bix[j] ] += aval * b[ j ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < len; j+=8 )
		{
			//read 64B cacheline of b
			//read 64B of c via 'gather'
			//compute c' = aval * b + c
			//write back 64B of c = c' via 'scatter'
			c[ ci+bix[j+0] ] += aval * b[ j+0 ];
			c[ ci+bix[j+1] ] += aval * b[ j+1 ];
			c[ ci+bix[j+2] ] += aval * b[ j+2 ];
			c[ ci+bix[j+3] ] += aval * b[ j+3 ];
			c[ ci+bix[j+4] ] += aval * b[ j+4 ];
			c[ ci+bix[j+5] ] += aval * b[ j+5 ];
			c[ ci+bix[j+6] ] += aval * b[ j+6 ];
			c[ ci+bix[j+7] ] += aval * b[ j+7 ];
		}
	}
	
	
	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 * 
	 * @param ret
	 */
	private static void copyUpperToLowerTriangle( MatrixBlockDSM ret )
	{
		double[] c = ret.denseBlock;
		final int m = ret.rlen;
		final int n = ret.clen;
		
		//copy symmetric values
		for( int i=0, uix=0; i<m; i++, uix+=n )
			for( int j=i+1, lix=j*n+i; j<n; j++, lix+=n )
			{
				c[ lix ] = c[ uix+j ];
				//c[ j*ret.clen+i ] = c[i*ret.clen+j];
			}
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
