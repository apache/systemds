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
 *   JNI calls incl data transfer. Furthermore, BLAS does not support sparse 
 *   matrices (except Sparse BLAS, with dedicated function calls and matrix formats) 
 *   and would be an external dependency. 
 * * Experiments in 02/2014 showed that on dense-dense this implementation now achieves
 *   almost 30% peak FP performance. Compared to Intel MKL 11.1 (dgemm, N=1000) it is
 *   just 3.2x (sparsity=1.0) and 1.9x (sparsity=0.5) slower, respectively.
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
		
		if( m1.isUltraSparse() || m2.isUltraSparse() )
			matrixMultUltraSparse(m1, m2, ret);
		else if(!m1.sparse && !m2.sparse)
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		//if( m1.denseBlock==null || m2.denseBlock==null )
		//	return;
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		double[] a = m1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m1.clen;

		if( LOW_LEVEL_OPTIMIZATION )
		{
			if( m==1 && n==1 ) 		   //DOT PRODUCT
			{
				c[0] = dotProduct(a, b, cd);
			}
			else if( n>1 && cd == 1 )  //OUTER PRODUCT
			{
				for( int i=0, cix=0; i < m; i++, cix+=n) {
					if( a[i]!=0 )
						vectMultiplyWrite(a[i], b, c, 0, cix, n);
					else
						Arrays.fill(c, cix, cix+n, 0);
				}
			}
			else if( n==1 && cd == 1 ) //VECTOR-SCALAR
			{
				vectMultiplyWrite(b[0], a, c, 0, 0, m);
			}
			else if( n==1 )            //MATRIX-VECTOR
			{
				for( int i=0, aix=0; i < m; i++, aix+=cd) 
					c[ i ] = dotProduct(a, b, aix, 0, cd);	
			}
			else //MATRIX-MATRIX
			{	
				//init empty result
				Arrays.fill(c, 0, c.length, 0);
				
				//1) Unrolled inner loop (for better instruction-level parallelism)
				//2) Blocked execution (for less cache trashing in parallel exec) 	
				//3) Asymmetric block sizes (for less misses in inner loop, yet blocks in L1/L2)
				
				final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 
				final int blocksizeK = 24; //64//256KB b block (typical L2 size per core), used while read 512B of a / read/write 4KB of c 
				final int blocksizeJ = 1024; //512//4KB (typical main-memory page size), for scan 

				//temporary arrays (nnz a, b index)
				double[] ta = new double[ blocksizeK ];
				int[]  tbi  = new int[ blocksizeK ];
				
				//blocked execution
				for( int bi = 0; bi < m; bi+=blocksizeI )
					for( int bk = 0, bimin = Math.min(m, bi+blocksizeI); bk < cd; bk+=blocksizeK ) 
						for( int bj = 0, bkmin = Math.min(cd, bk+blocksizeK); bj < n; bj+=blocksizeJ ) 
						{
							int bklen = bkmin-bk;
							int bjlen = Math.min(n, bj+blocksizeJ)-bj;
							
							//core sub block matrix multiplication
				    		for( int i = bi; i < bimin; i++) 
				    		{
				    			int aixi = i * cd + bk; //start index on a
				    			int cixj = i * n + bj; //scan index on c
				    			
				    			//determine nnz of a (for sparsity-aware skipping of rows)
				    			int knnz = copyNonZeroElements(a, aixi, bk, bj, n, ta, tbi, bklen);
				    			
				    			//rest not aligned to blocks of 4 rows
				    			final int bn = knnz % 4;
				    			switch( bn ){
					    			case 1: vectMultiplyAdd(ta[0], b, c, tbi[0], cixj, bjlen); break;
					    	    	case 2: vectMultiplyAdd2(ta[0],ta[1], b, c, tbi[0], tbi[1], cixj, bjlen); break;
					    			case 3: vectMultiplyAdd3(ta[0],ta[1],ta[2], b, c, tbi[0], tbi[1],tbi[2], cixj, bjlen); break;
				    			}
				    			
				    			//compute blocks of 4 rows (core inner loop)
				    			for( int k = bn; k<knnz; k+=4 ){
				    				vectMultiplyAdd4( ta[k], ta[k+1], ta[k+2], ta[k+3], b, c, 
				    						          tbi[k], tbi[k+1], tbi[k+2], tbi[k+3], cixj, bjlen );
				    			}
				    		}
						}
			}
		}
		else
		{
			//init empty result
			Arrays.fill(c, 0, c.length, 0);
			
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		//if( m1.denseBlock==null || m2.sparseRows==null  )
		//	return;
		ret.sparse = false;
		ret.allocateDenseBlock();
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		//if( m1.sparseRows==null || m2.denseBlock==null )
		//	return;	
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;

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
				//init empty result
				Arrays.fill(c, 0, c.length, 0);
				
				for( int i=0, cix=0; i<Math.min(m, m1.sparseRows.length); i++, cix+=n )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && arow.size() > 0 ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
						
						//rest not aligned to blocks of 4 rows
		    			final int bn = alen % 4;
		    			switch( bn ){
			    			case 1: vectMultiplyAdd(avals[0], b, c, aix[0]*n, cix, n); break;
			    	    	case 2: vectMultiplyAdd2(avals[0],avals[1], b, c, aix[0]*n, aix[1]*n, cix, n); break;
			    			case 3: vectMultiplyAdd3(avals[0],avals[1],avals[2], b, c, aix[0]*n, aix[1]*n, aix[2]*n, cix, n); break;
		    			}
		    			
		    			//compute blocks of 4 rows (core inner loop)
		    			for( int k = bn; k<alen; k+=4 ) {
		    				vectMultiplyAdd4( avals[k], avals[k+1], avals[k+2], avals[k+3], b, c, 
		    						          aix[k]*n, aix[k+1]*n, aix[k+2]*n, aix[k+3]*n, cix, n );
		    			}
					}
				}					
			}
		}
		else
		{
			//init empty result
			Arrays.fill(c, 0, c.length, 0);
			
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		//if( m1.sparseRows==null || m2.sparseRows==null )
		//	return;	
		ret.sparse=false;
		ret.allocateDenseBlock();
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
	 * This implementation applies to any combination of dense/sparse if at least one
	 * input is ultrasparse (sparse and very few nnz). In that case, most importantly,
	 * we want to create a sparse output and only iterate over the few nnz as the major
	 * dimension. Low-level optimization have less importance in that case and having
	 * this generic implementation helps to reduce the implementations from (2+1)^2
	 * to 2^2+1.
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultUltraSparse(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret) 
		throws DMLRuntimeException 
	{
		//check inputs / outputs
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		ret.sparse = true;		
		ret.reset();
		
		boolean leftUS = m1.isUltraSparse();
		int m  = m1.rlen;
		int cd = m1.clen;
		int n  = m2.clen;
		
		if( leftUS ) //left is ultra-sparse (IKJ)
		{
			for( int i=0; i<m; i++ )
			{
				SparseRow arow = m1.sparseRows[ i ];
				if( arow != null && arow.size() > 0 ) 
				{
					int alen = arow.size();
					int[] aixs = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();	
					for( int k=0; k<alen; k++ )
					{
						double aval = avals[k];
						int aix = aixs[k];
						for( int j=0; j<n; j++ )
						{
							double cval = ret.quickGetValue(i, j);
							double cvald = aval*m2.quickGetValue(aix, j);
							if( cvald != 0 )
								ret.quickSetValue(i, j, cval+cvald);
						}
					}
				}
			}
		}
		else //right is ultra-sparse (KJI)
		{
			for(int k = 0; k < cd; k++ ) 
			{			
				SparseRow brow = m2.sparseRows[ k ];
				if( brow != null && brow.size() > 0 ) 
				{
					int blen = brow.size();
					int[] bixs = brow.getIndexContainer();
					double[] bvals = brow.getValueContainer();								
					for( int j=0; j<blen; j++ )
					{
						double bval = bvals[j];
						int bix = bixs[j];
						for( int i=0; i<m; i++ )
						{
							double cvald = bval*m1.quickGetValue(i, k);
							if( cvald != 0 ){
								double cval = ret.quickGetValue(i, bix);
								ret.quickSetValue(i, bix, cval+cvald);
							}
						}
					}
				}
			}
		}
		
		//check if sparse output if correct
		ret.examSparsity();	
	}

	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private static void matrixMultTransposeSelfDense( MatrixBlockDSM m1, MatrixBlockDSM ret, boolean leftTranspose ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//1) allocate output block
		ret.rlen = leftTranspose ? m1.clen : m1.rlen;
		ret.clen = leftTranspose ? m1.clen : m1.rlen;
		ret.sparse = false;
		ret.allocateDenseBlock();
	
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
				if( n==1 ) //VECTOR (col)
				{
					c[0] = dotProduct(a, a, m);
				}
				else //MATRIX
				{	
					//init empty result
					Arrays.fill(c, 0, c.length, 0);

					//1) Unrolled inner loop (for better instruction-level parallelism)
					//2) Blocked execution (for less cache trashing in parallel exec) 	
					//3) Asymmetric block sizes (for less misses in inner loop, yet blocks in L1/L2)
					
					final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 
					final int blocksizeK = 24; //64//256KB b block (typical L2 size per core), used while read 512B of a / read/write 4KB of c 
					final int blocksizeJ = 1024; //512//4KB (typical main-memory page size), for scan 

					//temporary arrays (nnz a, b index)
					double[] ta = new double[ blocksizeK ];
					int[]  tbi  = new int[ blocksizeK ];
					
					final int mx = n;
					final int cdx = m;
					final int nx = n;
					
					//blocked execution
					for( int bi = 0; bi < mx; bi+=blocksizeI ) //from bi due to symmetry
						for( int bk = 0, bimin = Math.min(mx, bi+blocksizeI); bk < cdx; bk+=blocksizeK ) 
							for( int bj = bi, bkmin = Math.min(cdx, bk+blocksizeK); bj < nx; bj+=blocksizeJ ) 
							{
								int bklen = bkmin-bk;
								int bjlen = Math.min(nx, bj+blocksizeJ)-bj;
								
								//core sub block matrix multiplication
					    		for( int i = bi; i < bimin; i++) 
					    		{
					    			int aixi = bk*n +i; //start index on a (logical t(X))
					    			int cixj = i * nx + bj; //scan index on c
					    			
					    			//determine nnz of a (for sparsity-aware skipping of rows)
					    			int knnz = copyNonZeroElements(a, aixi, bk, bj, n, nx, ta, tbi, bklen);
					    			
					    			//rest not aligned to blocks of 4 rows
					    			final int bn = knnz % 4;
					    			switch( bn ){
						    			case 1: vectMultiplyAdd(ta[0], a, c, tbi[0], cixj, bjlen); break;
						    	    	case 2: vectMultiplyAdd2(ta[0],ta[1], a, c, tbi[0], tbi[1], cixj, bjlen); break;
						    			case 3: vectMultiplyAdd3(ta[0],ta[1],ta[2], a, c, tbi[0], tbi[1],tbi[2], cixj, bjlen); break;
					    			}
					    			
					    			//compute blocks of 4 rows (core inner loop)
					    			for( int k = bn; k<knnz; k+=4 ){
					    				vectMultiplyAdd4( ta[k], ta[k+1], ta[k+2], ta[k+3], a, c, 
					    						          tbi[k], tbi[k+1], tbi[k+2], tbi[k+3], cixj, bjlen );
					    			}
					    		}
							}
				}
			}
			else
			{	
				//init empty result
				Arrays.fill(c, 0, c.length, 0);
				
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
					//init empty result
					Arrays.fill(c, 0, c.length, 0);
					
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
		if( m1.sparseRows == null )
			return;
		ret.allocateDenseBlock();
		
		//2) transpose self matrix multiply sparse
		// (compute only upper-triangular matrix due to symmetry)		
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m1.clen;

		if( leftTranspose ) // t(X)%*%X 
		{
			//init empty result
			Arrays.fill(c, 0, c.length, 0);
			
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
								vectMultiplyAdd(val, avals, c, aix, i, ix2, alen);
								
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
				//init empty result
				Arrays.fill(c, 0, c.length, 0);
				
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
	private static double dotProduct( double[] a, double[] b, final int len )
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
	private static double dotProduct( double[] a, double[] b, int ai, int bi, final int len )
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
	
	private static double dotProduct( double[] a, double[] b, int[] aix, final int bi, final int len )
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
	private static void vectMultiplyAdd( final double aval, double[] b, double[] c, int bi, int ci, final int len )
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
	 * @param aval1
	 * @param aval2
	 * @param b
	 * @param c
	 * @param bi
	 * @param bi2
	 * @param ci
	 * @param len
	 */
    private static void vectMultiplyAdd2( final double aval1, final double aval2, double[] b, double[] c, int bi1, int bi2, int ci, final int len )
	{
		final int bn = len%8;	
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi1+=8, bi2+=8, ci+=8 ) 
		{
			//read 64B cachelines of b (2x) and c
			//compute c' = aval_1 * b_1 + aval_2 * b_2 + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += aval1 * b[ bi1+0 ] + aval2 * b[ bi2+0 ];
			c[ ci+1 ] += aval1 * b[ bi1+1 ] + aval2 * b[ bi2+1 ];
			c[ ci+2 ] += aval1 * b[ bi1+2 ] + aval2 * b[ bi2+2 ];
			c[ ci+3 ] += aval1 * b[ bi1+3 ] + aval2 * b[ bi2+3 ];
			c[ ci+4 ] += aval1 * b[ bi1+4 ] + aval2 * b[ bi2+4 ];
			c[ ci+5 ] += aval1 * b[ bi1+5 ] + aval2 * b[ bi2+5 ];
			c[ ci+6 ] += aval1 * b[ bi1+6 ] + aval2 * b[ bi2+6 ];
			c[ ci+7 ] += aval1 * b[ bi1+7 ] + aval2 * b[ bi2+7 ];	
		}
	}
	
    /**
     * 
     * @param aval1
     * @param aval2
     * @param aval3
     * @param b
     * @param c
     * @param bi1
     * @param bi2
     * @param bi3
     * @param ci
     * @param len
     */
	private static void vectMultiplyAdd3( final double aval1, final double aval2, final double aval3, double[] b, double[] c, int bi1, int bi2, int bi3, int ci, final int len )
	{
		final int bn = len%8;	
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, bi3++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ] + aval3 * b[ bi3 ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi1+=8, bi2+=8, bi3+=8, ci+=8 ) 
		{
			//read 64B cachelines of b (3x) and c
			//compute c' = aval_1 * b_1 + aval_2 * b_2 + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += aval1 * b[ bi1+0 ] + aval2 * b[ bi2+0 ] + aval3 * b[ bi3+0 ];
			c[ ci+1 ] += aval1 * b[ bi1+1 ] + aval2 * b[ bi2+1 ] + aval3 * b[ bi3+1 ];
			c[ ci+2 ] += aval1 * b[ bi1+2 ] + aval2 * b[ bi2+2 ] + aval3 * b[ bi3+2 ];
			c[ ci+3 ] += aval1 * b[ bi1+3 ] + aval2 * b[ bi2+3 ] + aval3 * b[ bi3+3 ];
			c[ ci+4 ] += aval1 * b[ bi1+4 ] + aval2 * b[ bi2+4 ] + aval3 * b[ bi3+4 ];
			c[ ci+5 ] += aval1 * b[ bi1+5 ] + aval2 * b[ bi2+5 ] + aval3 * b[ bi3+5 ];
			c[ ci+6 ] += aval1 * b[ bi1+6 ] + aval2 * b[ bi2+6 ] + aval3 * b[ bi3+6 ];
			c[ ci+7 ] += aval1 * b[ bi1+7 ] + aval2 * b[ bi2+7 ] + aval3 * b[ bi3+7 ];	
		}
	}
	
	/**
	 * 
	 * @param aval1
	 * @param aval2
	 * @param aval3
	 * @param aval4
	 * @param b
	 * @param c
	 * @param bi1
	 * @param bi2
	 * @param bi3
	 * @param bi4
	 * @param ci
	 * @param len
	 */
	private static void vectMultiplyAdd4( final double aval1, final double aval2, final double aval3, final double aval4, double[] b, double[] c, int bi1, int bi2, int bi3, int bi4, int ci, final int len )
	{
		final int bn = len%8;	
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, bi3++, bi4++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ] + aval3 * b[ bi3 ] + aval4 * b[ bi4 ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi1+=8, bi2+=8, bi3+=8, bi4+=8, ci+=8) 
		{
			//read 64B cachelines of b (4x) and c 
			//compute c' = aval_1 * b_1 + aval_2 * b_2 + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += aval1 * b[ bi1+0 ] + aval2 * b[ bi2+0 ] + aval3 * b[ bi3+0 ] + aval4 * b[ bi4+0 ];
			c[ ci+1 ] += aval1 * b[ bi1+1 ] + aval2 * b[ bi2+1 ] + aval3 * b[ bi3+1 ] + aval4 * b[ bi4+1 ];
			c[ ci+2 ] += aval1 * b[ bi1+2 ] + aval2 * b[ bi2+2 ] + aval3 * b[ bi3+2 ] + aval4 * b[ bi4+2 ];
			c[ ci+3 ] += aval1 * b[ bi1+3 ] + aval2 * b[ bi2+3 ] + aval3 * b[ bi3+3 ] + aval4 * b[ bi4+3 ];
			c[ ci+4 ] += aval1 * b[ bi1+4 ] + aval2 * b[ bi2+4 ] + aval3 * b[ bi3+4 ] + aval4 * b[ bi4+4 ];
			c[ ci+5 ] += aval1 * b[ bi1+5 ] + aval2 * b[ bi2+5 ] + aval3 * b[ bi3+5 ] + aval4 * b[ bi4+5 ];
			c[ ci+6 ] += aval1 * b[ bi1+6 ] + aval2 * b[ bi2+6 ] + aval3 * b[ bi3+6 ] + aval4 * b[ bi4+6 ];
			c[ ci+7 ] += aval1 * b[ bi1+7 ] + aval2 * b[ bi2+7 ] + aval3 * b[ bi3+7 ] + aval4 * b[ bi4+7 ];	
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
	private static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int ci, final int len )
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
	
	private static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len )
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
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bi
	 * @param ci
	 * @param len
	 */
	private static void vectMultiplyWrite( final double aval, double[] b, double[] c, int bi, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, bi++, ci++)
			c[ ci ] = aval * b[ bi ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, bi+=8, ci+=8) 
		{
			//read 64B cachelines of b and c
			//compute c' = aval * b + c
			//write back 64B cacheline of c = c'
			c[ ci+0 ] = aval * b[ bi+0 ];
			c[ ci+1 ] = aval * b[ bi+1 ];
			c[ ci+2 ] = aval * b[ bi+2 ];
			c[ ci+3 ] = aval * b[ bi+3 ];
			c[ ci+4 ] = aval * b[ bi+4 ];
			c[ ci+5 ] = aval * b[ bi+5 ];
			c[ ci+6 ] = aval * b[ bi+6 ];
			c[ ci+7 ] = aval * b[ bi+7 ];
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
				c[ lix ] = c[ uix+j ];
	}

	/**
	 * 
	 * @param a
	 * @param aixi
	 * @param bk
	 * @param bj
	 * @param n
	 * @param tmpa
	 * @param tmpbi
	 * @param bklen
	 * @return
	 */
	private static int copyNonZeroElements( double[] a, final int aixi, final int bk, final int bj, final int n, double[] tmpa, int[] tmpbi, final int bklen )
	{
		int knnz = 0;
		for( int k = 0; k < bklen; k++ )
			if( a[ aixi+k ] != 0 ) {
				tmpa[ knnz ] = a[ aixi+k ];
				tmpbi[ knnz ] = (bk+k) * n + bj; //scan index on b
				knnz ++;
			}
		
		return knnz;
	}
	
	/**
	 * 
	 * @param a
	 * @param aixi
	 * @param bk
	 * @param bj
	 * @param n
	 * @param nx
	 * @param tmpa
	 * @param tmpbi
	 * @param bklen
	 * @return
	 */
	private static int copyNonZeroElements( double[] a, int aixi, final int bk, final int bj, final int n, final int nx, double[] tmpa, int[] tmpbi, final int bklen )
	{
		int knnz = 0;
		for( int k = 0; k < bklen; k++, aixi+=n )
			if( a[ aixi ] != 0 ) {
				tmpa[ knnz ] = a[ aixi ];
				tmpbi[ knnz ] = (bk+k) * nx + bj; //scan index on b
				knnz ++;
			}
		
		return knnz;
	}
	
	/////////////////////////////////////////////////////
	// old matrix mult implementation (for comparison) //
	/////////////////////////////////////////////////////	
	
	/*
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
	*/
	
	/*
	public static void matrixMult(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret, int k) 
			throws DMLRuntimeException
	{		
		//Timing time = new Timing(true);
		
		if(!m1.sparse && !m2.sparse)
			matrixMultDenseDense(m1, m2, ret, k);
		else 
			throw new DMLRuntimeException("Not implemented yet.");
		
		//System.out.println("MM("+k+") ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+")x" +
		//		              "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	}
	*/
	
	/*
	private static void matrixMultDenseDense(MatrixBlockDSM m1, MatrixBlockDSM m2, MatrixBlockDSM ret, int k) 
		throws DMLRuntimeException
	{	
		//check inputs / outputs
		if( m1.denseBlock==null || m2.denseBlock==null )
			return;
		ret.sparse = false;
		if( ret.denseBlock==null )
			ret.denseBlock = new double[ret.rlen * ret.clen];
		
		double[] a = m1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m1.clen;

		final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 

		//init empty result
		Arrays.fill(c, 0, c.length, 0);
		
		try
		{
			int blk = (m/k)+((m/k)%blocksizeI); 
			
			Thread[] t = new Thread[k];
			for( int i=0; i<k; i++ )
				t[i] = new Thread(new MatrixMultLib().new MMWorker( a,b,c,m,n,cd,i*blk,(i+1)*blk ));
	
			for( int i=0; i<k; i++ )
				t[i].start();
	
			for( int i=0; i<k; i++ )
				t[i].join();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		ret.recomputeNonZeros();
		ret.examSparsity();
	}
	
	private class MMWorker implements Runnable
	{
		private double[] a = null;
		private double[] b = null;
		private double[] c = null;
		private int m;
		private int n;
		private int cd;
		private int starti;
		private int endi;
		
		
		public MMWorker(double[] a, double[] b, double[] c, int m, int n, int cd, int starti, int endi) 
		{
			this.a = a;
			this.b = b;
			this.c = c;
			//this.a = new double[a.length]; System.arraycopy(a, 0, this.a, 0, a.length);
			//this.b = new double[b.length]; System.arraycopy(b, 0, this.b, 0, a.length);
			//this.c = new double[c.length]; System.arraycopy(c, 0, this.c, 0, a.length);
			
			this.m = m;
			this.n = n;
			this.cd = cd;
			this.starti = starti;
			this.endi = Math.min(endi,m);

			System.out.println("MMWorker: "+(endi-starti)+" rows");
		}

		@Override
		public void run() 
		{
			//1) Unrolled inner loop (for better instruction-level parallelism)
			//2) Blocked execution (for less cache trashing in parallel exec) 	
			//3) Asymmetric block sizes (for less misses in inner loop, yet blocks in L1/L2)
			
			final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 
			final int blocksizeK = 24; //64//256KB b block (typical L2 size per core), used while read 512B of a / read/write 4KB of c 
			final int blocksizeJ = 1024; //512//4KB (typical main-memory page size), for scan 

			//temporary arrays (nnz a, b index)
			double[] ta = new double[ blocksizeK ];
			int[]  tbi  = new int[ blocksizeK ];
			
			//blocked execution
			for( int bi = starti; bi < endi; bi+=blocksizeI )
				for( int bk = 0, bimin = Math.min(endi, bi+blocksizeI); bk < cd; bk+=blocksizeK ) 
					for( int bj = 0, bkmin = Math.min(cd, bk+blocksizeK); bj < n; bj+=blocksizeJ ) 
					{
						int bklen = bkmin-bk;
						int bjlen = Math.min(n, bj+blocksizeJ)-bj;
						
						//core sub block matrix multiplication
			    		for( int i = bi; i < bimin; i++) 
			    		{
			    			int aixi = i * cd + bk; //start index on a
			    			int cixj = i * n + bj; //scan index on c
			    			
			    			//determine nnz of a (for sparsity-aware skipping of rows)
			    			int knnz = 0;
			    			for( int k = 0; k < bklen; k++ )
			    				if( a[aixi+k] != 0 ) {
			    					ta[ knnz ] = a[aixi+k];
			    					tbi[ knnz ] = (bk+k) * n + bj; //scan index on b
			    					knnz ++;
			    				}
			    			
			    			//rest not aligned to blocks of 4 rows
			    			final int bn = knnz % 4;
			    			switch( bn ){
				    			case 1: vectMultiplyAdd(ta[0], b, c, tbi[0], cixj, bjlen); break;
				    	    	case 2: vectMultiplyAdd2(ta[0],ta[1], b, c, tbi[0], tbi[1], cixj, bjlen); break;
				    			case 3: vectMultiplyAdd3(ta[0],ta[1],ta[2], b, c, tbi[0], tbi[1],tbi[2], cixj, bjlen); break;
			    			}
			    			
			    			//compute blocks of 4 rows (core inner loop)
			    			for( int k = bn; k<knnz; k+=4 ){
			    				vectMultiplyAdd4( ta[k], ta[k+1], ta[k+2], ta[k+3], b, c, 
			    						          tbi[k], tbi[k+1], tbi[k+2], tbi[k+3], cixj, bjlen );
			    			}
			    		}
					}		
		}
		
	}
	*/

	public static void main(String[] args)
	{
		try 
		{
			int n = 1000;
			MatrixBlockDSM m1 = MatrixBlock.randOperations(n, n, 1000, 1000, 0.7, 0, 1, "uniform", 7);
			MatrixBlockDSM m2 = MatrixBlock.randOperations(n, n, 1000, 1000, 0.00001, 0, 1, "uniform", 3);
			MatrixBlock out = new MatrixBlock(n, n, false);
			
			for( int i=0; i<20; i++ )
			{
				
				long t0 = System.nanoTime();
				
				MatrixMultLib.matrixMult(m1, m2, out);
				//MatrixMultLib.matrixMult(m1, m2, out, 8);
				
				long t1 = System.nanoTime();
	
				System.out.println("Matrix Mult: "+(((double)(t1-t0))/1000000)+" ms");
			}
			
		} 
		catch (DMLRuntimeException e) {
			e.printStackTrace();
		}
		
	}
}
