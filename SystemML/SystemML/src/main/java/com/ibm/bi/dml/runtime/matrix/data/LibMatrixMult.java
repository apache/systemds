/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.ibm.bi.dml.lops.MapMultChain.ChainType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * MB:
 * Library for matrix multiplications including MM, MV, VV for all
 * combinations of dense, sparse, ultrasparse representations and special
 * operations such as transpose-self matrix multiplication.
 * 
 * In general all implementations use internally dense outputs
 * for direct access, but change the final result to sparse if necessary.
 * The only exceptions are ultra-sparse matrix mult, wsloss and wsigmoid.  
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
 */
public class LibMatrixMult 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final boolean LOW_LEVEL_OPTIMIZATION = true;

	private LibMatrixMult() {
		//prevent instantiation via private constructor
	}
	
	////////////////////////////////
	// public matrix mult interface
	////////////////////////////////
	
	/**
	 * Performs a matrix multiplication and stores the result in the output matrix.
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
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret) 
		throws DMLRuntimeException
	{	
		//check inputs / outputs
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		
		//Timing time = new Timing(true);
		
		//pre-processing: output allocation
		ret.sparse = (m1.isUltraSparse() || m2.isUltraSparse());
		if( !ret.sparse )
			ret.allocateDenseBlock();
		
		//core matrix mult computation
		if( m1.isUltraSparse() || m2.isUltraSparse() )
			matrixMultUltraSparse(m1, m2, ret, 0, m1.rlen);
		else if(!m1.sparse && !m2.sparse)
			matrixMultDenseDense(m1, m2, ret, 0, m1.rlen);
		else if(m1.sparse && m2.sparse)
			matrixMultSparseSparse(m1, m2, ret, 0, m1.rlen);
		else if(m1.sparse)
			matrixMultSparseDense(m1, m2, ret, 0, m1.rlen);
		else
			matrixMultDenseSparse(m1, m2, ret, 0, m1.rlen);
		
		//post-processing: nnz/representation
		if( !ret.sparse )
			ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+")x" +
		//		              "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	}
	
	/**
	 * Performs a multi-threaded matrix multiplication and stores the result in the output matrix.
	 * The parameter k (k>=1) determines the max parallelism k' with k'=min(k, vcores, m1.rlen).
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param k
	 * @throws DMLRuntimeException 
	 */
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) 
		throws DMLRuntimeException
	{	
		//check inputs / outputs
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		
		//check no parallelization benefit (fallback to sequential)
		if( m1.rlen == 1 ) {
			matrixMult(m1, m2, ret);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing: output allocation (in contrast to single-threaded,
		//we need to allocate sparse as well in order to prevent synchronization)
		ret.sparse = (m1.isUltraSparse() || m2.isUltraSparse());
		if( !ret.sparse )
			ret.allocateDenseBlock();
		else
			ret.allocateSparseRowsBlock();
		
		//core multi-threaded matrix mult computation
		//(currently: always parallelization over number of rows)
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<MatrixMultTask> tasks = new ArrayList<MatrixMultTask>();
			int blklen = (int)(Math.ceil((double)m1.rlen/k));
			for( int i=0; i<k & i*blklen<m1.rlen; i++ )
				tasks.add(new MatrixMultTask(m1, m2, ret, i*blklen, Math.min((i+1)*blklen, m1.rlen)));
			pool.invokeAll(tasks);	
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//post-processing: nnz/representation (in contrast to single-threaded,
		//we need to recompute nnz for sparse as well in order to prevent synchronization)
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MM k="+k+" ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+")x" +
		//		              "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	}
	
	/**
	 * Performs a matrix multiplication chain operation of type t(X)%*%(X%*%v) or t(X)%*%(w*(X%*%v)).
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * @param m1
	 * @param m2
	 * @param w
	 * @param ret
	 * @param ct
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static void matrixMultChain(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{		
		//check inputs / outputs (after that mV and mW guaranteed to be dense)
		if( mX.isEmptyBlock(false) || mV.isEmptyBlock(false) || (mW !=null && mW.isEmptyBlock(false)) )
			return;

		//Timing time = new Timing(true);
				
		//pre-processing
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		//core matrix mult chain computation
		if( mX.sparse )
			matrixMultChainSparse(mX, mV, mW, ret, ct, 0, mX.rlen);
		else
			matrixMultChainDense(mX, mV, mW, ret, ct, 0, mX.rlen);
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMChain "+ct.toString()+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//		             "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * Performs a parallel matrix multiplication chain operation of type t(X)%*%(X%*%v) or t(X)%*%(w*(X%*%v)).
	 * The parameter k (k>=1) determines the max parallelism k' with k'=min(k, vcores, m1.rlen).
	 * 
	 * NOTE: This multi-threaded mmchain operation has additional memory requirements of k*ncol(X)*8bytes 
	 * for partial aggregation. Current max memory: 256KB; otherwise redirectly to sequential execution.
	 * 
	 * @param mX
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param ct
	 * @param k
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static void matrixMultChain(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int k) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{		
		//check inputs / outputs (after that mV and mW guaranteed to be dense)
		if( mX.isEmptyBlock(false) || mV.isEmptyBlock(false) || (mW !=null && mW.isEmptyBlock(false)) )
			return;

		//check too high additional memory requirements (fallback to sequential)
		if( mV.rlen * 8 * k > 256*1024 ) { //256KB
			matrixMultChain(mX, mV, mW, ret, ct);
			return;
		}
		
		//Timing time = new Timing(true);
				
		//pre-processing
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		//core matrix mult chain computation
		//(currently: always parallelization over number of rows)
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<MatrixMultChainTask> tasks = new ArrayList<MatrixMultChainTask>();
			int blklen = (int)(Math.ceil((double)mX.rlen/k));
			blklen += (blklen%24 != 0)?24-blklen%24:0;
			for( int i=0; i<k & i*blklen<mX.rlen; i++ )
				tasks.add(new MatrixMultChainTask(mX, mV, mW, ret, ct, i*blklen, Math.min((i+1)*blklen, mX.rlen)));
			pool.invokeAll(tasks);	
			pool.shutdown();
			//aggregate partial results
			for( MatrixMultChainTask task : tasks )
				vectAdd(task.getResult().denseBlock, ret.denseBlock, 0, 0, mX.clen);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMChain "+ct.toString()+" k="+k+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//		              "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static void matrixMultTransposeSelf( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		if( m1.sparse )
			matrixMultTransposeSelfSparse(m1, ret, leftTranspose);
		else 
			matrixMultTransposeSelfDense(m1, ret, leftTranspose);

		//System.out.println("TSMM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret1
	 * @param ret2
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultPermute( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2 )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check inputs / outputs
		if( pm1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;

		//Timing time = new Timing(true);

		//pre-processing
		ret1.sparse = (m2.sparse || ret1.sparse);
		if( ret1.sparse )
			ret1.allocateSparseRowsBlock();
		else
			ret1.allocateDenseBlock();
		
		//core permutation mm computation
		if( m2.sparse )
			matrixMultPermuteSparse(pm1, m2, ret1, ret2, 0, pm1.rlen);
		else if( ret1.sparse )
			matrixMultPermuteDenseSparse(pm1, m2, ret1, ret2, 0, pm1.rlen);
		else
			matrixMultPermuteDense(pm1, m2, ret1, ret2, 0, pm1.rlen);

		//post-processing
		ret1.recomputeNonZeros();
		ret1.examSparsity();
		if( ret2 != null ) { //optional second output
			ret2.recomputeNonZeros();
			ret2.examSparsity();
		}

		//System.out.println("PMM Seq ("+pm1.isInSparseFormat()+","+pm1.getNumRows()+","+pm1.getNumColumns()+","+pm1.getNonZeros()+")x" +
		//                  "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	}	

	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret1
	 * @param ret2
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 * @throws DMLRuntimeException 
	 */
	public static void matrixMultPermute( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int k)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check inputs / outputs
		if( pm1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;

		//check no parallelization benefit (fallback to sequential)
		if (pm1.rlen == 1) {
			matrixMultPermute(pm1, m2, ret1, ret2);
			return;
		}
	
		//Timing time = new Timing(true);
		
		//allocate first output block (second allocated if needed)
		ret1.sparse = false;
		ret1.allocateDenseBlock();
		
		try
		{
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<PMMTask> tasks = new ArrayList<PMMTask>();
			int blklen = (int)(Math.ceil((double)pm1.rlen/k));
			for( int i=0; i<k & i*blklen<pm1.rlen; i++ )
				tasks.add(new PMMTask(pm1, m2, ret1, ret2, i*blklen, Math.min((i+1)*blklen, pm1.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		
		//post-processing
		ret1.recomputeNonZeros();
		ret1.examSparsity();
		if( ret2 != null ) { //optional second output
			ret2.recomputeNonZeros();
			ret2.examSparsity();
		}
		
		// System.out.println("PMM Par ("+pm1.isInSparseFormat()+","+pm1.getNumRows()+","+pm1.getNumColumns()+","+pm1.getNonZeros()+")x" +
		//                   "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop() + " with " + k + " threads");
	}	
	

	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException 
	 */
	public static void matrixMultWSLoss(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( wt==WeightsType.POST && mW.isEmptyBlock(false) )
			return; 

		//Timing time = new Timing(true);

		//core weighted square sum mm computation
		if( !mX.sparse && !mU.sparse && !mV.sparse && (mW==null || !mW.sparse) 
			&& !mX.isEmptyBlock() && !mU.isEmptyBlock() && !mV.isEmptyBlock() && (mW==null || !mW.isEmptyBlock()))
			matrixMultWSLossDense(mX, mU, mV, mW, ret, wt, mX.clen, mU.clen, 0, mX.rlen);
		else if( mX.sparse && !mU.sparse && !mV.sparse && (mW==null || mW.sparse)
				&& !mX.isEmptyBlock() && !mU.isEmptyBlock() && !mV.isEmptyBlock() && (mW==null || !mW.isEmptyBlock()))
			matrixMultWSLossSparseDense(mX, mU, mV, mW, ret, wt, 0, mX.rlen);
		else
			matrixMultWSLossGeneric(mX, mU, mV, mW, ret, wt, 0, mX.rlen);
		
		//System.out.println("MMWSLoss Seq "+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}
	
	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException 
	 */
	public static void matrixMultWSLoss(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int k) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( wt==WeightsType.POST && mW.isEmptyBlock(false) )
			return;
		
		//check no parallelization benefit (fallback to sequential)
		if (mX.rlen == 1) {
			matrixMultWSLoss(mX, mU, mV, mW, ret, wt);
			return;
		}
		
		//Timing time = new Timing(true);
		
		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<WSLMMTask> tasks = new ArrayList<WSLMMTask>();
			int blklen = (int)(Math.ceil((double)mX.rlen/k));
			for( int i=0; i<k & i*blklen<mX.rlen; i++ )
				tasks.add(new WSLMMTask(mX, mU, mV, mW, wt, i*blklen, Math.min((i+1)*blklen, mX.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
			//aggregate partial results
			double wsloss = 0;
			for(WSLMMTask rt : tasks)
				wsloss += rt.getWSLoss();
			ret.quickSetValue(0, 0, wsloss);
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		//System.out.println("MMWSLoss Par ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop() + " with " + k + " threads");
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
	private static void matrixMultDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) 
		throws DMLRuntimeException
	{			
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
				for( int i=rl, cix=rl*n; i < ru; i++, cix+=n) {
					if( a[i] == 1 )
						System.arraycopy(b, 0, c, cix, n);
				    else if( a[i] != 0 )
						vectMultiplyWrite(a[i], b, c, 0, cix, n);
					else
						Arrays.fill(c, cix, cix+n, 0);
				}
			}
			else if( n==1 && cd == 1 ) //VECTOR-SCALAR
			{
				vectMultiplyWrite(b[0], a, c, rl, rl, ru-rl);
			}
			else if( n==1 )            //MATRIX-VECTOR
			{
				for( int i=rl, aix=rl*cd; i < ru; i++, aix+=cd) 
					c[ i ] = dotProduct(a, b, aix, 0, cd);	
			}
			else //MATRIX-MATRIX
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
				for( int bi = rl; bi < ru; bi+=blocksizeI )
					for( int bk = 0, bimin = Math.min(ru, bi+blocksizeI); bk < cd; bk+=blocksizeK ) 
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
				    			//if( knnz > 0 ) //for skipping empty rows
				    			
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
			for( int i = rl, aix=rl*cd, cix=rl*n; i < ru; i++, cix+=n) 
				for( int k = 0, bix=0; k < cd; k++, aix++, bix+=n)
				{			
					val = a[ aix ];
					if( val != 0 )
						for( int j = 0; j < n; j++) 
							c[ cix+j ] += val * b[ bix+j ];
				}	
		}
		
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultDenseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) 
		throws DMLRuntimeException 
	{	
		double[] a = m1.denseBlock;
		double[] c = ret.denseBlock;
		int n = m2.clen;
		int cd = m1.clen;
	
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if( LOW_LEVEL_OPTIMIZATION )
		{
			final int blocksizeI = 32; //256KB c block (typical L2 size per core), 32KB a block 
			final int blocksizeK = 32; 
			//note: in contrast to dense-dense, no blocking over j (would require maintaining blocksizeK indexes, counter-productive on skew)
			
			SparseRow[] b = m2.sparseRows;
			
			//blocked execution
			for( int bi = rl; bi < ru; bi+=blocksizeI )
				for( int bk = 0, bimin = Math.min(ru, bi+blocksizeI); bk < cd; bk+=blocksizeK ) 
				{
					int bklen = Math.min(cd, bk+blocksizeK)-bk;
					
					//core sub block matrix multiplication
		    		for( int i = bi; i < bimin; i++) 
		    		{
		    			int aixi = i * cd + bk; //start index on a
		    			int cixj = i * n + 0; //scan index on c
		    			
		    			for( int k = 0; k < bklen; k++ )
						{
							double val = a[aixi+k];
							SparseRow brow = b[ bk+k ];
							if( val != 0 && brow != null && !brow.isEmpty() ) {
								int blen = brow.size();
								int[] bix = brow.getIndexContainer();
								double[] bvals = brow.getValueContainer();								
								vectMultiplyAdd(val, bvals, c, bix, cixj, blen);
							}
						}
		    		}
				}	
		}
		else
		{
			for( int i=rl, aix=rl*cd, cix=rl*n; i < ru; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					double val = a[aix];
					if( val!=0 )
					{
						SparseRow brow = m2.sparseRows[ k ];
						if( brow != null && !brow.isEmpty() ) 
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
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultSparseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) 
		throws DMLRuntimeException
	{	
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;

		if( LOW_LEVEL_OPTIMIZATION )
		{
		
			if( m==1 && n==1 ) //DOT PRODUCT
			{
				SparseRow arow = m1.sparseRows[0];
				if( arow != null && !arow.isEmpty() )
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					
					c[0] = dotProduct(avals, b, aix, 0, alen);
				}
			}
			else if( n==1 ) //MATRIX-VECTOR
			{
				for( int i=rl; i<Math.min(ru, m1.sparseRows.length); i++ )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && !arow.isEmpty() ) 
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
				for( int i=rl, cix=rl*n; i<Math.min(ru, m1.sparseRows.length); i++, cix+=n )
				{
					SparseRow arow = m1.sparseRows[i];
					if( arow != null && !arow.isEmpty() ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();					
						
						if( alen==1 && avals[0]==1 ) //ROW SELECTION 
						{
							//plain memcopy for permutation matrices
							System.arraycopy(b, aix[0]*n, c, cix, n);
						}
						else //GENERAL CASE
						{
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
		}
		else
		{
			//init empty result
			Arrays.fill(c, 0, c.length, 0);
			
			for( int i=rl, cix=rl*n; i<Math.min(ru, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && !arow.isEmpty() ) 
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
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) 
		throws DMLRuntimeException
	{	
		double[] c = ret.denseBlock;
		int n = m2.clen;
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if(LOW_LEVEL_OPTIMIZATION)
		{
			for( int i=rl, cix=rl*n; i<Math.min(ru, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && !arow.isEmpty() ) 
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						double val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && !brow.isEmpty() ) 
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
			for( int i=rl, cix=rl*n; i<Math.min(ru, m1.sparseRows.length); i++, cix+=n )
			{
				SparseRow arow = m1.sparseRows[i];
				if( arow != null && !arow.isEmpty() ) 
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					
					for(int k = 0; k < alen; k++) 
					{
						double val = avals[k];
						SparseRow brow = m2.sparseRows[ aix[k] ];
						if( brow != null && !brow.isEmpty() ) 
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
	private static void matrixMultUltraSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) 
		throws DMLRuntimeException 
	{
		boolean leftUS = m1.isUltraSparse();
		final int m  = m1.rlen;
		final int cd = m1.clen;
		final int n  = m2.clen;
		
		if( leftUS ) //left is ultra-sparse (IKJ)
		{
			boolean rightSparse = m2.sparse;
			
			for( int i=rl; i<ru; i++ )
			{
				SparseRow arow = m1.sparseRows[ i ];
				if( arow != null && !arow.isEmpty() ) 
				{
					int alen = arow.size();
					int[] aixs = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();	
					
					if( alen==1 && avals[0]==1 ) //ROW SELECTION (no aggregation)
					{
						int aix = aixs[0];
						if( rightSparse ) { //sparse right matrix (full row copy)
							if( m2.sparseRows!=null && m2.sparseRows[aix]!=null ) {
								ret.rlen=m;
								ret.allocateSparseRowsBlock(false); //allocation on demand
								ret.sparseRows[i] = new SparseRow(m2.sparseRows[aix]); 
								ret.nonZeros += ret.sparseRows[i].size();
							}
						}
						else { //dense right matrix (append all values)
							for( int j=0; j<n; j++ )
								ret.appendValue(i, j, m2.quickGetValue(aix, j));
						}
					}
					else //GENERAL CASE
					{
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
		}
		else //right is ultra-sparse (KJI)
		{
			for(int k = 0; k < cd; k++ ) 
			{			
				SparseRow brow = m2.sparseRows[ k ];
				if( brow != null && !brow.isEmpty() ) 
				{
					int blen = brow.size();
					int[] bixs = brow.getIndexContainer();
					double[] bvals = brow.getValueContainer();								
					for( int j=0; j<blen; j++ )
					{
						double bval = bvals[j];
						int bix = bixs[j];
						for( int i=rl; i<ru; i++ )
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
		//no need to recompute nonzeros because maintained internally
	}

	/**
	 * 
	 * @param mX
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param ct
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private static void matrixMultChainDense(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int rl, int ru) 
	{
		double[] a = mX.denseBlock;
		double[] b = mV.denseBlock;
		double[] w = (mW!=null) ? mW.denseBlock : null;
		double[] c = ret.denseBlock;
		final int cd = mX.clen; //features in X
		boolean weights = (ct == ChainType.XtwXv);

		//temporary array for cache blocking
		//(blocksize chosen to fit b+v in L2 (256KB) for default 1k blocks)
		final int blocksize = 24; // constraint: factor of 4
		double[] tmp = new double[blocksize];
			
		//blockwise mmchain computation
		final int bn = ru - ru % blocksize; //rl blocksize aligned
		for( int bi=rl; bi < bn; bi+=blocksize ) 
		{
			//compute 1st matrix-vector for row block
			for( int j=0, aix=bi*cd; j < blocksize; j++, aix+=cd)
				tmp[j] = dotProduct(a, b, aix, 0, cd);
			
			//multiply weights (in-place), if required
			if( weights ) 
				vectMultiply(w, tmp, bi, 0, blocksize);	
			
			//compute 2nd matrix vector for row block and aggregate
			for (int j=0, aix=bi*cd; j < blocksize; j+=4, aix+=4*cd)
				vectMultiplyAdd4(tmp[j], tmp[j+1], tmp[j+2], tmp[j+3], a, c, aix, aix+cd, aix+2*cd, aix+3*cd, 0, cd);
		}
		
		//compute rest (not aligned to blocksize)
		for( int i=bn, aix=bn*cd; i < ru; i++, aix+=cd ) {
			double val = dotProduct(a, b, aix, 0, cd);
			val *= (weights) ? w[i] : 1; 
			vectMultiplyAdd(val, a, c, aix, 0, cd);				
		}
	}
	
	/**
	 * 
	 * @param mX
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param ct
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private static void matrixMultChainSparse(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int rl, int ru) 
	{
		SparseRow[] a = mX.sparseRows;
		double[] b = mV.denseBlock;
		double[] w = (mW!=null) ? mW.denseBlock : null;
		double[] c = ret.denseBlock;
		boolean weights = (ct == ChainType.XtwXv);
		
		//temporary array for cache blocking
		//(blocksize chosen to fit b+v in L2 (256KB) for default 1k blocks)
		final int blocksize = 24;
		double[] tmp = new double[blocksize];
		
		//blockwise mmchain computation
		for( int bi=rl; bi < ru; bi+=blocksize ) 
		{
			//reset row block intermediate
			int tmplen = Math.min(blocksize, ru-bi);

			//compute 1st matrix-vector for row block
			for( int j=0; j < tmplen; j++) {
				SparseRow arow = a[bi+j];
				if( arow != null && !arow.isEmpty() ) {
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();					
					tmp[j] = dotProduct(avals, b, aix, 0, alen);							
				}
			}
			
			//multiply weights (in-place), if required
			if( weights ) 
				vectMultiply(w, tmp, bi, 0, tmplen);	
			
			//compute 2nd matrix vector for row block and aggregate
			for( int j=0; j < tmplen; j++) {
				SparseRow arow = a[bi+j];
				if( arow != null && !arow.isEmpty() && tmp[j] != 0 ) {
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();		
					vectMultiplyAdd(tmp[j], avals, c, aix, 0, alen);							
				}
			}
		}
	}
	

	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private static void matrixMultTransposeSelfDense( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose ) 
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
	private static void matrixMultTransposeSelfSparse( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose ) 
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
					if( arow != null && !arow.isEmpty() ) 
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
					if( arow != null && !arow.isEmpty() ) 
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
				if( arow !=null && !arow.isEmpty() )
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
				MatrixBlock tmpBlock = new MatrixBlock(n,m,m1.sparse);
				m1.reorgOperations(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), 
						       tmpBlock, 0, 0, -1);
			
				if( tmpBlock.sparseRows == null )
					return;
				
				//algorithm: scan rows, foreach row self join (KIJ)
				if( LOW_LEVEL_OPTIMIZATION )
				{
					for( SparseRow arow : tmpBlock.sparseRows )
						if( arow != null && !arow.isEmpty() ) 
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
						if( arow != null && !arow.isEmpty() ) 
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
	
	/**
	 * 
	 * @param pm1
	 * @param m2
	 * @param ret1
	 * @param ret2
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultPermuteDense(MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru) 
		throws DMLRuntimeException
	{
		double[] a = pm1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret1.denseBlock;

		final int n = m2.clen;
		final int brlen = ret1.getNumRows();
		
		int lastblk = -1;
		
		for( int i=rl, bix=rl*n; i<ru; i++, bix+=n ) 
		{
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast
			if( pos > 0 ) //selected row
			{
				int bpos = (pos-1) % brlen;
				int blk = (pos-1) / brlen;
				
				//allocate and switch to second output block
				//(never happens in cp, correct for multi-threaded usage)
				if( lastblk!=-1 && lastblk<blk ){ 
					ret2.sparse = false;
					ret2.allocateDenseBlock();
					c = ret2.denseBlock;		
				}
		
				//memcopy entire dense row into target position
				System.arraycopy(b, bix, c, bpos*n, n);
				lastblk = blk;
			}
		}
	}

	/**
	 * 
	 * @param pm1
	 * @param m2
	 * @param ret1
	 * @param ret2
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultPermuteDenseSparse( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
	{
		double[] a = pm1.denseBlock;
		double[] b = m2.denseBlock;
		SparseRow[] c = ret1.sparseRows;

		final int n = m2.clen;
		final int brlen = ret1.getNumRows();
		
		int lastblk = -1;
		for( int i=rl, bix=rl*n; i<ru; i++, bix+=n ) 
		{
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast
			if( pos > 0 ) //selected row
			{
				int bpos = (pos-1) % brlen;
				int blk = (pos-1) / brlen;
				
				//allocate and switch to second output block
				//(never happens in cp, correct for multi-threaded usage)
				if( lastblk!=-1 && lastblk<blk ){ 
					ret2.sparse = true;
					ret2.rlen=ret1.rlen;
					ret2.allocateSparseRowsBlock();
					c = ret2.sparseRows;		
				}
		
				//append entire dense row into sparse target position
				c[bpos] = new SparseRow( n );
				for( int j=0; j<n; j++ )
					c[bpos].append(j, b[bix+j]);
				lastblk = blk;
			}
		}
		
	}
	
	/**
	 * 
	 * @param pm1
	 * @param m2
	 * @param ret1
	 * @param ret2
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultPermuteSparse( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
	{
		double[] a = pm1.denseBlock;
		SparseRow[] b = m2.sparseRows;
		SparseRow[] c = ret1.sparseRows;

		final int brlen = ret1.getNumRows();
		
		int lastblk = -1;
		for( int i=rl; i<ru; i++ ) 
		{
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast			
			if( pos > 0 ) //selected row
			{
				int bpos = (pos-1) % brlen;
				int blk = (pos-1) / brlen;
				
				//allocate and switch to second output block
				//(never happens in cp, correct for multi-threaded usage)
				if( lastblk!=-1 && lastblk<blk ){ 
					ret2.sparse = true;
					ret2.allocateSparseRowsBlock();
					c = ret2.sparseRows;		
				}
		
				//memcopy entire sparse row into target position
				if( b[i] != null )
					c[bpos] = new SparseRow( b[i] );
				lastblk = blk;
			}
		}

	}
	
	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param wt
	 * @param n
	 * @param cd
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWSLossDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int n, int cd, int rl, int ru)
	{
		double[] x = mX.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		double[] w = (mW!=null)? mW.denseBlock : null;
		double wsloss = 0;
		
		// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
		if( wt==WeightsType.POST )
		{
			for( int i=rl, ix=rl*n, uix=rl*cd; i<ru; i++, uix+=cd )
				for( int j=0, vix=0; j<n; j++, ix++, vix+=cd) {
					double wij = w[ix];
					if( wij != 0 ) {
						double uvij = dotProduct(u, v, uix, vix, cd);
						wsloss += (wij*(x[ix]-uvij))*(wij*(x[ix]-uvij)); //^2
					}
				}	
		}
		// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
		else if( wt==WeightsType.PRE )
		{
			// approach: iterate over all cells of X maybe sparse and dense
			for( int i=rl, ix=rl*n, uix=rl*cd; i<ru; i++, uix+=cd )
				for( int j=0, vix=0; j<n; j++, ix++, vix+=cd) {
					double wij = w[ix];
					double uvij = 0;
					if( wij != 0 )
						uvij = dotProduct(u, v, uix, vix, cd);
					wsloss += (x[ix]-wij*uvij)*(x[ix]-wij*uvij); //^2
				}
		}
		// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
		else if( wt==WeightsType.NONE )
		{
			// approach: iterate over all cells of X and 
			for( int i=rl, ix=rl*n, uix=rl*cd; i<ru; i++, uix+=cd )
				for( int j=0, vix=0; j<n; j++, ix++, vix+=cd) {
					double uvij = dotProduct(u, v, uix, vix, cd);
					wsloss += (x[ix]-uvij)*(x[ix]-uvij); //^2
				}
		}
		
		ret.quickSetValue(0, 0, wsloss);
	}
	
	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWSLossSparseDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		final int n = mX.clen; 
		final int cd = mU.clen;
		SparseRow[] x = mX.sparseRows;
		SparseRow[] w = (mW!=null)? mW.sparseRows : null;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		double wsloss = 0; 
		
		// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
		if( wt==WeightsType.POST )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
				if( w[i] != null && !w[i].isEmpty() ) {
					int wlen = w[i].size();
					int[] wix = w[i].getIndexContainer();
					double[] wval = w[i].getValueContainer();
					for( int k=0; k<wlen; k++ ) {
						double xi = mX.quickGetValue(i, wix[k]);
						double uvij = dotProduct(u, v, uix, wix[k]*cd, cd);
						wsloss += (wval[k]*(xi-uvij))*(wval[k]*(xi-uvij));
					}
				}	
		}
		// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
		else if( wt==WeightsType.PRE )
		{
			// approach: iterate over all cells of X maybe sparse and dense
			// (note: tuning similar to pattern 3 possible but more complex)
			for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
				for( int j=0, vix=0; j<n; j++, vix+=cd)
				{
					double xij = mX.quickGetValue(i, j);
					double wij = mW.quickGetValue(i, j);
					double uvij = 0;
					if( wij != 0 )
						uvij = dotProduct(u, v, uix, vix, cd);
					wsloss += (xij-wij*uvij)*(xij-wij*uvij);
				}
		}
		// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
		else if( wt==WeightsType.NONE )
		{
			// approach: iterate over all cells of X and 
			for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd ) 
			{
				if( x[i]==null || x[i].isEmpty() ) { //empty row
					for( int j=0, vix=0; j<n; j++, vix+=cd) {
						double uvij = dotProduct(u, v, uix, vix, cd);
						wsloss += (-uvij)*(-uvij);
					}
				}
				else { //non-empty row
					int xlen = x[i].size();
					int[] xix = x[i].getIndexContainer();
					double[] xval = x[i].getValueContainer();
					int last = -1;
					for( int k=0; k<xlen; k++ ) {
						//process last nnz til current nnz
						for( int k2=last+1; k2<xix[k]; k2++ ){
							double uvij = dotProduct(u, v, uix, k2*cd, cd);
							wsloss += (-uvij)*(-uvij);							
						}
						//process current nnz
						double uvij = dotProduct(u, v, uix, xix[k]*cd, cd);
						wsloss += (xval[k]-uvij)*(xval[k]-uvij);
						last = xix[k];
					}
					//process last nnz til end of row
					for( int k2=last+1; k2<n; k2++ ) { 
						double uvij = dotProduct(u, v, uix, k2*cd, cd);
						wsloss += (-uvij)*(-uvij);							
					}
				}
			}
		}
		
		ret.quickSetValue(0, 0, wsloss);
	}

	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param mW
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWSLossGeneric (MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		final int n = mX.clen; 
		final int cd = mU.clen;
		double wsloss = 0; 
		
		// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
		if( wt==WeightsType.POST )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			if( mW.sparse ) //SPARSE
			{
				SparseRow[] wrows = mW.sparseRows;
				
				for( int i=rl; i<ru; i++ )
					if( wrows[i] != null && !wrows[i].isEmpty() ){
						int wlen = wrows[i].size();
						int[] wix = wrows[i].getIndexContainer();
						double[] wval = wrows[i].getValueContainer();
						for( int k=0; k<wlen; k++ ) {
							double xi = mX.quickGetValue(i, wix[k]);
							double uvij = 0;
							for( int k2=0; k2<cd; k2++ )
								uvij += mU.quickGetValue(i, k2) * mV.quickGetValue(wix[k], k2);
							wsloss += (wval[k]*(xi-uvij))*(wval[k]*(xi-uvij));
						}
					}	
			}
			else //DENSE
			{
				double[] w = mW.denseBlock;
				
				for( int i=rl, wix=rl*n; i<ru; i++, wix+=n )
					for( int j=0; j<n; j++)
					{
						double wij = w[wix+j];
						if( wij != 0 ) {
							double xij = mX.quickGetValue(i, j);
							double uvij = 0;
							for( int k=0; k<cd; k++ )
								uvij += mU.quickGetValue(i, k) * mV.quickGetValue(j, k);
							wsloss += (wij*(xij-uvij))*(wij*(xij-uvij));
						}
					}	
			}
		}
		// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
		else if( wt==WeightsType.PRE )
		{
			// approach: iterate over all cells of X maybe sparse and dense
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++)
				{
					double xij = mX.quickGetValue(i, j);
					double wij = mW.quickGetValue(i, j);
					double uvij = 0;
					if( wij != 0 )
						for( int k=0; k<cd; k++ )
							uvij += mU.quickGetValue(i, k) * mV.quickGetValue(j, k);
					wsloss += (xij-wij*uvij)*(xij-wij*uvij);
				}
		}
		// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
		else if( wt==WeightsType.NONE )
		{
			// approach: iterate over all cells of X and 
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++)
				{
					double xij = mX.quickGetValue(i, j);
					double uvij = 0;
					for( int k=0; k<cd; k++ )
						uvij += mU.quickGetValue(i, k) * mV.quickGetValue(j, k);
					wsloss += (xij-uvij)*(xij-uvij);
				}
		}

		ret.quickSetValue(0, 0, wsloss);
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
	 * 
	 * @param a
	 * @param b
	 * @param c
	 * @param ai
	 * @param bi
	 * @param ci
	 * @param len
	 */
	@SuppressWarnings("unused")
	private static void vectMultiplyWrite( double[] a, double[] b, double[] c, int ai, int bi, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ ci ] = a[ ai ] * b[ bi ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, bi+=8, ci+=8) 
		{
			//read 64B cachelines of a and b
			//compute c' = a * b
			//write back 64B cacheline of c = c'
			c[ ci+0 ] = a[ ai+0 ] * b[ bi+0 ];
			c[ ci+1 ] = a[ ai+1 ] * b[ bi+1 ];
			c[ ci+2 ] = a[ ai+2 ] * b[ bi+2 ];
			c[ ci+3 ] = a[ ai+3 ] * b[ bi+3 ];
			c[ ci+4 ] = a[ ai+4 ] * b[ bi+4 ];
			c[ ci+5 ] = a[ ai+5 ] * b[ bi+5 ];
			c[ ci+6 ] = a[ ai+6 ] * b[ bi+6 ];
			c[ ci+7 ] = a[ ai+7 ] * b[ bi+7 ];
		}
	}

	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 */
	private static void vectMultiply( double[] a, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] *= a[ ai ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a and c
			//compute c' = c * a
			//write back 64B cacheline of c = c'
			c[ ci+0 ] *= a[ ai+0 ];
			c[ ci+1 ] *= a[ ai+1 ];
			c[ ci+2 ] *= a[ ai+2 ];
			c[ ci+3 ] *= a[ ai+3 ];
			c[ ci+4 ] *= a[ ai+4 ];
			c[ ci+5 ] *= a[ ai+5 ];
			c[ ci+6 ] *= a[ ai+6 ];
			c[ ci+7 ] *= a[ ai+7 ];
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 */
	private static void vectAdd( double[] a, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] += a[ ai ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a and c
			//compute c' = c * a
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += a[ ai+0 ];
			c[ ci+1 ] += a[ ai+1 ];
			c[ ci+2 ] += a[ ai+2 ];
			c[ ci+3 ] += a[ ai+3 ];
			c[ ci+4 ] += a[ ai+4 ];
			c[ ci+5 ] += a[ ai+5 ];
			c[ ci+6 ] += a[ ai+6 ];
			c[ ci+7 ] += a[ ai+7 ];
		}
	}
	
	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 * 
	 * @param ret
	 */
	private static void copyUpperToLowerTriangle( MatrixBlock ret )
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

	/////////////////////////////////////////////////////////
	// Task Implementations for Multi-Threaded Operations  //
	/////////////////////////////////////////////////////////
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultTask implements Callable<Object> 
	{
		private MatrixBlock _m1  = null;
		private MatrixBlock _m2  = null;
		private MatrixBlock _ret = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultTask( MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru )
		{
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			if( _m1.isUltraSparse() || _m2.isUltraSparse() )
				matrixMultUltraSparse(_m1, _m2, _ret, _rl, _ru);
			else if(!_m1.sparse && !_m2.sparse)
				matrixMultDenseDense(_m1, _m2, _ret, _rl, _ru);
			else if(_m1.sparse && _m2.sparse)
				matrixMultSparseSparse(_m1, _m2, _ret, _rl, _ru);
			else if(_m1.sparse)
				matrixMultSparseDense(_m1, _m2, _ret, _rl, _ru);
			else
				matrixMultDenseSparse(_m1, _m2, _ret, _rl, _ru);
			
			return null;
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultChainTask implements Callable<Object> 
	{
		private MatrixBlock _m1  = null;
		private MatrixBlock _m2  = null;
		private MatrixBlock _m3  = null;
		private MatrixBlock _ret = null;
		private ChainType _ct = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultChainTask( MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int rl, int ru ) 
			throws DMLRuntimeException
		{
			_m1 = mX;
			_m2 = mV;
			_m3 = mW;
			_ct = ct;
			_rl = rl;
			_ru = ru;
			
			//allocate local result for partial aggregation
			_ret = new MatrixBlock(ret.rlen, ret.clen, false);
			_ret.allocateDenseBlock();
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			if( _m1.sparse )
				matrixMultChainSparse(_m1, _m2, _m3, _ret, _ct, _rl, _ru);
			else
				matrixMultChainDense(_m1, _m2, _m3, _ret, _ct, _rl, _ru);
			
			//NOTE: we dont do global aggregation from concurrent tasks in order
			//to prevent synchronization (sequential aggregation led to better 
			//performance after JIT)
			
			return null;
		}
		
		public MatrixBlock getResult() {
			return _ret;
		}
	}

	/**
	 * 
	 * 
	 */
	private static class PMMTask implements Callable<Object> 
	{
		private MatrixBlock _pm1  = null;
		private MatrixBlock _m2 = null;
		private MatrixBlock _ret1 = null;
		private MatrixBlock _ret2 = null;
		private int _rl = -1;
		private int _ru = -1;

		protected PMMTask( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
		{
			_pm1 = pm1;
			_m2 = m2;
			_ret1 = ret1;
			_ret2 = ret2;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			if( _m2.sparse )
				matrixMultPermuteSparse(_pm1, _m2, _ret1, _ret2, _rl, _ru);
			else if( _ret1.sparse )
				matrixMultPermuteDenseSparse(_pm1, _m2, _ret1, _ret2, _rl, _ru);
			else 
				matrixMultPermuteDense(_pm1, _m2, _ret1, _ret2, _rl, _ru);

			return null;
		}
	}

	/**
	 * 
	 * 
	 */
	private static class WSLMMTask implements Callable<Object> 
	{
		private MatrixBlock _mX = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _mW = null;
		private MatrixBlock _ret = null;
		private WeightsType _wt = null;
		private int _rl = -1;
		private int _ru = -1;

		protected WSLMMTask(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, WeightsType wt, int rl, int ru) 
			throws DMLRuntimeException
		{
			_mX = mX;
			_mU = mU;
			_mV = mV;
			_mW = mW;
			_wt = wt;
			_rl = rl;
			_ru = ru;
			
			//allocate local result for partial aggregation
			_ret = new MatrixBlock(1, 1, false);
			_ret.allocateDenseBlock();
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			if( !_mX.sparse && !_mU.sparse && !_mV.sparse && (_mW==null || !_mW.sparse) 
				&& !_mX.isEmptyBlock() && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() 
				&& (_mW==null || !_mW.isEmptyBlock()))
				matrixMultWSLossDense(_mX, _mU, _mV, _mW, _ret, _wt, _mX.clen, _mU.clen, _rl, _ru);
			else if( _mX.sparse && !_mU.sparse && !_mV.sparse && (_mW==null || _mW.sparse)
				    && !_mX.isEmptyBlock() && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() 
				    && (_mW==null || !_mW.isEmptyBlock()))
				matrixMultWSLossSparseDense(_mX, _mU, _mV, _mW, _ret, _wt, _rl, _ru);
			else
				matrixMultWSLossGeneric(_mX, _mU, _mV, _mW, _ret, _wt, _rl, _ru);

			return null;
		}
		
		public double getWSLoss()
		{
			return _ret.quickGetValue(0, 0);
		}
	}
}