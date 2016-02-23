/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysml.lops.MapMultChain.ChainType;
import org.apache.sysml.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysml.lops.WeightedDivMM.WDivMMType;
import org.apache.sysml.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysml.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysml.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.UtilFunctions;

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
	//internal configuration
	public static final boolean LOW_LEVEL_OPTIMIZATION = true;
	public static final long MEM_OVERHEAD_THRESHOLD = 2L*1024*1024; //MAX 2 MB
	private static final long PAR_MINFLOP_THRESHOLD = 2L*1024*1024; //MIN 2 MFLOP
	
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing: output allocation
		boolean tm2 = checkPrepMatrixMultRightInput(m1,m2);
		m2 = prepMatrixMultRightInput(m1, m2);
		ret.sparse = (m1.isUltraSparse() || m2.isUltraSparse());
		if( !ret.sparse )
			ret.allocateDenseBlock();
		
		//prepare row-upper for special cases of vector-matrix
		boolean pm2 = checkParMatrixMultRightInput(m1, m2, Integer.MAX_VALUE);
		int ru = pm2 ? m2.rlen : m1.rlen; 
		
		//core matrix mult computation
		if( m1.isUltraSparse() || m2.isUltraSparse() )
			matrixMultUltraSparse(m1, m2, ret, 0, ru);
		else if(!m1.sparse && !m2.sparse)
			matrixMultDenseDense(m1, m2, ret, tm2, pm2, 0, ru);
		else if(m1.sparse && m2.sparse)
			matrixMultSparseSparse(m1, m2, ret, pm2, 0, ru);
		else if(m1.sparse)
			matrixMultSparseDense(m1, m2, ret, pm2, 0, ru);
		else
			matrixMultDenseSparse(m1, m2, ret, pm2, 0, ru);
		
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
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check too high additional vector-matrix memory requirements (fallback to sequential)
		//check too small workload in terms of flops (fallback to sequential too)
		if( m1.rlen == 1 && (8L * m2.clen * k > MEM_OVERHEAD_THRESHOLD || !LOW_LEVEL_OPTIMIZATION || m2.clen==1 || m1.isUltraSparse() || m2.isUltraSparse()) 
			|| 2L * m1.rlen * m1.clen * m2.clen < PAR_MINFLOP_THRESHOLD ) 
		{ 
			matrixMult(m1, m2, ret);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing: output allocation (in contrast to single-threaded,
		//we need to allocate sparse as well in order to prevent synchronization)
		boolean tm2 = checkPrepMatrixMultRightInput(m1,m2);
		m2 = prepMatrixMultRightInput(m1, m2);
		ret.sparse = (m1.isUltraSparse() || m2.isUltraSparse());
		if( !ret.sparse )
			ret.allocateDenseBlock();
		else
			ret.allocateSparseRowsBlock();
		
		//prepare row-upper for special cases of vector-matrix / matrix-matrix
		boolean pm2 = checkParMatrixMultRightInput(m1, m2, k);
		int ru = pm2 ? m2.rlen : m1.rlen; 
		
		//core multi-threaded matrix mult computation
		//(currently: always parallelization over number of rows)
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<MatrixMultTask> tasks = new ArrayList<MatrixMultTask>();
			int blklen = (int)(Math.ceil((double)ru/k));
			for( int i=0; i<k & i*blklen<ru; i++ )
				tasks.add(new MatrixMultTask(m1, m2, ret, tm2, pm2, i*blklen, Math.min((i+1)*blklen, ru)));
			pool.invokeAll(tasks);	
			pool.shutdown();
			//aggregate partial results (nnz, ret for vector/matrix)
			ret.nonZeros = 0; //reset after execute
			for( MatrixMultTask task : tasks ) {
				if( pm2 )
					vectAdd(task.getResult().denseBlock, ret.denseBlock, 0, 0, ret.rlen*ret.clen);
				else
					ret.nonZeros += task.getPartialNnz();
			}
			if( pm2 )
				ret.recomputeNonZeros();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//post-processing (nnz maintained in parallel)
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
		if( mX.isEmptyBlock(false) || mV.isEmptyBlock(false) || (mW !=null && mW.isEmptyBlock(false)) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}

		//Timing time = new Timing(true);
				
		//pre-processing: output allocation
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
		if( mX.isEmptyBlock(false) || mV.isEmptyBlock(false) || (mW !=null && mW.isEmptyBlock(false)) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}

		//check too high additional memory requirements (fallback to sequential)
		//check too small workload in terms of flops (fallback to sequential too)
		if( 8L * mV.rlen * k > MEM_OVERHEAD_THRESHOLD 
			|| 4L * mX.rlen * mX.clen < PAR_MINFLOP_THRESHOLD ) 
		{ 
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
		//check inputs / outputs
		if( m1.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing
		m1 = prepMatrixMultTransposeSelfInput(m1, leftTranspose);
		ret.sparse = false;
		ret.allocateDenseBlock();

		if( m1.sparse )
			matrixMultTransposeSelfSparse(m1, ret, leftTranspose, 0, ret.rlen);
		else 
			matrixMultTransposeSelfDense(m1, ret, leftTranspose, 0, ret.rlen );

		//post-processing
		copyUpperToLowerTriangle( ret );		
		ret.recomputeNonZeros();
		ret.examSparsity();	
		
		//System.out.println("TSMM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());
	}

	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param leftTranspose
	 * @param k
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultTransposeSelf( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int k )
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//check inputs / outputs
		if( m1.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check no parallelization benefit (fallback to sequential)
		//check too small workload in terms of flops (fallback to sequential too)
		if( ret.rlen == 1 
			|| leftTranspose && 1L * m1.rlen * m1.clen * m1.clen < PAR_MINFLOP_THRESHOLD
			|| !leftTranspose && 1L * m1.clen * m1.rlen * m1.rlen < PAR_MINFLOP_THRESHOLD ) 
		{ 
			matrixMultTransposeSelf(m1, ret, leftTranspose);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing
		m1 = prepMatrixMultTransposeSelfInput(m1, leftTranspose);
		ret.sparse = false;
		ret.allocateDenseBlock();
	
		//core multi-threaded matrix mult computation
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<MatrixMultTransposeTask> tasks = new ArrayList<MatrixMultTransposeTask>();
			//load balance via #tasks=2k due to triangular shape 
			int blklen = (int)(Math.ceil((double)ret.rlen/(2*k)));
			for( int i=0; i<2*k & i*blklen<ret.rlen; i++ )
				tasks.add(new MatrixMultTransposeTask(m1, ret, leftTranspose, i*blklen, Math.min((i+1)*blklen, ret.rlen)));
			pool.invokeAll(tasks);	
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//post-processing
		copyUpperToLowerTriangle( ret );		
		ret.recomputeNonZeros();
		ret.examSparsity();	
		
		//System.out.println("TSMM k="+k+" ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());
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
			ArrayList<MatrixMultPermuteTask> tasks = new ArrayList<MatrixMultPermuteTask>();
			int blklen = (int)(Math.ceil((double)pm1.rlen/k));
			for( int i=0; i<k & i*blklen<pm1.rlen; i++ )
				tasks.add(new MatrixMultPermuteTask(pm1, m2, ret1, ret2, i*blklen, Math.min((i+1)*blklen, pm1.rlen)));
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
		//                   "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
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
		if( wt==WeightsType.POST && mW.isEmptyBlock(false) 
			|| wt==WeightsType.POST_NZ && mX.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//core weighted square sum mm computation
		if( !mX.sparse && !mU.sparse && !mV.sparse && (mW==null || !mW.sparse) 
			&& !mX.isEmptyBlock() && !mU.isEmptyBlock() && !mV.isEmptyBlock() && (mW==null || !mW.isEmptyBlock()))
			matrixMultWSLossDense(mX, mU, mV, mW, ret, wt, 0, mX.rlen);
		else if( mX.sparse && !mU.sparse && !mV.sparse && (mW==null || mW.sparse)
				&& !mX.isEmptyBlock() && !mU.isEmptyBlock() && !mV.isEmptyBlock() && (mW==null || !mW.isEmptyBlock()))
			matrixMultWSLossSparseDense(mX, mU, mV, mW, ret, wt, 0, mX.rlen);
		else
			matrixMultWSLossGeneric(mX, mU, mV, mW, ret, wt, 0, mX.rlen);
		
		//System.out.println("MMWSLoss " +wt.toString()+ " ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                  "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
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
		if( wt==WeightsType.POST && mW.isEmptyBlock(false)
			|| wt==WeightsType.POST_NZ && mX.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check no parallelization benefit (fallback to sequential)
		if (mX.rlen == 1) {
			matrixMultWSLoss(mX, mU, mV, mW, ret, wt);
			return;
		}
		
		//Timing time = new Timing(true);
		
		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<ScalarResultTask> tasks = new ArrayList<ScalarResultTask>();
			int blklen = (int)(Math.ceil((double)mX.rlen/k));
			for( int i=0; i<k & i*blklen<mX.rlen; i++ )
				tasks.add(new MatrixMultWSLossTask(mX, mU, mV, mW, wt, i*blklen, Math.min((i+1)*blklen, mX.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
			//aggregate partial results
			sumScalarResults(tasks, ret);
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		//System.out.println("MMWSLoss "+wt.toString()+" k="+k+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWSigmoid(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateDenseOrSparseBlock();
		
		//core weighted square sum mm computation
		if( !mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWSigmoidDense(mW, mU, mV, ret, wt, 0, mW.rlen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock())
			matrixMultWSigmoidSparseDense(mW, mU, mV, ret, wt, 0, mW.rlen);
		else
			matrixMultWSigmoidGeneric(mW, mU, mV, ret, wt, 0, mW.rlen);
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMWSig "+wt.toString()+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param k
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWSigmoid(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int k) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//check no parallelization benefit (fallback to sequential)
		if (mW.rlen == 1) {
			matrixMultWSigmoid(mW, mU, mV, ret, wt);
			return;
		}
		
		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateDenseOrSparseBlock();
		
		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<MatrixMultWSigmoidTask> tasks = new ArrayList<MatrixMultWSigmoidTask>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWSigmoidTask(mW, mU, mV, ret, wt, i*blklen, Math.min((i+1)*blklen, mW.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
			ret.nonZeros = 0; //reset after execute
			for( MatrixMultWSigmoidTask task : tasks )
				ret.nonZeros += task.getPartialNnz();
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		//post-processing (nnz maintained in parallel)
		ret.examSparsity();

		//System.out.println("MMWSig "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop() + ".");
	}
	
	/**
	 * NOTE: This operation has limited NaN support, which is acceptable because all our sparse-safe operations
	 * have only limited NaN support. If this is not intended behavior, please disable the rewrite. In detail, 
	 * this operator will produce for W/(U%*%t(V)) a zero intermediate for each zero in W (even if UVij is zero 
	 * which would give 0/0=NaN) but INF/-INF for non-zero entries in V where the corresponding cell in (Y%*%X) 
	 * is zero.
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWDivMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt) 
		throws DMLRuntimeException 
	{
		//check for empty result 
		if(   mW.isEmptyBlock(false) 
		   || (wt.isLeft() && mU.isEmptyBlock(false))
		   || (wt.isRight() && mV.isEmptyBlock(false))
		   || (wt.isBasic() && mW.isEmptyBlock(false)))  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = wt.isBasic()?mW.sparse:false;
		ret.allocateDenseOrSparseBlock();
		
		//core weighted div mm computation
		if( !mW.sparse && !mU.sparse && !mV.sparse && (mX==null || !mX.sparse) && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWDivMMDense(mW, mU, mV, mX, ret, wt, 0, mW.rlen, 0, mW.clen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && (mX==null || mX.sparse) && !mU.isEmptyBlock() && !mV.isEmptyBlock())
			matrixMultWDivMMSparseDense(mW, mU, mV, mX, ret, wt, 0, mW.rlen, 0, mW.clen);
		else
			matrixMultWDivMMGeneric(mW, mU, mV, mX, ret, wt, 0, mW.rlen, 0, mW.clen);
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMWDiv "+wt.toString()+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}
	
	/**
	 * NOTE: This operation has limited NaN support, which is acceptable because all our sparse-safe operations
	 * have only limited NaN support. If this is not intended behavior, please disable the rewrite. In detail, 
	 * this operator will produce for W/(U%*%t(V)) a zero intermediate for each zero in W (even if UVij is zero 
	 * which would give 0/0=NaN) but INF/-INF for non-zero entries in V where the corresponding cell in (Y%*%X) 
	 * is zero.
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param k
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWDivMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int k) 
		throws DMLRuntimeException 
	{
		//check for empty result 
		if(   mW.isEmptyBlock(false) 
		   || (wt.isLeft() && mU.isEmptyBlock(false))
		   || (wt.isRight() && mV.isEmptyBlock(false)) 
		   || (wt.isBasic() && mW.isEmptyBlock(false)))  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = wt.isBasic()?mW.sparse:false;
		ret.allocateDenseOrSparseBlock();

		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<MatrixMultWDivTask> tasks = new ArrayList<MatrixMultWDivTask>();			
			//create tasks (for wdivmm-left, parallelization over columns;
			//for wdivmm-right, parallelization over rows; both ensure disjoint results)
			if( wt.isLeft() ) {
				int blklen = (int)(Math.ceil((double)mW.clen/k));
				for( int j=0; j<k & j*blklen<mW.clen; j++ )
					tasks.add(new MatrixMultWDivTask(mW, mU, mV, mX, ret, wt, 0, mW.rlen, j*blklen, Math.min((j+1)*blklen, mW.clen)));
			}
			else { //basic/right
				int blklen = (int)(Math.ceil((double)mW.rlen/k));
				for( int i=0; i<k & i*blklen<mW.rlen; i++ )
					tasks.add(new MatrixMultWDivTask(mW, mU, mV, mX, ret, wt, i*blklen, Math.min((i+1)*blklen, mW.rlen), 0, mW.clen));
			}
			//execute tasks
			pool.invokeAll(tasks);
			pool.shutdown();
			//aggregate partial nnz 
			ret.nonZeros = 0;  //reset after execute
			for( MatrixMultWDivTask task : tasks )
				ret.nonZeros += task.getPartialNnz();
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		//post-processing
		ret.examSparsity();
		
		//System.out.println("MMWDiv "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}	

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWCeMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WCeMMType wt) 
		throws DMLRuntimeException 
	{
		//check for empty result 
		if( mW.isEmptyBlock(false) )  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		//core weighted div mm computation
		if( !mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWCeMMDense(mW, mU, mV, ret, wt, 0, mW.rlen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock())
			matrixMultWCeMMSparseDense(mW, mU, mV, ret, wt, 0, mW.rlen);
		else
			matrixMultWCeMMGeneric(mW, mU, mV, ret, wt, 0, mW.rlen);
		
		//System.out.println("MMWCe "+wt.toString()+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}
	
	/**
	 * 
	 * @param mX
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param k
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWCeMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WCeMMType wt, int k) 
		throws DMLRuntimeException 
	{
		//check for empty result 
		if( mW.isEmptyBlock(false) )  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<ScalarResultTask> tasks = new ArrayList<ScalarResultTask>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWCeTask(mW, mU, mV, wt, i*blklen, Math.min((i+1)*blklen, mW.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
			//aggregate partial results
			sumScalarResults(tasks, ret);
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}
		
		//System.out.println("MMWCe "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWuMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateDenseOrSparseBlock();
		
		//core weighted square sum mm computation
		if( !mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWuMMDense(mW, mU, mV, ret, wt, fn, 0, mW.rlen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock())
			matrixMultWuMMSparseDense(mW, mU, mV, ret, wt, fn, 0, mW.rlen);
		else
			matrixMultWuMMGeneric(mW, mU, mV, ret, wt, fn, 0, mW.rlen);
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMWu "+wt.toString()+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param k
	 * @throws DMLRuntimeException
	 */
	public static void matrixMultWuMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int k) 
		throws DMLRuntimeException 
	{
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//check no parallelization benefit (fallback to sequential)
		if (mW.rlen == 1) {
			matrixMultWuMM(mW, mU, mV, ret, wt, fn);
			return;
		}
		
		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateDenseOrSparseBlock();
		
		try 
		{			
			ExecutorService pool = Executors.newFixedThreadPool(k);
			ArrayList<MatrixMultWuTask> tasks = new ArrayList<MatrixMultWuTask>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWuTask(mW, mU, mV, ret, wt, fn, i*blklen, Math.min((i+1)*blklen, mW.rlen)));
			pool.invokeAll(tasks);
			pool.shutdown();
			ret.nonZeros = 0; //reset after execute
			for( MatrixMultWuTask task : tasks )
				ret.nonZeros += task.getPartialNnz();
		} 
		catch (InterruptedException e) {
			throw new DMLRuntimeException(e);
		}

		//post-processing (nnz maintained in parallel)
		ret.examSparsity();

		//System.out.println("MMWu "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop() + ".");
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
	private static void matrixMultDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean tm2, boolean pm2, int rl, int ru) 
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
			else if( pm2 && m==1 )     //VECTOR-MATRIX
			{
				//parallelization over rows in rhs matrix
				//rest not aligned to blocks of 2 rows
				final int kn = (ru-rl)%2;
				if( kn == 1 && a[rl] != 0 )
					vectMultiplyAdd(a[rl], b, c, rl*n, 0, n);
				
				//compute blocks of 2 rows (2 instead of 4 for small n<64) 
				for( int k=rl+kn, bix=(rl+kn)*n; k<ru; k+=2, bix+=2*n ){
					if( a[k] != 0 && a[k+1] != 0  )
						vectMultiplyAdd2(a[k], a[k+1], b, c, bix, bix+n, 0, n);
					else if( a[k] != 0 )
						vectMultiplyAdd(a[k], b, c, bix, 0, n);
					else if( a[k+1] != 0 )	
						vectMultiplyAdd(a[k+1], b, c, bix+n, 0, n);
				}
			}
			else if( pm2 && m<=16 )    //MATRIX-MATRIX (short lhs) 
			{
				//parallelization over rows in rhs matrix
				final int kn = (ru-rl)%2;				
				
				//rest not aligned to blocks of 2 rows
				if( kn == 1 )
					for( int i=0, aix=0, cix=0; i<m; i++, aix+=cd, cix+=n )
						if( a[aix+rl] != 0 )
							vectMultiplyAdd(a[aix+rl], b, c, rl*n, cix, n);
				
				//compute blocks of 2 rows (w/ repeated scan for each row in lhs) 
				for( int k=rl+kn, bix=(rl+kn)*n; k<ru; k+=2, bix+=2*n )
					for( int i=0, aix=0, cix=0; i<m; i++, aix+=cd, cix+=n ){
						if( a[aix+k] != 0 && a[aix+k+1] != 0  )
							vectMultiplyAdd2(a[aix+k], a[aix+k+1], b, c, bix, bix+n, cix, n);
						else if( a[aix+k] != 0 )
							vectMultiplyAdd(a[aix+k], b, c, bix, cix, n);
						else if( a[aix+k+1] != 0 )	
							vectMultiplyAdd(a[aix+k+1], b, c, bix+n, cix, n);
					}				
			}
			else if( tm2 )             //MATRIX-MATRIX (skinny rhs)
			{
				//note: prepared rhs input via transpose for: m > n && cd > 64 && n < 64
				//however, explicit flag required since dimension change m2
				final int n2 = m2.rlen;
				for( int i=rl, aix=rl*cd, cix=rl*n2; i < ru; i++, aix+=cd, cix+=n2 ) 
					for( int j=0, bix=0; j<n2; j++, bix+=cd )
						c[cix+j] = dotProduct(a, b, aix, bix, cd);
			}
			else                       //MATRIX-MATRIX
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
	private static void matrixMultDenseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl, int ru) 
		throws DMLRuntimeException 
	{	
		double[] a = m1.denseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int cd = m1.clen;
		int n = m2.clen;
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if( LOW_LEVEL_OPTIMIZATION )
		{
			final int blocksizeI = 32; //256KB c block (typical L2 size per core), 32KB a block 
			final int blocksizeK = 32; 
			//note: in contrast to dense-dense, no blocking over j (would require maintaining blocksizeK indexes, counter-productive on skew)
			
			SparseBlock b = m2.sparseBlock;
			
			if( pm2 && m==1 )          //VECTOR-MATRIX
			{
				//parallelization over rows in rhs matrix
				for( int k=rl; k<ru; k++ )
					if( a[k] != 0 && !b.isEmpty(k) ) {								
						vectMultiplyAdd(a[k], b.values(k), c, b.indexes(k), b.pos(k), 0, b.size(k));
					}
			}
			else                       //MATRIX-MATRIX
			{
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
								if( val != 0 && !b.isEmpty(bk+k) ) {
									vectMultiplyAdd(val, b.values(bk+k), c, b.indexes(bk+k), b.pos(bk+k), cixj, b.size(bk+k));
								}
							}
			    		}
					}
			}
		}
		else
		{
			SparseBlock b = m2.sparseBlock;
			for( int i=rl, aix=rl*cd, cix=rl*n; i < ru; i++, cix+=n ) 
				for(int k = 0; k < cd; k++, aix++ ) 
				{
					double val = a[aix];
					if( val!=0 )
					{
						if( !b.isEmpty(k) ) 
						{
							int bpos = b.pos(k);
							int blen = b.size(k);
							int[] bix = b.indexes(k);
							double[] bvals = b.values(k);	
							for(int j = bpos; j < bpos+blen; j++)
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
	private static void matrixMultSparseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl, int ru) 
		throws DMLRuntimeException
	{	
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m2.rlen;

		if( LOW_LEVEL_OPTIMIZATION )
		{
			SparseBlock a = m1.sparseBlock;
			
			if( m==1 && n==1 )         //DOT PRODUCT
			{
				if( !a.isEmpty(0) ) {
					c[0] = dotProduct(a.values(0), b, a.indexes(0), a.pos(0), 0, a.size(0));
				}
			}
			else if( n==1 )            //MATRIX-VECTOR
			{
				for( int i=rl; i<Math.min(ru, a.numRows()); i++ )
				{
					if( !a.isEmpty(i) ) {
						c[i] = dotProduct(a.values(i), b, a.indexes(i), a.pos(i), 0, a.size(i));							
					}
				}
			}
			else if( pm2 && m==1 )     //VECTOR-MATRIX
			{
				//parallelization over rows in rhs matrix
				if( !a.isEmpty(0) ) 
				{
					int alen = a.size(0);
					int[] aix = a.indexes(0);
					double[] avals = a.values(0);					
					int rlix = (rl==0) ? 0 : a.posFIndexGTE(0,rl);
					rlix = (rlix>=0) ? rlix : alen;
					
					for( int k=rlix; k<alen && aix[k]<ru; k++ ) {
						if( k+1<alen && aix[k+1]<ru )
							vectMultiplyAdd2(avals[k], avals[k+1], b, c, aix[k]*n, aix[++k]*n, 0, n);
						else
							vectMultiplyAdd(avals[k], b, c, aix[k]*n, 0, n);
					}
				}
			}
			else if( pm2 && m<=16 )    //MATRIX-MATRIX (short lhs) 
			{
				for( int i=0, cix=0; i<a.numRows(); i++, cix+=n )
					if( !a.isEmpty(i) ) 
					{
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);					
						
						int k1 = (rl==0) ? apos : a.posFIndexGTE(i, rl);
						k1 = (k1>=0) ? k1 : apos+alen;
						int k2 = (ru==cd) ? apos+alen : a.posFIndexGTE(i, ru);
						k2 = (k2>=0) ? k2 : apos+alen;
						
						//rest not aligned to blocks of 4 rows
		    			final int bn = (k2-k1) % 4;
		    			switch( bn ){
			    			case 1: vectMultiplyAdd(avals[k1], b, c, aix[k1]*n, cix, n); break;
			    	    	case 2: vectMultiplyAdd2(avals[k1],avals[k1+1], b, c, aix[k1]*n, aix[k1+1]*n, cix, n); break;
			    			case 3: vectMultiplyAdd3(avals[k1],avals[k1+1],avals[k1+2], b, c, aix[k1]*n, aix[k1+1]*n, aix[k1+2]*n, cix, n); break;
		    			}
		    			
		    			//compute blocks of 4 rows (core inner loop)
		    			for( int k = k1+bn; k<k2; k+=4 ) {
		    				vectMultiplyAdd4( avals[k], avals[k+1], avals[k+2], avals[k+3], b, c, 
		    						          aix[k]*n, aix[k+1]*n, aix[k+2]*n, aix[k+3]*n, cix, n );
		    			}
					}
			}
			else                       //MATRIX-MATRIX
			{
				for( int i=rl, cix=rl*n; i<Math.min(ru, a.numRows()); i++, cix+=n )
				{
					if( !a.isEmpty(i) ) 
					{
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);					
						
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
				    			case 1: vectMultiplyAdd(avals[apos], b, c, aix[apos]*n, cix, n); break;
				    	    	case 2: vectMultiplyAdd2(avals[apos],avals[apos+1], b, c, aix[apos]*n, aix[apos+1]*n, cix, n); break;
				    			case 3: vectMultiplyAdd3(avals[apos],avals[apos+1],avals[apos+2], b, c, aix[apos]*n, aix[apos+1]*n, aix[apos+2]*n, cix, n); break;
			    			}
			    			
			    			//compute blocks of 4 rows (core inner loop)
			    			for( int k = apos+bn; k<apos+alen; k+=4 ) {
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
			SparseBlock a = m1.sparseBlock;
			for( int i=rl, cix=rl*n; i<Math.min(ru, a.numRows()); i++, cix+=n )
			{
				if( !a.isEmpty(i) ) 
				{
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);					
					
					for(int k = apos; k < apos+alen; k++) {
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
	private static void matrixMultSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl, int ru) 
		throws DMLRuntimeException
	{	
		SparseBlock a = m1.sparseBlock;
		SparseBlock b = m2.sparseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m2.clen;
		
		//TODO perf sparse block
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if(LOW_LEVEL_OPTIMIZATION)
		{
			if( pm2 && m==1 )          //VECTOR-MATRIX
			{
				//parallelization over rows in rhs matrix
				if( !a.isEmpty(0) ) 
				{
					int alen = a.size(0);
					int[] aix = a.indexes(0);
					double[] avals = a.values(0);					
					int rlix = (rl==0) ? 0 : a.posFIndexGTE(0,rl);
					rlix = (rlix>=0) ? rlix : alen;
					
					for( int k=rlix; k<alen && aix[k]<ru; k++ )
						if( !b.isEmpty(aix[k]) ) {
							int bpos = b.pos(aix[k]);
							int blen = b.size(aix[k]);
							int[] bix = b.indexes(aix[k]);
							double[] bvals = b.values(aix[k]);								
							vectMultiplyAdd(avals[k], bvals, c, bix, bpos, 0, blen);
						}			
				}
			}	
			else                       //MATRIX-MATRIX
			{
				for( int i=rl, cix=rl*n; i<Math.min(ru, a.numRows()); i++, cix+=n )
				{
					if( !a.isEmpty(i) ) 
					{
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);					
						
						for(int k = apos; k < apos+alen; k++) 
						{
							double val = avals[k];
							if( !b.isEmpty(aix[k]) ) 
							{
								int bpos = b.pos(aix[k]);
								int blen = b.size(aix[k]);
								int[] bix = b.indexes(aix[k]);
								double[] bvals = b.values(aix[k]);	
								
								vectMultiplyAdd(val, bvals, c, bix, bpos, cix, blen);
							}
						}						
					}
				}
			}
		}
		else
		{
			for( int i=rl, cix=rl*n; i<Math.min(ru, a.numRows()); i++, cix+=n )
			{
				if( !a.isEmpty(i) ) 
				{
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);					
					
					for(int k = apos; k < apos+alen; k++) 
					{
						double val = avals[k];
						if( !b.isEmpty(aix[k]) ) 
						{
							int bpos = b.pos(aix[k]);
							int blen = b.size(aix[k]);
							int[] bix = b.indexes(aix[k]);
							double[] bvals = b.values(aix[k]);	
							for(int j = bpos; j < bpos+blen; j++)
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
		//TODO perf sparse block, consider iterators
		
		boolean leftUS = m1.isUltraSparse();
		final int m  = m1.rlen;
		final int cd = m1.clen;
		final int n  = m2.clen;
		
		if( leftUS ) //left is ultra-sparse (IKJ)
		{
			SparseBlock a = m1.sparseBlock;
			boolean rightSparse = m2.sparse;
			
			for( int i=rl; i<ru; i++ )
			{
				if( !a.isEmpty(i) ) 
				{
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aixs = a.indexes(i);
					double[] avals = a.values(i);
					
					if( alen==1 && avals[apos]==1 ) //ROW SELECTION (no aggregation)
					{
						int aix = aixs[apos];
						if( rightSparse ) { //sparse right matrix (full row copy)
							if( !m2.sparseBlock.isEmpty(aix) ) {
								ret.rlen=m;
								ret.allocateSparseRowsBlock(false); //allocation on demand
								ret.sparseBlock.set(i, m2.sparseBlock.get(aix), true); 
								ret.nonZeros += ret.sparseBlock.size(i);
							}
						}
						else { //dense right matrix (append all values)
							for( int j=0; j<n; j++ )
								ret.appendValue(i, j, m2.quickGetValue(aix, j));
						}
					}
					else //GENERAL CASE
					{
						for( int k=apos; k<apos+alen; k++ )
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
			SparseBlock b = m2.sparseBlock;
			
			for(int k = 0; k < cd; k++ ) 
			{			
				if( !b.isEmpty(k) ) 
				{
					int bpos = b.pos(k);
					int blen = b.size(k);
					int[] bixs = b.indexes(k);
					double[] bvals = b.values(k);								
					for( int j=bpos; j<bpos+blen; j++ )
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
		boolean weights2 = (ct == ChainType.XtXvy);
		
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
			
			//multiply/subtract weights (in-place), if required
			if( weights ) 
				vectMultiply(w, tmp, bi, 0, blocksize);	
			else if( weights2 )
				vectSubtract(w, tmp, bi, 0, blocksize);
				
			//compute 2nd matrix vector for row block and aggregate
			for (int j=0, aix=bi*cd; j < blocksize; j+=4, aix+=4*cd)
				vectMultiplyAdd4(tmp[j], tmp[j+1], tmp[j+2], tmp[j+3], a, c, aix, aix+cd, aix+2*cd, aix+3*cd, 0, cd);
		}
		
		//compute rest (not aligned to blocksize)
		for( int i=bn, aix=bn*cd; i < ru; i++, aix+=cd ) {
			double val = dotProduct(a, b, aix, 0, cd);
			val *= (weights) ? w[i] : 1;
			val -= (weights2) ? w[i] : 0;
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
		SparseBlock a = mX.sparseBlock;
		double[] b = mV.denseBlock;
		double[] w = (mW!=null) ? mW.denseBlock : null;
		double[] c = ret.denseBlock;
		boolean weights = (ct == ChainType.XtwXv);
		boolean weights2 = (ct == ChainType.XtXvy);
		
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
				if( !a.isEmpty(bi+j) ) {
					int apos = a.pos(bi+j);
					int alen = a.size(bi+j);				
					tmp[j] = dotProduct(a.values(bi+j), b, a.indexes(bi+j), apos, 0, alen);							
				}
			}
			
			//multiply weights (in-place), if required
			if( weights ) 
				vectMultiply(w, tmp, bi, 0, tmplen);	
			else if( weights2 )
				vectSubtract(w, tmp, bi, 0, tmplen);
		
			//compute 2nd matrix vector for row block and aggregate
			for( int j=0; j < tmplen; j++) {
				if( !a.isEmpty(bi+j) && tmp[j] != 0 ) {
					int apos = a.pos(bi+j);
					int alen = a.size(bi+j);
					int[] aix = a.indexes(bi+j);
					double[] avals = a.values(bi+j);		
					vectMultiplyAdd(tmp[j], avals, c, aix, apos, 0, alen);							
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
	private static void matrixMultTransposeSelfDense( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int rl, int ru ) 
		throws DMLRuntimeException
	{
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
					//1) Unrolled inner loop (for better instruction-level parallelism)
					//2) Blocked execution (for less cache trashing in parallel exec) 	
					//3) Asymmetric block sizes (for less misses in inner loop, yet blocks in L1/L2)
					
					final int blocksizeI = 32; //64//256KB c block (typical L2 size per core), 32KB a block 
					final int blocksizeK = 24; //64//256KB b block (typical L2 size per core), used while read 512B of a / read/write 4KB of c 
					final int blocksizeJ = 1024; //512//4KB (typical main-memory page size), for scan 

					//temporary arrays (nnz a, b index)
					double[] ta = new double[ blocksizeK ];
					int[]  tbi  = new int[ blocksizeK ];
					
					final int mx = ru;
					final int cdx = m;
					final int nx = n;
					
					//blocked execution
					for( int bi = rl; bi < mx; bi+=blocksizeI ) //from bi due to symmetry
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
				for(int k = 0, ix1 = 0; k < m; k++, ix1+=n)
					for(int i = rl, ix3 = 0; i < ru; i++, ix3+=n) 
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
					for( int bi = rl; bi<ru; bi+=blocksize )
						for( int bj = bi; bj<m; bj+=blocksize ) 
						{
							final int bimin = Math.min(ru, bi+blocksize);
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
				for(int i = rl, ix1 = 0, ix3 = 0; i < ru; i++, ix1+=n, ix3+=m)
					for(int j = i, ix2 = i*n; j < m; j++, ix2+=n) //from i due to symmetry
					{
						double val = 0;
						for(int k = 0; k < n; k++)
							val += a[ ix1+k ] * a[ix2+k];
						c[ ix3+j ] = val;	
					}
			}
		}
	}
	
	/**
	 * 
	 * @param out
	 * @param leftTranspose
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultTransposeSelfSparse( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int rl, int ru ) 
		throws DMLRuntimeException
	{
		//2) transpose self matrix multiply sparse
		// (compute only upper-triangular matrix due to symmetry)		
		SparseBlock a = m1.sparseBlock;
		double[] c = ret.denseBlock;
		int m = m1.rlen;
		int n = m1.clen;

		if( leftTranspose ) // t(X)%*%X 
		{
			//only general case (because vectors always dense)
			//algorithm: scan rows, foreach row self join (KIJ)
			if( LOW_LEVEL_OPTIMIZATION )
			{
				for( int r=0; r<a.numRows(); r++ )
					if( !a.isEmpty(r) ) 
					{
						int apos = a.pos(r);
						int alen = a.size(r);
						int[] aix = a.indexes(r);
						double[] avals = a.values(r);					
						int rlix = (rl==0) ? apos : a.posFIndexGTE(r, rl);
						rlix = (rlix>=0) ? rlix : apos+alen;
						
						for(int i = rlix; i < apos+alen && aix[i]<ru; i++) 
						{
							double val = avals[i];
							if( val != 0 ) {
								int ix2 = aix[i]*n;
								vectMultiplyAdd(val, avals, c, aix, i, ix2, alen-i);
							}
						}
					}
			}
			else
			{
				for( int r=0; r<a.numRows(); r++ )
					if( !a.isEmpty(r) ) 
					{
						int apos = a.pos(r);
						int alen = a.size(r);
						int[] aix = a.indexes(r);
						double[] avals = a.values(r);					
						int rlix = (rl==0) ? apos : a.posFIndexGTE(r, rl);
						rlix = (rlix>=0) ? rlix : apos+alen;
						
						for(int i = rlix; i < apos+alen && aix[i]<ru; i++) 
						{
							double val = avals[i];
							if( val != 0 )
								for(int j = i, ix2 = aix[i]*n; j < apos+alen; j++)
									c[ix2+aix[j]] += val * avals[j];
						}
					}
			}
		}
		else // X%*%t(X)
		{
			if( m==1 ) //VECTOR 
			{
				if( !m1.sparseBlock.isEmpty(0) ) {
					int alen = m1.sparseBlock.size(0); //pos always 0
					double[] avals = a.values(0);	
					c[0] = dotProduct(avals, avals, alen);
				}
			}
			else //MATRIX
			{			
				//note: reorg to similar layout as t(X)%*%X because faster than 
				//direct computation with IJK (no dependencies/branches in inner loop)
				//see preprocessMatrixMultTransposeSelf m1<-tmpBlock
				m = m1.clen;
				n = m1.rlen;
				
				//algorithm: scan rows, foreach row self join (KIJ)
				if( LOW_LEVEL_OPTIMIZATION )
				{
					for( int r=0; r<a.numRows(); r++ )
						if( !a.isEmpty(r) ) 
						{
							int apos = a.pos(r);
							int alen = a.size(r);
							int[] aix = a.indexes(r);
							double[] avals = a.values(r);					
							int rlix = (rl==0) ? apos : a.posFIndexGTE(r, rl);
							rlix = (rlix>=0) ? rlix : apos+alen;
							
							for(int i = rlix; i < apos+alen && aix[i]<ru; i++) 
							{
								double val = avals[i];
								if( val != 0 ) {
									int ix2 = aix[i]*m;
									vectMultiplyAdd(val, avals, c, aix, i, ix2, alen-i);
								}
							}
						}
				}
				else
				{
					for( int r=0; r<a.numRows(); r++ )
						if( !a.isEmpty(r) ) 
						{
							int apos = a.pos(r);
							int alen = a.size(r);
							int[] aix = a.indexes(r);
							double[] avals = a.values(r);					
							int rlix = (rl==0) ? apos : a.posFIndexGTE(r,rl);
							rlix = (rlix>=0) ? rlix : apos+alen;
							
							for(int i = rlix; i < apos+alen && aix[i]<ru; i++) 
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
		SparseBlock c = ret1.sparseBlock;

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
					c = ret2.sparseBlock;		
				}
		
				//append entire dense row into sparse target position
				for( int j=0; j<n; j++ )
					c.append(bpos, j, b[bix+j]);
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
		SparseBlock b = m2.sparseBlock;
		SparseBlock c = ret1.sparseBlock;

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
					c = ret2.sparseBlock;		
				}
		
				//memcopy entire sparse row into target position
				c.set(bpos, b.get(i), true);
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
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWSLossDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		double[] x = mX.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		double[] w = (mW!=null)? mW.denseBlock : null;
		final int n = mX.clen;
		final int cd = mU.clen;
		double wsloss = 0;
		
		// approach: iterate over all cells of X 
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
					
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		

		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = 0, bimin = Math.min(ru, bi+blocksizeIJ); bj < n; bj+=blocksizeIJ ) 
			{
				int bjmin = Math.min(n, bj+blocksizeIJ);
								
				// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
				if( wt==WeightsType.POST )
				{
					for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
						for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
							double wij = w[ix+j];
							if( wij != 0 ) {
								double uvij = dotProduct(u, v, uix, vix, cd);
								wsloss += wij*(x[ix+j]-uvij)*(x[ix+j]-uvij); //^2
							}
						}	
				}
				// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post_nz weighting)
				else if( wt==WeightsType.POST_NZ )
				{
					for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
						for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
							double xij = x[ix+j];
							if( xij != 0 ) {
								double uvij = dotProduct(u, v, uix, vix, cd);
								wsloss += (xij-uvij)*(xij-uvij); //^2
							}
						}	
				}
				// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
				else if( wt==WeightsType.PRE )
				{
					for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
						for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
							double wij = w[ix+j];
							double uvij = 0;
							if( wij != 0 )
								uvij = dotProduct(u, v, uix, vix, cd);
							wsloss += (x[ix+j]-wij*uvij)*(x[ix+j]-wij*uvij); //^2
						}
				}
				// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
				else if( wt==WeightsType.NONE )
				{
					for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
						for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
							double uvij = dotProduct(u, v, uix, vix, cd);
							wsloss += (x[ix+j]-uvij)*(x[ix+j]-uvij); //^2
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
	private static void matrixMultWSLossSparseDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		SparseBlock x = mX.sparseBlock;
		SparseBlock w = (mW!=null)? mW.sparseBlock : null;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int n = mX.clen; 
		final int cd = mU.clen;
		double wsloss = 0; 
		
		// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
		if( wt==WeightsType.POST )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
				if( !w.isEmpty(i) ) {
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					if( w.isAligned(i, x) ) {
						//O(n) where n is nnz in w/x 
						double[] xval = x.values(i);
						for( int k=wpos; k<wpos+wlen; k++ ) {
							double uvij = dotProduct(u, v, uix, wix[k]*cd, cd);
							wsloss += wval[k]*(xval[k]-uvij)*(xval[k]-uvij);
						}		
					}
					else {
						//O(n log m) where n/m is nnz in w/x 
						for( int k=wpos; k<wpos+wlen; k++ ) {
							double xi = mX.quickGetValue(i, wix[k]);
							double uvij = dotProduct(u, v, uix, wix[k]*cd, cd);
							wsloss += wval[k]*(xi-uvij)*(xi-uvij);
						}	
					}
				}	
		}
		// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post weighting)
		else if( wt==WeightsType.POST_NZ )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
				if( !x.isEmpty(i) ) {
					int xpos = x.pos(i);
					int xlen = x.size(i);
					int[] xix = x.indexes(i);
					double[] xval = x.values(i);
					for( int k=xpos; k<xpos+xlen; k++ ) {
						double uvij = dotProduct(u, v, uix, xix[k]*cd, cd);
						wsloss += (xval[k]-uvij)*(xval[k]-uvij);
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
				if( x.isEmpty(i) ) { //empty row
					for( int j=0, vix=0; j<n; j++, vix+=cd) {
						double uvij = dotProduct(u, v, uix, vix, cd);
						wsloss += (-uvij)*(-uvij);
					}
				}
				else { //non-empty row
					int xpos = x.pos(i);
					int xlen = x.size(i);
					int[] xix = x.indexes(i);
					double[] xval = x.values(i);
					int last = -1;
					for( int k=xpos; k<xpos+xlen; k++ ) {
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
				SparseBlock w = mW.sparseBlock;
				
				for( int i=rl; i<ru; i++ )
					if( !w.isEmpty(i) ) {
						int wpos = w.pos(i);
						int wlen = w.size(i);
						int[] wix = w.indexes(i);
						double[] wval = w.values(i);
						for( int k=wpos; k<wpos+wlen; k++ ) {
							double uvij = dotProductGeneric(mU, mV, i, wix[k], cd);
							double xi = mX.quickGetValue(i, wix[k]);
							wsloss += wval[k]*(xi-uvij)*(xi-uvij);
						}
					}	
			}
			else //DENSE
			{
				double[] w = mW.denseBlock;
				
				for( int i=rl, wix=rl*n; i<ru; i++, wix+=n )
					for( int j=0; j<n; j++)
						if( w[wix+j] != 0 ) {
							double uvij = dotProductGeneric(mU, mV, i, j, cd);
							double xij = mX.quickGetValue(i, j);
							wsloss += w[wix+j]*(xij-uvij)*(xij-uvij);
						}
			}
		}
		// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post weighting)
		else if( wt==WeightsType.POST_NZ )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			if( mW.sparse ) //SPARSE
			{
				SparseBlock x = mX.sparseBlock;
				
				for( int i=rl; i<ru; i++ )
					if( !x.isEmpty(i) ) {
						int xpos = x.pos(i);
						int xlen = x.size(i);
						int[] xix = x.indexes(i);
						double[] xval = x.values(i);
						for( int k=xpos; k<xpos+xlen; k++ ) {
							double uvij = dotProductGeneric(mU, mV, i, xix[k], cd);
							wsloss += (xval[k]-uvij)*(xval[k]-uvij);
						}
					}	
			}
			else //DENSE
			{
				double[] x = mX.denseBlock;
				
				for( int i=rl, xix=rl*n; i<ru; i++, xix+=n )
					for( int j=0; j<n; j++)
						if( x[xix+j] != 0 ) {
							double uvij = dotProductGeneric(mU, mV, i, j, cd);
							wsloss += (x[xix+j]-uvij)*(x[xix+j]-uvij);
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
						uvij = dotProductGeneric(mU, mV, i, j, cd);
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
					double uvij = dotProductGeneric(mU, mV, i, j, cd);
					wsloss += (xij-uvij)*(xij-uvij);
				}
		}

		ret.quickSetValue(0, 0, wsloss);
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultWSigmoidDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) 
		throws DMLRuntimeException 
	{	
		double[] w = mW.denseBlock;
		double[] c = ret.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int n = mW.clen;
		final int cd = mU.clen;
		
		//note: cannot compute U %*% t(V) in-place of result w/ regular mm because
		//t(V) comes in transformed form and hence would require additional memory
	
		boolean flagminus = (wt==WSigmoidType.MINUS || wt==WSigmoidType.LOG_MINUS); 
		boolean flaglog = (wt==WSigmoidType.LOG || wt==WSigmoidType.LOG_MINUS);
		
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		
		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = 0, bimin = Math.min(ru, bi+blocksizeIJ); bj < n; bj+=blocksizeIJ ) 
			{
				int bjmin = Math.min(n, bj+blocksizeIJ);
						
				//core wsigmoid computation
				for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
					for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
						double wij = w[ix+j];
						if( wij != 0 )
							c[ix+j] = wsigmoid(wij, u, v, uix, vix, flagminus, flaglog, cd);
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWSigmoidSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) 
		throws DMLRuntimeException
	{
		SparseBlock w = mW.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int cd = mU.clen;
		
		boolean flagminus = (wt==WSigmoidType.MINUS || wt==WSigmoidType.LOG_MINUS); 
		boolean flaglog = (wt==WSigmoidType.LOG || wt==WSigmoidType.LOG_MINUS);
	
		//approach: iterate over non-zeros of w, selective mm computation
		for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
			if( !w.isEmpty(i) ) {
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				
				c.allocate(i, wlen);
				for( int k=wpos; k<wpos+wlen; k++ ) {
					double cval = wsigmoid(wval[k], u, v, uix, wix[k]*cd, flagminus, flaglog, cd);
					c.append(i, wix[k], cval);
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWSigmoidGeneric (MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) 
		throws DMLRuntimeException
	{
		final int n = mW.clen; 
		final int cd = mU.clen;
	
		boolean flagminus = (wt==WSigmoidType.MINUS || wt==WSigmoidType.LOG_MINUS); 
		boolean flaglog = (wt==WSigmoidType.LOG || wt==WSigmoidType.LOG_MINUS);
	
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			//w and c always in same representation
			SparseBlock w = mW.sparseBlock;
			SparseBlock c = ret.sparseBlock;
			
			for( int i=rl; i<ru; i++ )
				if( !w.isEmpty(i) ) {
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					
					c.allocate(i, wlen);
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double cval = wsigmoid(wval[k], mU, mV, i, wix[k], flagminus, flaglog, cd);
						c.append(i, wix[k], cval);
					}
				}	
		}
		else //DENSE
		{
			//w and c always in same representation
			double[] w = mW.denseBlock;
			double[] c = ret.denseBlock;
		
			for( int i=rl, ix=rl*n; i<ru; i++ )
				for( int j=0; j<n; j++, ix++) {
					double wij = w[ix];
					if( wij != 0 ) {
						c[ix] = wsigmoid(wij, mU, mV, i, j, flagminus, flaglog, cd);
					}
				}
		}
	}
	
	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultWDivMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) 
		throws DMLRuntimeException 
	{	
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft();
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final int n = mW.clen;
		final int cd = mU.clen;
		
		double[] w = mW.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		double[] x = (mX==null) ? null : mX.denseBlock;
		double[] c = ret.denseBlock;
		
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		
		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = cl, bimin = Math.min(ru, bi+blocksizeIJ); bj < cu; bj+=blocksizeIJ ) 
			{
				int bjmin = Math.min(cu, bj+blocksizeIJ);
						
				//core wsigmoid computation
				for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
					for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd)
						if( w[ix+j] != 0 ) {
							if( basic ) 
								c[ix+j] = w[ix+j] * dotProduct(u, v, uix, vix, cd);	
							else if( four ) //left/right 
								if (scalar)
									wdivmm(w[ix+j], x[0], u, v, c, uix, vix, left, scalar, cd);
								else
									wdivmm(w[ix+j], x[ix+j], u, v, c, uix, vix, left, scalar, cd);
							else //left/right minus/default
								wdivmm(w[ix+j], u, v, c, uix, vix, left, mult, minus, cd);
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWDivMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) 
		throws DMLRuntimeException
	{
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft();
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final double eps = scalar ? mX.quickGetValue(0, 0) : 0;
		final int cd = mU.clen;
		
		SparseBlock w = mW.sparseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		double[] c = ret.denseBlock;
		SparseBlock x = (mX==null) ? null : mX.sparseBlock;
		
		//approach: iterate over non-zeros of w, selective mm computation
		for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd ) {
			if( !w.isEmpty(i) ) {
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
			
				if( basic ) {
					for( int k=wpos; k<wpos+wlen; k++ )
						ret.appendValue( i, wix[k], wval[k] * dotProduct(u, v, uix, wix[k]*cd, cd));
				}
				else if( four ) { //left/right
					int k = (cl==0) ? wpos : w.posFIndexGTE(i,cl);
					k = (k>=0) ? k : wpos+wlen;
					//checking alignment per row is ok because early abort if false, 
					//row nnz likely fit in L1/L2 cache, and asymptotically better if aligned
					if( w.isAligned(i, x) ) {
						//O(n) where n is nnz in w/x 
						double[] xvals = x.values(i);
						for( ; k<wpos+wlen && wix[k]<cu; k++ )
							if (scalar)
								wdivmm(wval[k], eps, u, v, c, uix, wix[k]*cd, left, scalar, cd);
							else
								wdivmm(wval[k], xvals[k], u, v, c, uix, wix[k]*cd, left, scalar, cd);
					}
					else {
						//O(n log m) where n/m are nnz in w/x
						for( ; k<wpos+wlen && wix[k]<cu; k++ )
							if (scalar)
								wdivmm(wval[k], eps, u, v, c, uix, wix[k]*cd, left, scalar, cd);
							else
								wdivmm(wval[k], x.get(i, wix[k]), u, v, c, uix, wix[k]*cd, left, scalar, cd);
					}
				}
				else { //left/right minus default
					int k = (cl==0) ? wpos : w.posFIndexGTE(i,cl);
					k = (k>=0) ? k : wpos+wlen;
					for( ; k<wpos+wlen && wix[k]<cu; k++ )
						wdivmm(wval[k], u, v, c, uix, wix[k]*cd, left, mult, minus, cd);
				}
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWDivMMGeneric(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) 
		throws DMLRuntimeException
	{
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft(); 
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final double eps = scalar ? mX.quickGetValue(0, 0) : 0;
		final int n = mW.clen; 
		final int cd = mU.clen;

		//output always in dense representation
		double[] c = ret.denseBlock;
		
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			SparseBlock w = mW.sparseBlock;
			
			for( int i=rl; i<ru; i++ ) {
				if( !w.isEmpty(i) ) {
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					int k = (cl==0) ? wpos : w.posFIndexGTE(i,cl);
					k = (k>=0) ? k : wpos+wlen;
					for( ; k<wpos+wlen && wix[k]<cu; k++ ) { 
						if( basic ) {
							double uvij = dotProductGeneric(mU,mV, i, wix[k], cd);
							ret.appendValue(i, wix[k], uvij);
						}
						else if( four ) { //left/right
							double xij = scalar ? eps : mX.quickGetValue(i, wix[k]);
							wdivmm(wval[k], xij, mU, mV, c, i, wix[k], left, scalar, cd);
						}
						else { //left/right minus/default
							wdivmm(wval[k], mU, mV, c, i, wix[k], left, mult, minus, cd);
						}
					}
				}	
			}
		}
		else //DENSE
		{
			double[] w = mW.denseBlock;
		
			for( int i=rl, ix=rl*n; i<ru; i++, ix+=n )
				for( int j=cl; j<cu; j++)
					if( w[ix+j] != 0 ) {
						if( basic ) {
							c[ix+j] = dotProductGeneric(mU,mV, i, j, cd);
						}
						else if( four ) { //left/right
							double xij = scalar ? eps : mX.quickGetValue(i, j);
							wdivmm(w[ix+j], xij, mU, mV, c, i, j, left, scalar, cd);
						}
						else { //left/right minus/default
							wdivmm(w[ix+j], mU, mV, c, i, j, left, mult, minus, cd);
						}
					}
		}
	}
	
	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWCeMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		double[] w = mW.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int n = mW.clen;
		final int cd = mU.clen;
		double wceval = 0;
		
		// approach: iterate over all cells of X 
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 

		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = 0, bimin = Math.min(ru, bi+blocksizeIJ); bj < n; bj+=blocksizeIJ ) 
			{
				int bjmin = Math.min(n, bj+blocksizeIJ);
								
				for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
					for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
						double wij = w[ix+j];
						if( wij != 0 ) {
							double uvij = dotProduct(u, v, uix, vix, cd);
							wceval += wij * FastMath.log(uvij);
						}
					}
		}
		
		ret.quickSetValue(0, 0, wceval);
	}
	
	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 */
	private static void matrixMultWCeMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		SparseBlock w = mW.sparseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int cd = mU.clen;
		double wceval = 0; 
		
		// approach: iterate over all cells of X and 
		for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd ) {
			if( !w.isEmpty(i) ) { 
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				for( int k=wpos; k<wpos+wlen; k++ ) {
					double uvij = dotProduct(u, v, uix, wix[k]*cd, cd);
					wceval += wval[k] * FastMath.log(uvij);					
				}
			}
		}
		
		ret.quickSetValue(0, 0, wceval);
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
	private static void matrixMultWCeMMGeneric(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		final int n = mW.clen; 
		final int cd = mU.clen;
		double wceval = 0; 
		
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			SparseBlock w = mW.sparseBlock;
			
			for( int i=rl; i<ru; i++ )
				if( !w.isEmpty(i) ) {
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double uvij = dotProductGeneric(mU, mV, i, wix[k], cd);
						wceval += wval[k] * FastMath.log(uvij);	
					}
				}	
		}
		else //DENSE
		{
			double[] w = mW.denseBlock;
		
			for( int i=rl, ix=rl*n; i<ru; i++ )
				for( int j=0; j<n; j++, ix++) {
					double wij = w[ix];
					if( wij != 0 ) {
						double uvij = dotProductGeneric(mU, mV, i, j, cd);
						wceval += wij * FastMath.log(uvij);	
					}
				}
		}
		

		ret.quickSetValue(0, 0, wceval);
	}

	/**
	 * 
	 * @param mW
	 * @param mU
	 * @param mV
	 * @param ret
	 * @param wt
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException
	 */
	private static void matrixMultWuMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) 
		throws DMLRuntimeException 
	{	
		double[] w = mW.denseBlock;
		double[] c = ret.denseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int n = mW.clen;
		final int cd = mU.clen;
		
		//note: cannot compute U %*% t(V) in-place of result w/ regular mm because
		//t(V) comes in transformed form and hence would require additional memory
	
		boolean flagmult = (wt==WUMMType.MULT); 
		
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		
		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ )
			for( int bj = 0, bimin = Math.min(ru, bi+blocksizeIJ); bj < n; bj+=blocksizeIJ ) 
			{
				int bjmin = Math.min(n, bj+blocksizeIJ);
						
				//core wsigmoid computation
				for( int i=bi, ix=bi*n, uix=bi*cd; i<bimin; i++, ix+=n, uix+=cd )
					for( int j=bj, vix=bj*cd; j<bjmin; j++, vix+=cd) {
						double wij = w[ix+j];
						if( wij != 0 )
							c[ix+j] = wumm(wij, u, v, uix, vix, flagmult, fn, cd);
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWuMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) 
		throws DMLRuntimeException
	{
		SparseBlock w = mW.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		double[] u = mU.denseBlock;
		double[] v = mV.denseBlock;
		final int cd = mU.clen;
		
		boolean flagmult = (wt==WUMMType.MULT); 
	
		//approach: iterate over non-zeros of w, selective mm computation
		for( int i=rl, uix=rl*cd; i<ru; i++, uix+=cd )
			if( !w.isEmpty(i) ) {
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				
				c.allocate(i, wlen);
				for( int k=wpos; k<wpos+wlen; k++ ) {
					double cval = wumm(wval[k], u, v, uix, wix[k]*cd, flagmult, fn, cd);
					c.append(i, wix[k], cval);
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
	 * @param rl
	 * @param ru
	 * @throws DMLRuntimeException 
	 */
	private static void matrixMultWuMMGeneric (MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) 
		throws DMLRuntimeException
	{
		final int n = mW.clen; 
		final int cd = mU.clen;
	
		boolean flagmult = (wt==WUMMType.MULT); 
		
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			//w and c always in same representation
			SparseBlock w = mW.sparseBlock;
			SparseBlock c = ret.sparseBlock;
			
			for( int i=rl; i<ru; i++ )
				if( !w.isEmpty(i) ) {
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					
					c.allocate(i, wlen); 
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double cval = wumm(wval[k], mU, mV, i, wix[k], flagmult, fn, cd);
						c.append(i, wix[k], cval);
					}
				}	
		}
		else //DENSE
		{
			//w and c always in same representation
			double[] w = mW.denseBlock;
			double[] c = ret.denseBlock;
		
			for( int i=rl, ix=rl*n; i<ru; i++ )
				for( int j=0; j<n; j++, ix++) {
					double wij = w[ix];
					if( wij != 0 ) {
						c[ix] = wumm(wij, mU, mV, i, j, flagmult, fn, cd);
					}
				}
		}
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
	
	private static double dotProduct( double[] a, double[] b, int[] aix, int ai, final int bi, final int len )
	{
		double val = 0;
		final int bn = len%8;
				
		//compute rest
		for( int i = ai; i < ai+bn; i++ )
			val += a[ i ] * b[ bi+aix[i] ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = ai+bn; i < ai+len; i+=8 )
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
	@SuppressWarnings("unused")
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
	
	/**
	 * 
	 * @param aval
	 * @param b
	 * @param c
	 * @param bix
	 * @param bi
	 * @param ci
	 * @param len
	 */
	private static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			c[ ci + bix[j] ] += aval * b[ j ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < bi+len; j+=8 )
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
	 * 
	 * @param a1
	 * @param a2
	 * @param a3
	 * @param a4
	 * @param c
	 * @param ai
	 * @param ci
	 * @param len
	 */
	private static void vectAdd4( double[] a1, double[] a2, double[] a3, double[] a4, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] += a1[ ai ] + a2[ ai ] + a3[ ai ] + a4[ ai ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a (4x) and c
			//compute c' = c + a1 + a2 + a3 + a4
			//write back 64B cacheline of c = c'
			c[ ci+0 ] += a1[ ai+0 ] + a2[ ai+0 ] + a3[ ai+0 ] + a4[ ai+0 ];
			c[ ci+1 ] += a1[ ai+1 ] + a2[ ai+1 ] + a3[ ai+1 ] + a4[ ai+1 ];
			c[ ci+2 ] += a1[ ai+2 ] + a2[ ai+2 ] + a3[ ai+2 ] + a4[ ai+2 ];
			c[ ci+3 ] += a1[ ai+3 ] + a2[ ai+3 ] + a3[ ai+3 ] + a4[ ai+3 ];
			c[ ci+4 ] += a1[ ai+4 ] + a2[ ai+4 ] + a3[ ai+4 ] + a4[ ai+4 ];
			c[ ci+5 ] += a1[ ai+5 ] + a2[ ai+5 ] + a3[ ai+5 ] + a4[ ai+5 ];
			c[ ci+6 ] += a1[ ai+6 ] + a2[ ai+6 ] + a3[ ai+6 ] + a4[ ai+6 ];
			c[ ci+7 ] += a1[ ai+7 ] + a2[ ai+7 ] + a3[ ai+7 ] + a4[ ai+7 ];
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
	private static void vectSubtract( double[] a, double[] c, int ai, int ci, final int len )
	{
		final int bn = len%8;
		
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] -= a[ ai ];
		
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ai+=8, ci+=8) 
		{
			//read 64B cachelines of a and c
			//compute c' = c * a
			//write back 64B cacheline of c = c'
			c[ ci+0 ] -= a[ ai+0 ];
			c[ ci+1 ] -= a[ ai+1 ];
			c[ ci+2 ] -= a[ ai+2 ];
			c[ ci+3 ] -= a[ ai+3 ];
			c[ ci+4 ] -= a[ ai+4 ];
			c[ ci+5 ] -= a[ ai+5 ];
			c[ ci+6 ] -= a[ ai+6 ];
			c[ ci+7 ] -= a[ ai+7 ];
		}
	}

	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param uix
	 * @param vix
	 * @param flagminus
	 * @param flaglog
	 * @param len
	 * @return
	 */
	private static double wsigmoid( final double wij, double[] u, double[] v, final int uix, final int vix, final boolean flagminus, final boolean flaglog, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute core sigmoid function  
		double cval = flagminus ?
				1 / (1 + FastMath.exp(uvij)) :
				1 / (1 + FastMath.exp(-uvij));
				
		//compute weighted output
		return wij * ((flaglog) ? FastMath.log(cval) : cval);
	}
	
	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param uix
	 * @param vix
	 * @param flagminus
	 * @param flaglog
	 * @param len
	 * @return
	 */
	private static double wsigmoid( final double wij, MatrixBlock u, MatrixBlock v, final int uix, final int vix, final boolean flagminus, final boolean flaglog, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);
		
		//compute core sigmoid function  
		double cval = flagminus ?
				1 / (1 + FastMath.exp(uvij)) :
				1 / (1 + FastMath.exp(-uvij));
				
		//compute weighted output
		return wij * ((flaglog) ? FastMath.log(cval) : cval);
	}
	
	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param c
	 * @param uix
	 * @param vix
	 * @param flagleft
	 * @param len
	 */
	private static void wdivmm( final double wij, double[] u, double[] v, double[] c, final int uix, final int vix, final boolean left, final boolean mult, final boolean minus, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute core wdivmm  
		double tmpval = minus ? uvij - wij :
			            mult ? wij * uvij : wij / uvij;
		
		//prepare inputs for final mm
		int bix = left ? uix : vix;
		int cix = left ? vix : uix;
		double[] b = left ? u : v;		
		
		//compute final mm output
		vectMultiplyAdd(tmpval, b, c, bix, cix, len);
	}
	
	/**
	 * 
	 * @param wij
	 * @param xij
	 * @param u
	 * @param v
	 * @param c
	 * @param uix
	 * @param vix
	 * @param left
	 * @param len
	 */
	private static void wdivmm( final double wij, final double xij, double[] u, double[] v, double[] c, final int uix, final int vix, final boolean left, final boolean scalar, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute core wdivmm  
		double tmpval = scalar ? wij / (uvij + xij) : wij * (uvij - xij);
		
		//prepare inputs for final mm
		int bix = left ? uix : vix;
		int cix = left ? vix : uix;
		double[] b = left ? u : v;		
		
		//compute final mm output
		vectMultiplyAdd(tmpval, b, c, bix, cix, len);
	}


	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param c
	 * @param uix
	 * @param vix
	 * @param flagleft
	 * @param len
	 */
	private static void wdivmm( final double wij, MatrixBlock u, MatrixBlock v, double[] c, final int uix, final int vix, final boolean left, boolean mult, final boolean minus, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);
		
		//compute core wdivmm
		double wtmp = minus ? uvij - wij :
					  mult ? wij * uvij : wij / uvij;
		
		//prepare inputs for final mm
		int bix = left ? uix : vix;
		int cix = left ? vix*len : uix*len;
		MatrixBlock b = left ? u : v;		
		
		//compute final mm
		for( int k2=0; k2<len; k2++ )
			c[cix+k2] += b.quickGetValue(bix, k2) * wtmp;
	}
	
	/**
	 * 
	 * @param wij
	 * @param xij
	 * @param u
	 * @param v
	 * @param c
	 * @param uix
	 * @param vix
	 * @param left
	 * @param len
	 */
	private static void wdivmm( final double wij, final double xij, MatrixBlock u, MatrixBlock v, double[] c, final int uix, final int vix, final boolean left, final boolean scalar, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);
		
		//compute core wdivmm
		double wtmp = scalar ? wij / (uvij + xij) : wij * (uvij - xij);
		
		//prepare inputs for final mm
		int bix = left ? uix : vix;
		int cix = left ? vix*len : uix*len;
		MatrixBlock b = left ? u : v;		
		
		//compute final mm
		for( int k2=0; k2<len; k2++ )
			c[cix+k2] += b.quickGetValue(bix, k2) * wtmp;
	}
	
	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param uix
	 * @param vix
	 * @param flagmult
	 * @param fn
	 * @param len
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static double wumm( final double wij, double[] u, double[] v, final int uix, final int vix, final boolean flagmult, ValueFunction fn, final int len ) 
		throws DMLRuntimeException
	{
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute unary operations
		double cval = fn.execute(uvij);
		
		//compute weighted output
		return flagmult ? wij * cval : wij / cval;
	}
	
	/**
	 * 
	 * @param wij
	 * @param u
	 * @param v
	 * @param uix
	 * @param vix
	 * @param flagminus
	 * @param flaglog
	 * @param len
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static double wumm( final double wij, MatrixBlock u, MatrixBlock v, final int uix, final int vix, final boolean flagmult, ValueFunction fn, final int len ) 
		throws DMLRuntimeException
	{
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);

		//compute unary operations
		double cval = fn.execute(uvij);
		
		//compute weighted output
		return flagmult ? wij * cval : wij / cval;
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
	private static double dotProductGeneric(MatrixBlock a, MatrixBlock b, final int ai, final int bi, int len)
	{
		double val = 0;
		for( int k2=0; k2<len; k2++ )
			val += a.quickGetValue(ai, k2) * b.quickGetValue(bi, k2);
		
		return val;
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
	 * @param m1
	 * @param leftTranspose
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock prepMatrixMultTransposeSelfInput( MatrixBlock m1, boolean leftTranspose ) 
		throws DMLRuntimeException
	{
		MatrixBlock ret = m1;
		
		if( !leftTranspose && m1.sparse && m1.rlen > 1) //X%*%t(X) SPARSE MATRIX
		{	
			//directly via LibMatrixReorg in order to prevent sparsity change
			MatrixBlock tmpBlock = new MatrixBlock(m1.clen, m1.rlen, m1.sparse);
			LibMatrixReorg.reorg(m1, tmpBlock, new ReorgOperator(SwapIndex.getSwapIndexFnObject()));
			ret = tmpBlock;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	private static boolean checkPrepMatrixMultRightInput( MatrixBlock m1, MatrixBlock m2 )
	{
		//transpose if dense-dense, skinny rhs matrix (not vector), and memory guarded by output 
		return (!m1.sparse && !m2.sparse && m1.rlen>m2.clen && m2.rlen > 64 && m2.clen > 1 && m2.clen < 64);
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	private static boolean checkParMatrixMultRightInput( MatrixBlock m1, MatrixBlock m2, int k )
	{
		//parallelize over rows in rhs matrix if number of rows in lhs/output is very small
		return (m1.rlen==1 && LOW_LEVEL_OPTIMIZATION && m2.clen>1 && !(m1.isUltraSparse()||m2.isUltraSparse()))
			|| (m1.rlen<=16 && LOW_LEVEL_OPTIMIZATION && m2.clen>1 && m2.rlen > m1.rlen 
			   && ( !m1.isUltraSparse() && !m2.sparse ) //dense-dense / sparse/dense
			   && (long)k * 8 * m1.rlen * m2.clen < MEM_OVERHEAD_THRESHOLD ); 
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock prepMatrixMultRightInput( MatrixBlock m1, MatrixBlock m2 ) 
		throws DMLRuntimeException
	{
		MatrixBlock ret = m2;
		
		//transpose if dense-dense, skinny rhs matrix (not vector), and memory guarded by output 
		if( checkPrepMatrixMultRightInput(m1, m2)  ) {
			MatrixBlock tmpBlock = new MatrixBlock(m2.clen, m2.rlen, m2.sparse);
			LibMatrixReorg.reorg(m2, tmpBlock, new ReorgOperator(SwapIndex.getSwapIndexFnObject()));
			ret = tmpBlock;
		}
		
		return ret;
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

	/**
	 * 
	 * @param tasks
	 * @param ret
	 */
	private static void sumScalarResults(ArrayList<ScalarResultTask> tasks, MatrixBlock ret)
	{
		//aggregate partial results
		double val = 0;
		for(ScalarResultTask task : tasks)
			val += task.getScalarResult();
		ret.quickSetValue(0, 0, val);
	}
	
	/**
	 * 
	 * @param partret
	 * @param ret
	 */
	@SuppressWarnings("unused")
	private static void sumDenseResults( double[][] partret, double[] ret )
	{
		final int len = ret.length;
		final int k = partret.length;
		final int bk = k % 4;
		final int blocksize = 2 * 1024; //16KB (half of common L1 data)
		
		//cache-conscious aggregation to prevent repreated scans/writes of ret
		for( int bi=0; bi<len; bi+=blocksize ) {
			int llen = Math.min(len-bi, blocksize);
			
			//aggregate next block from all partial results
			for( int j=0; j<bk; j++ ) //rest (not aligned to 4)
				vectAdd(partret[j], ret, bi, bi, llen);
			for( int j=bk; j<k; j+=4 ) //4 partial results at a time
				vectAdd4(partret[j], partret[j+1], partret[j+2], partret[j+3], ret, bi, bi, llen);
		}
		
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
		private boolean _tm2 = false; //transposed m2
		private boolean _pm2 = false; //par over m2
		private int _rl = -1;
		private int _ru = -1;
		private long _nnz = -1;

		protected MatrixMultTask( MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean tm2, boolean pm2, int rl, int ru )
		{
			_m1 = m1;
			_m2 = m2;
			_tm2 = tm2;
			_pm2 = pm2;
			_rl = rl;
			_ru = ru;
			
			if( pm2 ) { //vector-matrix / matrix-matrix
				//allocate local result for partial aggregation
				_ret = new MatrixBlock(ret.rlen, ret.clen, false);
			}
			else { //default case
				_ret = ret;
			}
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			//thread-local allocation
			if( _pm2 )
				_ret.allocateDenseBlock();
			
			//compute block matrix multiplication
			if( _m1.isUltraSparse() || _m2.isUltraSparse() )
				matrixMultUltraSparse(_m1, _m2, _ret, _rl, _ru);
			else if(!_m1.sparse && !_m2.sparse)
				matrixMultDenseDense(_m1, _m2, _ret, _tm2, _pm2, _rl, _ru);
			else if(_m1.sparse && _m2.sparse)
				matrixMultSparseSparse(_m1, _m2, _ret, _pm2, _rl, _ru);
			else if(_m1.sparse)
				matrixMultSparseDense(_m1, _m2, _ret, _pm2, _rl, _ru);
			else
				matrixMultDenseSparse(_m1, _m2, _ret, _pm2, _rl, _ru);
			
			//maintain block nnz (upper bounds inclusive)
			if( !_pm2 )
				_nnz = _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
			
			return null;
		}
		
		public long getPartialNnz(){
			return _nnz;
		}
		
		public MatrixBlock getResult(){
			return _ret;
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

	private static class MatrixMultTransposeTask implements Callable<Object> 
	{
		private MatrixBlock _m1  = null;
		private MatrixBlock _ret = null;
		private boolean _left = true;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultTransposeTask( MatrixBlock m1, MatrixBlock ret, boolean left, int rl, int ru )
		{
			_m1 = m1;
			_ret = ret;
			_left = left;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			if( _m1.sparse )
				matrixMultTransposeSelfSparse(_m1, _ret, _left, _rl, _ru);
			else
				matrixMultTransposeSelfDense(_m1, _ret, _left, _rl, _ru);
				
			return null;
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultPermuteTask implements Callable<Object> 
	{
		private MatrixBlock _pm1  = null;
		private MatrixBlock _m2 = null;
		private MatrixBlock _ret1 = null;
		private MatrixBlock _ret2 = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultPermuteTask( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
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
	 */
	private static interface ScalarResultTask extends Callable<Object>{
		public double getScalarResult();
	}
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultWSLossTask implements ScalarResultTask
	{
		private MatrixBlock _mX = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _mW = null;
		private MatrixBlock _ret = null;
		private WeightsType _wt = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultWSLossTask(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, WeightsType wt, int rl, int ru) 
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
				matrixMultWSLossDense(_mX, _mU, _mV, _mW, _ret, _wt, _rl, _ru);
			else if( _mX.sparse && !_mU.sparse && !_mV.sparse && (_mW==null || _mW.sparse)
				    && !_mX.isEmptyBlock() && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() 
				    && (_mW==null || !_mW.isEmptyBlock()))
				matrixMultWSLossSparseDense(_mX, _mU, _mV, _mW, _ret, _wt, _rl, _ru);
			else
				matrixMultWSLossGeneric(_mX, _mU, _mV, _mW, _ret, _wt, _rl, _ru);

			return null;
		}
		
		@Override
		public double getScalarResult() {
			return _ret.quickGetValue(0, 0);
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultWSigmoidTask implements Callable<Object> 
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _ret = null;
		private WSigmoidType _wt = null;
		private int _rl = -1;
		private int _ru = -1;
		private long _nnz = -1;
		
		protected MatrixMultWSigmoidTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) 
			throws DMLRuntimeException
		{
			_mW = mW;
			_mU = mU;
			_mV = mV;
			_ret = ret;
			_wt = wt;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			//core weighted square sum mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWSigmoidDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWSigmoidSparseDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else
				matrixMultWSigmoidGeneric(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			
			//maintain block nnz (upper bounds inclusive)
			_nnz = _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
			
			return null;
		}
		
		public long getPartialNnz(){
			return _nnz;
		}
	}
	
	/**
	 * 
	 * 
	 */
	private static class MatrixMultWDivTask implements Callable<Object> 
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _mX = null;
		private MatrixBlock _ret = null;
		private WDivMMType _wt = null;
		private int _rl = -1;
		private int _ru = -1;
		private int _cl = -1;
		private int _cu = -1;
		private long _nnz = -1;
		
		protected MatrixMultWDivTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) 
			throws DMLRuntimeException
		{
			_mW = mW;
			_mU = mU;
			_mV = mV;
			_mX = mX;
			_wt = wt;
			_rl = rl;
			_ru = ru;
			_cl = cl;
			_cu = cu;
			_ret = ret;	
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			//core weighted div mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && (_mX==null || !_mX.sparse) && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWDivMMDense(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && (_mX==null || _mX.sparse) && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWDivMMSparseDense(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
			else
				matrixMultWDivMMGeneric(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
		
			//maintain partial nnz for right (upper bounds inclusive)
			int rl = _wt.isLeft() ? _cl : _rl;
			int ru = _wt.isLeft() ? _cu : _ru;
			_nnz = _ret.recomputeNonZeros(rl, ru-1, 0, _ret.getNumColumns()-1);
				
			return null;
		}
		
		/**
		 * For wdivmm right.
		 * @return
		 */
		public long getPartialNnz(){
			return _nnz;
		}
	}
	
	private static class MatrixMultWCeTask implements ScalarResultTask
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _ret = null;
		private WCeMMType _wt = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultWCeTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, WCeMMType wt, int rl, int ru) 
			throws DMLRuntimeException
		{
			_mW = mW;
			_mU = mU;
			_mV = mV;
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
			//core weighted div mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWCeMMDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWCeMMSparseDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else
				matrixMultWCeMMGeneric(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			
			
			return null;
		}
		
		@Override
		public double getScalarResult() {
			return _ret.quickGetValue(0, 0);
		}
	}
	
	/**
	 * 
	 */
	private static class MatrixMultWuTask implements Callable<Object> 
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _ret = null;
		private WUMMType _wt = null;
		private ValueFunction _fn = null;
		private int _rl = -1;
		private int _ru = -1;
		private long _nnz = -1;
		
		protected MatrixMultWuTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) 
			throws DMLRuntimeException
		{
			_mW = mW;
			_mU = mU;
			_mV = mV;
			_ret = ret;
			_wt = wt;
			_fn = fn;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			//core weighted square sum mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWuMMDense(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWuMMSparseDense(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			else
				matrixMultWuMMGeneric(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			
			//maintain block nnz (upper bounds inclusive)
			_nnz = _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
			
			return null;
		}
		
		public long getPartialNnz() {
			return _nnz;
		}
	}
}