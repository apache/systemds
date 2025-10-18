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

package org.apache.sysds.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowScalar;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.NativeHelper;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

/**
 * MB: Library for matrix multiplications including MM, MV, VV for all
 * combinations of dense, sparse, ultrasparse representations and special
 * operations such as transpose-self matrix multiplication.
 * <p>
 * In general all implementations use internally dense outputs
 * for direct access, but change the final result to sparse if necessary.
 * The only exceptions are ultra-sparse matrix mult, wsloss and wsigmoid.
 */
public class LibMatrixMult 
{
	//internal configuration
	private static final long MEM_OVERHEAD_THRESHOLD = 2L*1024*1024; //MAX 2 MB
	public static final long PAR_MINFLOP_THRESHOLD1 = 2L*1024*1024; //MIN 2 MFLOP
	private static final long PAR_MINFLOP_THRESHOLD2 = 128L*1024; //MIN 2 MFLOP
	public static final int L2_CACHESIZE = 256 * 1024; //256KB (common size)
	public static final int L3_CACHESIZE = 16 * 1024 * 1024; //16MB (common size)
	private static final Log LOG = LogFactory.getLog(LibMatrixMult.class.getName());
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int vLen = SPECIES.length();

	private LibMatrixMult() {
		//prevent instantiation via private constructor
	}
	
	////////////////////////////////
	// public matrix mult interface
	////////////////////////////////
	
	/**
	 * Performs a matrix multiplication
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2) {
		return matrixMult(m1, m2, null, false, 1);
	}

	/**
	 * Performs a matrix multiplication
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param k maximum parallelism
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, int k) {
		return matrixMult(m1, m2, null, false, k);
	}

	/**
	 * Performs a matrix multiplication and stores the result in the output matrix.
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret) {
		return matrixMult(m1, m2, ret, false, 1);
	}
	
	/**
	 * This method allows one to disabling exam sparsity. This feature is useful if matrixMult is used as an intermediate
	 * operation (for example: LibMatrixDNN). It makes sense for LibMatrixDNN because the output is internally
	 * consumed by another dense instruction, which makes repeated conversion to sparse wasteful.
	 * This should be used in rare cases and if you are unsure,
	 * use the method 'matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret)' instead.
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @param fixedRet if true, output representation is fixed and nnzs not recomputed
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean fixedRet) {
		return matrixMult(m1, m2, ret, fixedRet, 1);
	}
	
	/**
	 * Performs a multi-threaded matrix multiplication and stores the result in the output matrix.
	 * The parameter k (k&gt;=1) determines the max parallelism k' with k'=min(k, vcores, m1.rlen).
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @param k maximum parallelism
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		if(NativeHelper.isNativeLibraryLoaded())
			return LibMatrixNative.matrixMult(m1, m2, ret, k);
		else
			return matrixMult(m1, m2, ret, false, k);
	}

	public static MatrixBlock matrixMultNonNative(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		return matrixMult(m1, m2, ret, false, k);
	}
	
	/**
	 * Performs a matrix multiplication and stores the result in the output matrix.
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * This method allows one to disabling exam sparsity. This feature is useful if matrixMult is used as an intermediate
	 * operation (for example: LibMatrixDNN). It makes sense for LibMatrixDNN because the output is internally
	 * consumed by another dense instruction, which makes repeated conversion to sparse wasteful.
	 * This should be used in rare cases and if you are unsure,
	 * use the method 'matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret)' instead.
	 * 
	 * The parameter k (k&gt;=1) determines the max parallelism k' with k'=min(k, vcores, m1.rlen).
	 * 
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @param fixedRet if true, output representation is fixed and nnzs not recomputed
	 * @param k maximum parallelism
	 * @return ret Matrix Block
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean fixedRet, int k) {
		if(m1.isEmptyBlock(false) || m2.isEmptyBlock(false)) 
			return emptyMatrixMult(m1, m2, ret);
		
		// Timing time = new Timing(true);
		
		// pre analysis
		boolean m1Perm = m1.isSparsePermutationMatrix();
		boolean ultraSparse = (fixedRet && ret.sparse) ||
			(!fixedRet && isUltraSparseMatrixMult(m1, m2, m1Perm));
		boolean sparse = !fixedRet && !ultraSparse && !m1Perm
			&& isSparseOutputMatrixMult(m1, m2);

		// allocate output
		if(ret == null)
			ret = new MatrixBlock(m1.rlen, m2.clen, ultraSparse | sparse);
		else 
			ret.reset(m1.rlen, m2.clen, ultraSparse | sparse);

		if(ret.isInSparseFormat() && ret.getSparseBlock() instanceof SparseBlockMCSR) {
			// we set the estimated number of non zeros per row to number of columns
			// to make the allocation of cells more aggressive.
			((SparseBlockMCSR) ret.getSparseBlock()).setNnzEstimatePerRow(m2.clen, m2.clen);
		}
		
		if(m1.denseBlock instanceof DenseBlockFP64DEDUP){
			DenseBlockFP64DEDUP tmp = (DenseBlockFP64DEDUP) m1.denseBlock;
			if(tmp.getNrEmbsPerRow() != 1){
				//TODO: currently impossible case, since Dedup reshape is not supported yet, once it is, this method needs
				// to be implemented
				throw new NotImplementedException("Check TODO");
			}
			ret.allocateDenseBlock(true, true);
			tmp = (DenseBlockFP64DEDUP) ret.denseBlock;
			tmp.setEmbeddingSize(ret.clen);
		}
		else
			ret.allocateBlock();
		
		if(ret.isInSparseFormat() && !( ret.getSparseBlock() instanceof SparseBlockMCSR)){
			throw new DMLRuntimeException("Matrix Multiplication Sparse output must be MCSR");
		}

		// Detect if we should transpose skinny right side.
		boolean tm2 = !fixedRet && checkPrepMatrixMultRightInput(m1,m2);
		m2 = prepMatrixMultRightInput(m1, m2, tm2);

		// check for multi-threading
		if (!ret.isThreadSafe() 
				|| !satisfiesMultiThreadingConstraints(m1, m2, m1.rlen==1, true, 2, k)
				|| fixedRet) // Fixed ret not supported in multithreaded execution yet
			k = 1;

		if(k <= 1)
			singleThreadedMatrixMult(m1, m2, ret, ultraSparse, sparse, tm2, m1Perm, fixedRet);
		else
			parallelMatrixMult(m1, m2, ret, k, ultraSparse, sparse, tm2, m1Perm);

		//System.out.println("MM "+k+" ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+")x" +
		//		"("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	
		return ret;
	}

	private static void singleThreadedMatrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,  
		boolean ultraSparse, boolean sparse, boolean tm2, boolean m1Perm, boolean fixedRet){
		// prepare row-upper for special cases of vector-matrix
		final boolean pm2 = !ultraSparse && checkParMatrixMultRightInputRows(m1, m2, Integer.MAX_VALUE);
		final int ru2 = (pm2) ? m2.rlen : m1.rlen;

		// core matrix mult computation
		if(ultraSparse && !fixedRet)
			matrixMultUltraSparse(m1, m2, ret, m1Perm, 0, ru2);
		else if(!m1.sparse && !m2.sparse && !ret.sparse)
			matrixMultDenseDense(m1, m2, ret, tm2, pm2, 0, ru2, 0, m2.clen);
		else if(m1.sparse && m2.sparse)
			matrixMultSparseSparse(m1, m2, ret, pm2, sparse, 0, ru2);
		else if(m1.sparse)
			matrixMultSparseDense(m1, m2, ret, pm2, 0, ru2);
		else
			matrixMultDenseSparse(m1, m2, ret, pm2, 0, ru2);

		// post-processing: nnz/representation
		if(!fixedRet) {
			if(!ret.sparse)
				ret.recomputeNonZeros();
			ret.examSparsity();
		}
	}

	private static void parallelMatrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k, 
		boolean ultraSparse, boolean sparse, boolean tm2, boolean m1Perm){
		// prepare row-upper for special cases of vector-matrix / matrix-matrix
		boolean pm2r = !ultraSparse && !sparse && checkParMatrixMultRightInputRows(m1, m2, k);
		boolean pm2c = !ultraSparse && checkParMatrixMultRightInputCols(m1, m2, k, pm2r);
		int num = pm2r ? m2.rlen : pm2c ? m2.clen : m1.rlen;
		
		// core multi-threaded matrix mult computation
		// (currently: always parallelization over number of rows)
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(num, k,
				(pm2r || pm2c || ret.denseBlock instanceof DenseBlockFP64DEDUP));
			ConcurrentHashMap<double[], double[]> cache = m1.denseBlock instanceof DenseBlockFP64DEDUP ? new ConcurrentHashMap<>(): null;
			for(int i = 0, lb = 0; i < blklens.size(); lb += blklens.get(i), i++)
				tasks.add(new MatrixMultTask(m1, m2, ret, tm2, pm2r, pm2c, m1Perm, sparse, lb, lb + blklens.get(i), cache));
			// execute tasks
			
			// aggregate partial results (nnz, ret for vector/matrix)
			// reset nonZero before execution.
			// nonZero count cannot be trusted since it is not atomic
			// and some of the matrix multiplication kernels call quick set value modifying the count.
			ret.nonZeros = 0; 
			long nnzCount = 0;
			for(Future<Object> task : pool.invokeAll(tasks)) {
				if(pm2r) // guaranteed single block
					vectAdd((double[]) task.get(), ret.getDenseBlockValues(), 0, 0, ret.rlen * ret.clen);
				else // or count non zeros of the block
					nnzCount += (Long) task.get();
			}
			if(pm2r)
				ret.recomputeNonZeros(k);
			else // set the non zeros to the counted values.
				ret.nonZeros = nnzCount;

			// post-processing (nnz maintained in parallel)
			ret.examSparsity(k);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}

	}

	public static MatrixBlock emptyMatrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret){
		final int rl = m1.rlen;
		final int cl = m2.clen;

		if(ret == null)
			return new MatrixBlock(rl, cl, true);
		else {
			ret.reset(rl, cl, true);
			ret.setNonZeros(0);
			ret.cleanupBlock(true, true);
			return ret;
		}
	}

	/**
	 * Performs a matrix multiplication chain operation of type t(X)%*%(X%*%v) or t(X)%*%(w*(X%*%v)).
	 * 
	 * All variants use a IKJ access pattern, and internally use dense output. After the
	 * actual computation, we recompute nnz and check for sparse/dense representation.
	 * 
	 * @param mX X matrix
	 * @param mV v matrix
	 * @param mW w matrix
	 * @param ret result matrix
	 * @param ct chain type
	 */
	public static void matrixMultChain(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct) {
		//check inputs / outputs (after that mV and mW guaranteed to be dense)
		if( mX.isEmptyBlock(false) || (mV.isEmptyBlock(false) && ct!=ChainType.XtXvy)
			|| (mW !=null && mW.isEmptyBlock(false)) ) {
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
	 * The parameter k (k&gt;=1) determines the max parallelism k' with k'=min(k, vcores, m1.rlen).
	 * 
	 * NOTE: This multi-threaded mmchain operation has additional memory requirements of k*ncol(X)*8bytes 
	 * for partial aggregation. Current max memory: 256KB; otherwise redirect to sequential execution.
	 * 
	 * @param mX X matrix
	 * @param mV v matrix
	 * @param mW w matrix
	 * @param ret result matrix
	 * @param ct chain type
	 * @param k maximum parallelism
	 */
	public static void matrixMultChain(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int k) {
		//check inputs / outputs (after that mV and mW guaranteed to be dense)
		if( mX.isEmptyBlock(false) || (mV.isEmptyBlock(false) && ct!=ChainType.XtXvy)
			|| (mW !=null && mW.isEmptyBlock(false)) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check temporary memory and too small workload for multi-threading
		if( !satisfiesMultiThreadingConstraints(mX, true, true, mX.sparse?2:4, k) ) { 
			matrixMultChain(mX, mV, mW, ret, ct);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing (no need to check isThreadSafe)
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		//core matrix mult chain computation
		//(currently: always parallelization over number of rows)
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(mX.rlen, k, true);
			ArrayList<MatrixMultChainTask> tasks = new ArrayList<>();
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
				tasks.add(new MatrixMultChainTask(mX, mV, mW, ct, lb, lb+blklens.get(i)));
			List<Future<double[]>> taskret = pool.invokeAll(tasks);

			//aggregate partial results and error handling
			double[][] a = new double[taskret.size()][];
			for(int i=0; i<taskret.size(); i++)
				a[i] = taskret.get(i).get();
			vectAddAll(a, ret.getDenseBlockValues(), 0, 0, mX.clen);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//post-processing
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("MMChain "+ct.toString()+" k="+k+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//		              "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	public static MatrixBlock matrixMultTransposeSelf( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose ) {
		matrixMultTransposeSelf(m1, ret, leftTranspose, true);
		return ret;
	}

	public static void matrixMultTransposeSelf(MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, boolean copyToLowerTriangle){
		//check inputs / outputs
		if( m1.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing
		ret.sparse = isSparseOutputTSMM(m1);
		ret.allocateBlock();
		MatrixBlock m1t = isSparseOutputTSMM(m1, true) ?
			LibMatrixReorg.transpose(m1) : null;
		
		//core tsmm operation
		matrixMultTransposeSelf(m1, m1t, ret, leftTranspose, 0, ret.rlen);

		//post-processing
		if(copyToLowerTriangle) {
			long nnz = copyUpperToLowerTriangle(ret);
			ret.setNonZeros(nnz);
			ret.examSparsity();
		}
		else {
			ret.recomputeNonZeros();
		}
		
		//System.out.println("TSMM ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());
	}

	/**
	 * TSMM with optional transposed left side or not (Transposed self matrix multiplication)
	 * 
	 * @param m1            The matrix to do tsmm
	 * @param ret           The output matrix to allocate the result to
	 * @param leftTranspose If the left side should be considered transposed
	 * @param k             the number of threads to use
	 */
	public static void matrixMultTransposeSelf(MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int k) {
		//check inputs / outputs
		if( m1.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check too small workload and fallback to sequential if necessary
		if( !satisfiesMultiThreadingConstraintsTSMM(m1, leftTranspose, 1, k) ) {
			matrixMultTransposeSelf(m1, ret, leftTranspose);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//pre-processing (no need to check isThreadSafe)
		ret.sparse = isSparseOutputTSMM(m1);
		ret.allocateBlock();
		MatrixBlock m1t = isSparseOutputTSMM(m1, true) ?
			LibMatrixReorg.transpose(m1, k) : null;
		
		//core multi-threaded matrix mult computation
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultTransposeTask> tasks = new ArrayList<>();
			//load balance via #tasks=4k due to triangular shape 
			int blklen = (int)(Math.ceil((double)ret.rlen / (4 * k)));
			for(int i = 0; i < ret.rlen; i += blklen)
				tasks.add(new MatrixMultTransposeTask(m1, m1t, ret, leftTranspose, i, Math.min(i+blklen, ret.rlen)));
			for( Future<Object> rtask :  pool.invokeAll(tasks) )
				rtask.get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//post-processing
		long nnz = copyUpperToLowerTriangle(ret);
		ret.setNonZeros(nnz);
		ret.examSparsity();
		
		//System.out.println("TSMM k="+k+" ("+m1.isInSparseFormat()+","+m1.getNumRows()+","+m1.getNumColumns()+","+m1.getNonZeros()+","+leftTranspose+") in "+time.stop());
	}

	public static void matrixMultPermute( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2 ) {
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

	public static void matrixMultPermute( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int k) {
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
		ret1.sparse = false;	  // no need to check isThreadSafe
		ret1.allocateDenseBlock();
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultPermuteTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)pm1.rlen/k));
			for( int i=0; i<k & i*blklen<pm1.rlen; i++ )
				tasks.add(new MatrixMultPermuteTask(pm1, m2, ret1, ret2, i*blklen, Math.min((i+1)*blklen, pm1.rlen)));
			for(Future<Object> f : pool.invokeAll(tasks))
				f.get();
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
		}
		
		//post-processing
		ret1.recomputeNonZeros(k);
		ret1.examSparsity();
		if( ret2 != null ) { //optional second output
			ret2.recomputeNonZeros(k);
			ret2.examSparsity();
		}
		
		// System.out.println("PMM Par ("+pm1.isInSparseFormat()+","+pm1.getNumRows()+","+pm1.getNumColumns()+","+pm1.getNonZeros()+")x" +
		//                   "("+m2.isInSparseFormat()+","+m2.getNumRows()+","+m2.getNumColumns()+","+m2.getNonZeros()+") in "+time.stop());
	}	

	public static void matrixMultWSLoss(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt) {
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
		
		//add correction for sparse wsloss w/o weight
		if( mX.sparse && wt==WeightsType.NONE )
			addMatrixMultWSLossNoWeightCorrection(mU, mV, ret, 1);
		
		//System.out.println("MMWSLoss " +wt.toString()+ " ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                  "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	public static void matrixMultWSLoss(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int k) {
		//check for empty result
		if( wt==WeightsType.POST && mW.isEmptyBlock(false)
			|| wt==WeightsType.POST_NZ && mX.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return;
		}
		
		//check no parallelization benefit (fallback to sequential)
		//no need to check isThreadSafe (scalar output)
		if( mX.rlen == 1 ) {
			matrixMultWSLoss(mX, mU, mV, mW, ret, wt);
			return;
		}
		
		//Timing time = new Timing(true);
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultWSLossTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)mX.rlen/k));
			for( int i=0; i<k & i*blklen<mX.rlen; i++ )
				tasks.add(new MatrixMultWSLossTask(mX, mU, mV, mW, wt, i*blklen, Math.min((i+1)*blklen, mX.rlen)));
	
			sumScalarResults(pool.invokeAll(tasks), ret);
		} 
		catch( Exception e ) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
		}

		//add correction for sparse wsloss w/o weight
		if( mX.sparse && wt==WeightsType.NONE )
			addMatrixMultWSLossNoWeightCorrection(mU, mV, ret, k);
		
		//System.out.println("MMWSLoss "+wt.toString()+" k="+k+" ("+mX.isInSparseFormat()+","+mX.getNumRows()+","+mX.getNumColumns()+","+mX.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	public static void matrixMultWSigmoid(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt) {
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateBlock();
		
		//core weighted square sum mm computation
		boolean allDense = !mW.sparse && !mU.sparse && !mV.sparse
			&& !mU.isEmptyBlock() && !mV.isEmptyBlock();
		if( NativeHelper.isNativeLibraryLoaded() && allDense && (mW.rlen == 1 || mW.clen == 1) 
			&& !LibMatrixNative.isMatMultMemoryBound(mU.rlen, mU.clen, mV.rlen)
			&& mW.getDenseBlock().isContiguous() && mU.getDenseBlock().isContiguous() && mV.getDenseBlock().isContiguous() )
			matrixMultWSigmoidDenseNative(mW, mU, mV, ret, wt);
		else if( allDense )
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

	public static void matrixMultWSigmoid(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int k) {
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//check no parallelization benefit (fallback to sequential)
		if (mW.rlen == 1 || !MatrixBlock.isThreadSafe(mW.sparse)) {
			matrixMultWSigmoid(mW, mU, mV, ret, wt);
			return;
		}
		
		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateBlock();
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultWSigmoidTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWSigmoidTask(mW, mU, mV, ret, wt, i*blklen, Math.min((i+1)*blklen, mW.rlen)));

			//aggregate partial nnz and check for errors
			ret.nonZeros = 0; //reset after execute
			for( Future<Long> task : pool.invokeAll(tasks) )
				ret.nonZeros += task.get();
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
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
	 * @param mW matrix W
	 * @param mU matrix U
	 * @param mV matrix V
	 * @param mX matrix X
	 * @param ret result type
	 * @param wt weighted divide matrix multiplication type
	 */
	public static void matrixMultWDivMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt) {
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
		ret.allocateBlock();
		
		//core weighted div mm computation
		boolean scalarX = wt.hasScalar();
		if( !mW.sparse && !mU.sparse && !mV.sparse && (mX==null || !mX.sparse || scalarX) && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWDivMMDense(mW, mU, mV, mX, ret, wt, 0, mW.rlen, 0, mW.clen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && (mX==null || mX.sparse || scalarX) && !mU.isEmptyBlock() && !mV.isEmptyBlock())
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
	 * @param mW matrix W
	 * @param mU matrix U
	 * @param mV matrix V
	 * @param mX matrix X
	 * @param ret result matrix
	 * @param wt weighted divide matrix multiplication type
	 * @param k maximum parallelism
	 */
	public static void matrixMultWDivMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int k) {
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
		ret.allocateBlock();

		if (!ret.isThreadSafe()){
			matrixMultWDivMM(mW, mU, mV, mX, ret, wt);
			return;
		}
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultWDivTask> tasks = new ArrayList<>();
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

			//aggregate partial nnz and check for errors
			ret.nonZeros = 0;  //reset after execute
			for( Future<Long> task : pool.invokeAll(tasks) )
				ret.nonZeros += task.get();
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		} 
		finally{
			pool.shutdown();
		}

		//post-processing
		ret.examSparsity();
		
		//System.out.println("MMWDiv "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}	

	public static void matrixMultWCeMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, MatrixBlock ret, WCeMMType wt) {
		//check for empty result 
		if( mW.isEmptyBlock(false) )  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		//core weighted cross entropy mm computation
		if( !mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock() )
			matrixMultWCeMMDense(mW, mU, mV, eps, ret, wt, 0, mW.rlen);
		else if( mW.sparse && !mU.sparse && !mV.sparse && !mU.isEmptyBlock() && !mV.isEmptyBlock())
			matrixMultWCeMMSparseDense(mW, mU, mV, eps, ret, wt, 0, mW.rlen);
		else
			matrixMultWCeMMGeneric(mW, mU, mV, eps, ret, wt, 0, mW.rlen);
		
		//System.out.println("MMWCe "+wt.toString()+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	public static void matrixMultWCeMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, MatrixBlock ret, WCeMMType wt, int k) {
		//check for empty result 
		if( mW.isEmptyBlock(false) )  {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing (no need to check isThreadSafe)
		ret.sparse = false;
		ret.allocateDenseBlock();
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultWCeTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWCeTask(mW, mU, mV, eps, wt, i*blklen, Math.min((i+1)*blklen, mW.rlen)));
			List<Future<Double>> taskret = pool.invokeAll(tasks);
			//aggregate partial results
			sumScalarResults(taskret, ret);
		} 
		catch( Exception e ) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
		}
		
		//System.out.println("MMWCe "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                 "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop());
	}

	public static void matrixMultWuMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn) {
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}

		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateBlock();
		
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

	public static void matrixMultWuMM(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int k) {
		//check for empty result
		if( mW.isEmptyBlock(false) ) {
			ret.examSparsity(); //turn empty dense into sparse
			return; 
		}
		
		//check no parallelization benefit (fallback to sequential)
		if (mW.rlen == 1 || !MatrixBlock.isThreadSafe(mW.sparse)) {
			matrixMultWuMM(mW, mU, mV, ret, wt, fn);
			return;
		}
		
		//Timing time = new Timing(true);

		//pre-processing
		ret.sparse = mW.sparse;
		ret.allocateBlock();
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<MatrixMultWuTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)mW.rlen/k));
			for( int i=0; i<k & i*blklen<mW.rlen; i++ )
				tasks.add(new MatrixMultWuTask(mW, mU, mV, ret, wt, fn, i*blklen, Math.min((i+1)*blklen, mW.rlen)));

			//aggregate partial nnz and check for errors
			ret.nonZeros = 0; //reset after execute
			for( Future<Long> task : pool.invokeAll(tasks) )
				ret.nonZeros += task.get();
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally{
			pool.shutdown();
		}

		//post-processing (nnz maintained in parallel)
		ret.examSparsity();

		//System.out.println("MMWu "+wt.toString()+" k="+k+" ("+mW.isInSparseFormat()+","+mW.getNumRows()+","+mW.getNumColumns()+","+mW.getNonZeros()+")x" +
		//                   "("+mV.isInSparseFormat()+","+mV.getNumRows()+","+mV.getNumColumns()+","+mV.getNonZeros()+") in "+time.stop() + ".");
	}
	
	//////////////////////////////////////////
	// optimized matrix mult implementation //
	//////////////////////////////////////////

	private static void matrixMultDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean tm2, boolean pm2, int rl, int ru, int cl, int cu) {
		DenseBlock a = m1.getDenseBlock();
		DenseBlock b = m2.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m1.clen;
		
		if( m==1 && n==1 ) {            //DOT PRODUCT
			double[] avals = a.valuesAt(0);
			double[] bvals = b.valuesAt(0);
			if( ru > m ) //pm2r - parallelize over common dim
				c.set(0, 0, dotProduct(avals, bvals, rl, rl, ru-rl));
			else
				c.set(0, 0, dotProduct(avals, bvals, cd));
		}
		else if( n>1 && cd == 1 ) {     //OUTER PRODUCT
			double[] avals = a.valuesAt(0);
			double[] bvals = b.valuesAt(0);
			for( int i=rl; i < ru; i++) {
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				if( avals[i] == 1 )
					System.arraycopy(bvals, 0, cvals, cix, n);
				else if( avals[i] != 0 )
					vectMultiplyWrite(avals[i], bvals, cvals, 0, cix, n);
				else
					Arrays.fill(cvals, cix, cix+n, 0);
			}
		}
		else if( n==1 && cd == 1 ) {    //VECTOR-SCALAR
			double[] avals = a.valuesAt(0);
			double[] cvals = c.valuesAt(0);
			vectMultiplyWrite(b.get(0,0), avals, cvals, rl, rl, ru-rl);
		}
		else if( n==1 && cd<=2*1024 ) { //MATRIX-VECTOR (short rhs)
			matrixMultDenseDenseMVShortRHS(a, b, c, cd, rl, ru);
		}
		else if( n==1 ) {               //MATRIX-VECTOR (tall rhs)
			matrixMultDenseDenseMVTallRHS(a, b, c, pm2, cd, rl, ru);
		}
		else if( pm2 && m==1 ) {        //VECTOR-MATRIX
			matrixMultDenseDenseVM(a, b, c, n, cd, rl, ru);
		}
		else if( pm2 && m<=16 ) {       //MATRIX-MATRIX (short lhs) 
			matrixMultDenseDenseMMShortLHS(a, b, c, m, n, cd, rl, ru);
		}
		else if( tm2 ) {                //MATRIX-MATRIX (skinny rhs)
			matrixMultDenseDenseMMSkinnyRHS(a, b, c, m2.rlen, cd, rl, ru);
		}
		else {                          //MATRIX-MATRIX
			matrixMultDenseDenseMM(a, b, c, n, cd, rl, ru, cl, cu);
		}
	}
	
	private static void matrixMultDenseDenseMVShortRHS(DenseBlock a, DenseBlock b, DenseBlock c, int cd, int rl, int ru) {
		double[] bvals = b.valuesAt(0);
		double[] cvals = c.valuesAt(0);
		for( int i=rl; i < ru; i++ )
			cvals[i] = dotProduct(a.values(i), bvals, a.pos(i), 0, cd);
	}
	
	private static void matrixMultDenseDenseMVTallRHS(DenseBlock a, DenseBlock b, DenseBlock c, boolean pm2, int cd, int rl, int ru) {
		final int blocksizeI = 32;
		final int blocksizeK = 2*1024; //16KB vector blocks (L1)
		double[] bvals = b.valuesAt(0);
		double[] cvals = c.valuesAt(0);
		// setup bounds according to parallelization strategy
		// (default: rows in lhs, pm2: rows in rhs)
		int cl = pm2 ? rl : 0, cu = pm2 ? ru : cd;
		int rl2 = pm2 ? 0 : rl, ru2 = pm2 ? a.numRows() : ru;
		// matrix-vector multication with cache blocking of vector
		for( int bi=rl2; bi<ru2; bi+=blocksizeI ) {
			int bimin = Math.min(bi+blocksizeI, ru2);
			for( int bk=cl; bk<cu; bk+=blocksizeK ) {
				int bkmin = Math.min(bk+blocksizeK, cu);
				for( int i=bi; i<bimin; i++) 
					cvals[i] += dotProduct(a.values(i), bvals, a.pos(i,bk), bk, bkmin-bk);
			}
		}
	}
	
	private static void matrixMultDenseDenseVM(DenseBlock a, DenseBlock b, DenseBlock c, int n, int cd, int rl, int ru) {
		double[] avals = a.valuesAt(0); //vector
		double[] cvals = c.valuesAt(0); //vector
		
		//parallelization over rows in rhs matrix
		//rest not aligned to blocks of 2 rows
		final int kn = b.isContiguous() ? rl+(ru-rl)%2 : ru;
		for( int k = rl; k < kn; k++ )
			if( avals[k] != 0 )
				vectMultiplyAdd(avals[k], b.values(k), cvals, b.pos(k), 0, n);
		
		//compute blocks of 2 rows (2 instead of 4 for small n<64)
		double[] bvals = b.valuesAt(0); //only for special case
		for( int k=kn, bix=kn*n; k<ru; k+=2, bix+=2*n ){
			if( avals[k] != 0 && avals[k+1] != 0  )
				vectMultiplyAdd2(avals[k], avals[k+1], bvals, cvals, bix, bix+n, 0, n);
			else if( avals[k] != 0 )
				vectMultiplyAdd(avals[k], bvals, cvals, bix, 0, n);
			else if( avals[k+1] != 0 )
				vectMultiplyAdd(avals[k+1], bvals, cvals, bix+n, 0, n);
		}
	}
	
	private static void matrixMultDenseDenseMMShortLHS(DenseBlock a, DenseBlock b, DenseBlock c, int m, int n, int cd, int rl, int ru) {
		//cache-conscious parallelization over rows in rhs matrix
		final int kn = (ru-rl)%4;
		
		//rest not aligned to blocks of 2 rows
		for( int i=0; i<m; i++ ) {
			double[] avals = a.values(i), cvals = c.values(i);
			int aix = a.pos(i), cix = c.pos(i);
			for( int k=rl; k<rl+kn; k++ )
				if( avals[aix+k] != 0 )
					vectMultiplyAdd(avals[aix+k], b.values(k), cvals, b.pos(k), cix, n);
		}
		
		final int blocksizeK = 48;
		final int blocksizeJ = 1024;
		
		//blocked execution
		for( int bk = rl+kn; bk < ru; bk+=blocksizeK ) {
			int bkmin = Math.min(ru, bk+blocksizeK);
			for( int bj = 0; bj < n; bj+=blocksizeJ ) {
				//compute blocks of 4 rows in rhs w/ IKJ
				int bjlen = Math.min(n, bj+blocksizeJ)-bj;
				for( int i=0; i<m; i++ ) {
					double[] avals = a.values(i), cvals = c.values(i);
					int aix = a.pos(i), cix = c.pos(i, bj);
					if( b.isContiguous(bk, bkmin-1) ) {
						double[] bvals = b.values(bk);
						for( int k=bk, bix=b.pos(bk, bj); k<bkmin; k+=4, bix+=4*n )
							vectMultiplyAdd4(avals[aix+k], avals[aix+k+1], avals[aix+k+2], avals[aix+k+3],
								bvals, cvals, bix, bix+n, bix+2*n, bix+3*n, cix, bjlen);
					}
					else {
						for( int k=rl; k<rl+kn; k++ )
							if( avals[aix+k] != 0 )
								vectMultiplyAdd(avals[aix+k], b.values(k), cvals, b.pos(k), cix, n);
					}
				}
			}
		}
	}
	
	private static void matrixMultDenseDenseMMSkinnyRHS(DenseBlock a, DenseBlock b, DenseBlock c, int n2, int cd, int rl, int ru) {
		//note: prepared rhs input via transpose for: m > n && cd > 64 && n < 64
		//however, explicit flag required since dimension change m2
		for( int i=rl; i < ru; i++ ) {
			double[] avals = a.values(i), cvals = c.values(i);
			int aix = a.pos(i), cix = c.pos(i);
			for( int j=0; j<n2; j++ )
				cvals[cix+j] = dotProduct(avals, b.values(j), aix, b.pos(j), cd);
		}
	}

	public static void matrixMultDenseDenseMMDedup(DenseBlockFP64DEDUP a, DenseBlock b, DenseBlockFP64DEDUP c, int n, int cd, int rl, int ru, ConcurrentHashMap<double[], double[]> cache) {
		//n = m2.clen;
		//cd = m1.clen;
		if(a.getNrEmbsPerRow() != 1){
			//TODO: currently impossible case, since Dedup reshape is not supported yet, once it is, this method needs
			// to be implemented
			throw new NotImplementedException("Check TODO");
		}
		for (int i = rl; i < ru; i++) {
			double[] a_row = a.getDedupDirectly(i);
			double[] c_row = cache.getOrDefault(a_row, null);
			if (c_row == null) {
				c_row = new double[n];
				for (int j = 0; j < n; j++) {
					c_row[j] = 0.0;
					//the following requires b.isContiguous(0,cd)
					double[] b_column = b.values(0);
					for (int k = 0; k < cd; k++) {
						c_row[j] += a_row[k] * b_column[b.pos(k, j)];
					}
				}
				cache.put(a_row, c_row);
			}
			c.setDedupDirectly(i, c_row);
		}
	}

	//note: public for use by codegen for consistency
	public static void matrixMultDenseDenseMM(DenseBlock a, DenseBlock b, DenseBlock c, int n, int cd, int rl, int ru, int cl, int cu) {
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
				for( int bj = cl, bkmin = Math.min(cd, bk+blocksizeK); bj < cu; bj+=blocksizeJ ) 
				{
					int bklen = bkmin-bk;
					int bjlen = Math.min(cu, bj+blocksizeJ)-bj;
					
					//core sub block matrix multiplication
					for( int i = bi; i < bimin; i++) {
						double[] avals = a.values(i), cvals = c.values(i);
						int aixi = a.pos(i, bk), cixj = c.pos(i, bj);
						
						if( b.isContiguous(bk, bkmin-1) ) {
							double[] bvals = b.values(bk);
							int bkpos = b.pos(bk, bj);
							
							//determine nnz of a (for sparsity-aware skipping of rows)
							int knnz = copyNonZeroElements(avals, aixi, bkpos, n, ta, tbi, bklen);
							
							//rest not aligned to blocks of 4 rows
							final int bn = knnz % 4;
							switch( bn ){
								case 1: vectMultiplyAdd(ta[0], bvals, cvals, tbi[0], cixj, bjlen); break;
								case 2: vectMultiplyAdd2(ta[0],ta[1], bvals, cvals, tbi[0], tbi[1], cixj, bjlen); break;
								case 3: vectMultiplyAdd3(ta[0],ta[1],ta[2], bvals, cvals, tbi[0], tbi[1],tbi[2], cixj, bjlen); break;
							}
							
							//compute blocks of 4 rows (core inner loop)
							for( int k = bn; k<knnz; k+=4 ){
								vectMultiplyAdd4( ta[k], ta[k+1], ta[k+2], ta[k+3], bvals, cvals, 
									tbi[k], tbi[k+1], tbi[k+2], tbi[k+3], cixj, bjlen );
							}
						}
						else {
							for( int k = bk; k<bkmin; k++ ) {
								if( avals[k] != 0 )
									vectMultiplyAdd( avals[k], b.values(k),
										cvals, b.pos(k, bj), cixj, bjlen );
							}
						}
					}
				}
	}

	private static void matrixMultDenseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl, int ru) {
		if(ret.isInSparseFormat()){
			if(!m1.sparse && !m2.sparse)
				matrixMultDenseDenseOutSparse(m1,m2,ret, pm2, rl, ru);
			else 
				matrixMultDenseSparseOutSparse(m1, m2, ret, pm2, rl, ru);
		}
		else
			matrixMultDenseSparseOutDense(m1, m2, ret, pm2, rl, ru);
	}


	private static void matrixMultDenseDenseOutSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2,
		int rl, int ru) {
		final DenseBlock a = m1.getDenseBlock();
		final DenseBlock b = m2.getDenseBlock();
		final SparseBlock c = ret.getSparseBlock();
		final int m = m1.rlen;  // rows left
		final int cd = m1.clen; // common dim
		final int n = m2.clen;

		final int rl1 = pm2 ? 0 : rl;
		final int ru1 = pm2 ? m : ru;
		final int rl2 = pm2 ? rl : 0;
		final int ru2 = pm2 ? ru : cd;

		final int blocksizeK = 32;
		final int blocksizeI = 32;

		for(int bi = rl1; bi < ru1; bi += blocksizeI) {
			for(int bk = rl2, bimin = Math.min(ru1, bi + blocksizeI); bk < ru2; bk += blocksizeK) {
				final int bkmin = Math.min(ru2, bk + blocksizeK);
				// core sub block matrix multiplication
				for(int i = bi; i < bimin; i++) { // rows left
					final double[] avals = a.values(i);
					final int aix = a.pos(i);
					for(int k = bk; k < bkmin; k++) { // common dimension
						final double aval = avals[aix + k];
						if(aval != 0) {
							final double[] bvals = b.values(k);
							final int bpos = b.pos(k);
							for(int j = 0; j < n; j++) {
								final double bv = bvals[bpos + j];
								c.add(i, j, aval * bv);
							}
						}
					}
				}
			}
		}
	}


	private static void matrixMultDenseSparseOutSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2,
		int rl, int ru) {
		final DenseBlock a = m1.getDenseBlock();
		final SparseBlock b = m2.getSparseBlock();
		final SparseBlock c = ret.getSparseBlock();
		final int m = m1.rlen;  // rows left
		final int cd = m1.clen; // common dim

		final int rl1 = pm2 ? 0 : rl;
		final int ru1 = pm2 ? m : ru;
		final int rl2 = pm2 ? rl : 0;
		final int ru2 = pm2 ? ru : cd;

		final int blocksizeK = 32;
		final int blocksizeI = 32;

		for(int bi = rl1; bi < ru1; bi += blocksizeI) {
			for(int bk = rl2, bimin = Math.min(ru1, bi + blocksizeI); bk < ru2; bk += blocksizeK) {
				final int bkmin = Math.min(ru2, bk + blocksizeK);
				// core sub block matrix multiplication
				for(int i = bi; i < bimin; i++) { // rows left
					final double[] avals = a.values(i);
					final int aix = a.pos(i);
					for(int k = bk; k < bkmin; k++) { // common dimension
						final double aval = avals[aix + k];
						if(aval == 0 || b.isEmpty(k))
							continue;
						final int[] bIdx = b.indexes(k);
						final double[] bVals = b.values(k);
						final int bPos = b.pos(k);
						final int bEnd = bPos + b.size(k);
						for(int j = bPos; j < bEnd ; j++){
							c.add(i, bIdx[j], aval * bVals[j]);
						}
					}
				}
			}
		}
	}

	private static void matrixMultDenseSparseOutDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl,
		int ru) {
		DenseBlock a = m1.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		int m = m1.rlen;
		int cd = m1.clen;
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		SparseBlock b = m2.sparseBlock;
			
		if( pm2 && m==1 ) {        //VECTOR-MATRIX
			//parallelization over rows in rhs matrix
			double[] avals = a.valuesAt(0); //vector
			double[] cvals = c.valuesAt(0); //vector
			for( int k=rl; k<ru; k++ )
				if( avals[k] != 0 && !b.isEmpty(k) ) {
					vectMultiplyAdd(avals[k], b.values(k), cvals,
						b.indexes(k), b.pos(k), 0, b.size(k));
				}
		}
		else {                     //MATRIX-MATRIX
			//best effort blocking, without blocking over J because it is 
			//counter-productive, even with front of current indexes
			final int blocksizeK = 32;
			final int blocksizeI = 32;
			
			int rl1 = pm2 ? 0 : rl;
			int ru1 = pm2 ? m : ru;
			int rl2 = pm2 ? rl : 0;
			int ru2 = pm2 ? ru : cd;
			
			//blocked execution
			for( int bi = rl1; bi < ru1; bi+=blocksizeI )
				for( int bk = rl2, bimin = Math.min(ru1, bi+blocksizeI); bk < ru2; bk+=blocksizeK ) {
					int bkmin = Math.min(ru2, bk+blocksizeK);
					//core sub block matrix multiplication
					for(int i = bi; i < bimin; i++) {
						double[] avals = a.values(i), cvals = c.values(i);
						int aix = a.pos(i), cix = c.pos(i);
						for( int k = bk; k < bkmin; k++ ) {
							double aval = avals[aix+k];
							if( aval == 0 || b.isEmpty(k) )
								continue;
							vectMultiplyAdd(aval, b.values(k), cvals, 
								b.indexes(k), b.pos(k), cix, b.size(k));
						}
					}
				}
		}
	}

	private static void matrixMultSparseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, int rl, int ru) {
		SparseBlock a = m1.sparseBlock;
		DenseBlock b = m2.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		final int m = m1.rlen;
		final int n = m2.clen;
		final int cd = m2.rlen;
		final long xsp = (long)m*cd/m1.nonZeros;

		if( m==1 && n==1 ) {            //DOT PRODUCT
			if( !a.isEmpty(0) )
				c.set(0, 0, dotProduct(a.values(0), b.values(0), a.indexes(0), a.pos(0), 0, a.size(0)));
		}
		else if( n==1 && cd<=2*1024 ) { //MATRIX-VECTOR (short rhs)
			matrixMultSparseDenseMVShortRHS(a, b, c, cd, rl, ru);
		}
		else if( n==1 ) {               //MATRIX-VECTOR (tall rhs)
			matrixMultSparseDenseMVTallRHS(a, b, c, cd, xsp, rl, ru);
		}
		else if( pm2 && m==1 ) {        //VECTOR-MATRIX
			matrixMultSparseDenseVM(a, b, c, n, rl, ru);
		}
		else if( pm2 && m<=16 ) {       //MATRIX-MATRIX (short lhs) 
			matrixMultSparseDenseMMShortLHS(a, b, c, n, cd, rl, ru);
		}
		else if( n<=64 ) {              //MATRIX-MATRIX (skinny rhs)
			matrixMultSparseDenseMMSkinnyRHS(a, b, c, n, rl, ru);
		}
		else {                          //MATRIX-MATRIX
			matrixMultSparseDenseMM(a, b, c, n, cd, xsp, rl, ru);
		}
	}
	
	private static void matrixMultSparseDenseMVShortRHS(SparseBlock a, DenseBlock b, DenseBlock c, int cd, int rl, int ru) {
		double[] bvals = b.valuesAt(0);
		double[] cvals = c.valuesAt(0);
		for( int i=rl; i<ru; i++ ) {
			if( a.isEmpty(i) ) continue;
			int alen = a.size(i);
			int apos = a.pos(i);
			double[] avals = a.values(i);
			cvals[i] = (alen==cd) ? dotProduct(avals, bvals, apos, 0, cd) :
				dotProduct(avals, bvals, a.indexes(i), apos, 0, alen);
		}
	}
	
	private static void matrixMultSparseDenseMVTallRHS(SparseBlock a, DenseBlock b, DenseBlock c, int cd, long xsp, int rl, int ru) {
		final int blocksizeI = 512; //8KB curk+cvals in L1
		final int blocksizeK = (int)Math.max(2048,2048*xsp/32); //~256KB bvals in L2
		
		//short-cut to kernel w/o cache blocking if no benefit
		if( blocksizeK >= cd ) {
			matrixMultSparseDenseMVShortRHS(a, b, c, cd, rl, ru);
			return;
		}
		
		//sparse matrix-vector w/ cache blocking (keep front of positions)
		double[] bvals = b.valuesAt(0);
		double[] cvals = c.valuesAt(0);
		int[] curk = new int[blocksizeI];
		
		for( int bi = rl; bi < ru; bi+=blocksizeI ) {
			Arrays.fill(curk, 0); //reset positions
			for( int bk=0, bimin = Math.min(ru, bi+blocksizeI); bk<cd; bk+=blocksizeK ) {
				for( int i=bi, bkmin = bk+blocksizeK; i<bimin; i++) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					int k = curk[i-bi] + apos;
					for( ; k<apos+alen && aix[k]<bkmin; k++ )
						cvals[i] += avals[k] * bvals[aix[k]];
					curk[i-bi] = k - apos;
				}
			}
		}
	}
	
	private static void matrixMultSparseDenseVM(SparseBlock a, DenseBlock b, DenseBlock c, int n, int rl, int ru) {
		if( a.isEmpty(0) )
			return;
		
		//parallelization over rows in rhs matrix
		int alen = a.size(0);
		int[] aix = a.indexes(0);
		double[] avals = a.values(0);
		double[] cvals = c.valuesAt(0);
		int rlix = (rl==0) ? 0 : a.posFIndexGTE(0,rl);
		rlix = (rlix>=0) ? rlix : alen;
		
		if( b.isContiguous() ) {
			double[] bvals = b.valuesAt(0);
			for( int k=rlix; k<alen && aix[k]<ru; k++ )
				if( k+1<alen && aix[k+1]<ru )
					vectMultiplyAdd2(avals[k], avals[k+1], bvals, cvals, aix[k]*n, aix[++k]*n, 0, n);
				else
					vectMultiplyAdd(avals[k], bvals, cvals, aix[k]*n, 0, n);
		}
		else {
			for( int k=rlix; k<alen && aix[k]<ru; k++ )
				vectMultiplyAdd(avals[k], b.values(aix[k]), cvals, b.pos(aix[k]), 0, n);
		}
	}
	
	private static void matrixMultSparseDenseMMShortLHS(SparseBlock a, DenseBlock b, DenseBlock c, int n, int cd, int rl, int ru) {
		int arlen = a.numRows();
		for( int i=0; i<arlen; i++ ) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			double[] cvals = c.values(i);
			int cix = c.pos(i);
			
			int k1 = (rl==0) ? 0 : a.posFIndexGTE(i, rl);
			k1 = (k1>=0) ? apos+k1 : apos+alen;
			int k2 = (ru==cd) ? alen : a.posFIndexGTE(i, ru);
			k2 = (k2>=0) ? apos+k2 : apos+alen;
			
			//note: guard k1 (and thus also k2) against overrun nnz, and guard
			//contiguous check for k2-1 against underrun of start pos for k1==k2.
			if( k1<apos+alen && (k1==k2 || b.isContiguous(aix[k1], aix[k2-1])) ) {
				double[] bvals = b.values(aix[k1]);
				int base = aix[k1]*n - b.pos(aix[k1]);
				//rest not aligned to blocks of 4 rows
				final int bn = (k2-k1) % 4;
				switch( bn ){
					case 1: vectMultiplyAdd(avals[k1], bvals, cvals, aix[k1]*n-base, cix, n); break;
					case 2: vectMultiplyAdd2(avals[k1],avals[k1+1], bvals, cvals, aix[k1]*n-base, aix[k1+1]*n-base, cix, n); break;
					case 3: vectMultiplyAdd3(avals[k1],avals[k1+1],avals[k1+2], bvals, cvals, aix[k1]*n-base, aix[k1+1]*n-base, aix[k1+2]*n-base, cix, n); break;
				}
				
				//compute blocks of 4 rows (core inner loop)
				for( int k = k1+bn; k<k2; k+=4 ) {
					vectMultiplyAdd4( avals[k], avals[k+1], avals[k+2], avals[k+3], bvals, cvals, 
						aix[k]*n-base, aix[k+1]*n-base, aix[k+2]*n-base, aix[k+3]*n-base, cix, n );
				}
			}
			else {
				for( int k = k1; k<k2; k++ )
					vectMultiplyAdd( avals[k], b.values(aix[k]), cvals, b.pos(aix[k]), cix, n );
			}
		}
	}
	
	private static void matrixMultSparseDenseMMSkinnyRHS(SparseBlock a, DenseBlock b, DenseBlock c, int n, int rl, int ru) {
		//no blocking since b and c fit into cache anyway
		for( int i=rl, cix=rl*n; i<ru; i++, cix+=n ) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			double[] cvals = c.values(i);
			//rest not aligned to blocks of 4 rows
			int bn = b.isContiguous() ? alen%4 : alen;
			for( int k=apos; k<apos+bn; k++ )
				vectMultiplyAdd(avals[k], b.values(aix[k]), cvals, b.pos(aix[k]), cix, n);
			//compute blocks of 4 rows (core inner loop)
			double[] bvals = b.valuesAt(0); //only for contiguous
			for( int k=apos+bn; k<apos+alen; k+=4 )
				vectMultiplyAdd4( avals[k], avals[k+1], avals[k+2], avals[k+3], bvals, cvals,
					aix[k]*n, aix[k+1]*n, aix[k+2]*n, aix[k+3]*n, cix, n );
		}
	}
	
	private static void matrixMultSparseDenseMM(SparseBlock a, DenseBlock b, DenseBlock c, int n, int cd, long xsp, int rl, int ru) {
		//blocksizes to fit blocks of B (dense) and several rows of A/C in common L2 cache size, 
		//while blocking A/C for L1/L2 yet allowing long scans (2 pages) in the inner loop over j
		//in case of almost ultra-sparse matrices, we cannot ensure the blocking for the rhs and
		//output - however, in this case it's unlikely that we consume every cache line in the rhs
		final int blocksizeI = (int) (8L*xsp);
		final int blocksizeK = (int) (8L*xsp);
		final int blocksizeJ = 1024; 
		
		//temporary array of current sparse positions
		int[] curk = new int[Math.min(blocksizeI, ru-rl)];
		
		//blocked execution over IKJ 
		for( int bi = rl; bi < ru; bi+=blocksizeI ) {
			Arrays.fill(curk, 0); //reset positions
			for( int bk = 0, bimin = Math.min(ru, bi+blocksizeI); bk < cd; bk+=blocksizeK ) {
				for( int bj = 0, bkmin = Math.min(cd, bk+blocksizeK); bj < n; bj+=blocksizeJ ) {
					int bjlen = Math.min(n, bj+blocksizeJ)-bj;
					
					//core sub block matrix multiplication
					for( int i=bi; i<bimin; i++ ) {
						if( a.isEmpty(i) ) continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						double[] cvals = c.values(i);
						int cix = c.pos(i, bj);
						
						int k = curk[i-bi] + apos;
						//rest not aligned to blocks of 4 rows
						int bn = b.isContiguous() ? alen%4 : alen;
						for( ; k<apos+bn && aix[k]<bkmin; k++ )
							vectMultiplyAdd(avals[k], b.values(aix[k]), cvals, b.pos(aix[k],bj), cix, bjlen); 
						//compute blocks of 4 rows (core inner loop), allowed to exceed bkmin
						double[] bvals = b.valuesAt(0); //only for contiguous
						for( ; k<apos+alen && aix[k]<bkmin; k+=4 )
							vectMultiplyAdd4( avals[k], avals[k+1], avals[k+2], avals[k+3], bvals, cvals, 
								aix[k]*n+bj, aix[k+1]*n+bj, aix[k+2]*n+bj, aix[k+3]*n+bj, cix, bjlen );
						//update positions on last bj block
						if( bj+bjlen==n )
							curk[i-bi] = k - apos;
					}
				}
			}
		}
	}

	private static void matrixMultSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean pm2, boolean sparse, int rl, int ru) {
		SparseBlock a = m1.sparseBlock;
		SparseBlock b = m2.sparseBlock;
		int m = m1.rlen;
		int cd = m1.clen;
		int n = m2.clen;
		
		// MATRIX-MATRIX (VV, MV not applicable here because V always dense)
		if( pm2 && m==1 )               //VECTOR-MATRIX
			matrixMultSparseSparseVM(a, b, ret.getDenseBlock(), rl, ru);
		else if( sparse )               //SPARSE OUPUT
			ret.setNonZeros(matrixMultSparseSparseSparseMM(a, b, ret.getSparseBlock(), n, rl, ru));
		else if( m2.nonZeros < 2048 )   //MATRIX-SMALL MATRIX
			matrixMultSparseSparseMMSmallRHS(a, b, ret.getDenseBlock(), rl, ru);
		else                            //MATRIX-MATRIX
			matrixMultSparseSparseMM(a, b, ret.getDenseBlock(), m, cd, m1.nonZeros, rl, ru);
	}
	
	private static void matrixMultSparseSparseVM(SparseBlock a, SparseBlock b, DenseBlock c, int rl, int ru) {
		//parallelization over rows in rhs matrix
		if( a.isEmpty(0) )
			return;
		
		int alen = a.size(0);
		int[] aix = a.indexes(0);
		double[] avals = a.values(0);
		double[] cvals = c.valuesAt(0);
		int rlix = (rl==0) ? 0 : a.posFIndexGTE(0,rl);
		rlix = (rlix>=0) ? rlix : alen;
		
		for( int k=rlix; k<alen && aix[k]<ru; k++ )
			if( !b.isEmpty(aix[k]) ) {
				int bpos = b.pos(aix[k]);
				int blen = b.size(aix[k]);
				int[] bix = b.indexes(aix[k]);
				double[] bvals = b.values(aix[k]);
				vectMultiplyAdd(avals[k], bvals, cvals, bix, bpos, 0, blen);
			}
	}
	
	private static long matrixMultSparseSparseSparseMM(SparseBlock a, SparseBlock b, SparseBlock c, int n, int rl, int ru) {
		double[] tmp = new double[n];
		long nnz = 0;
		for( int i=rl; i<Math.min(ru, a.numRows()); i++ ) {
			if( a.isEmpty(i) ) continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			//compute row output in dense buffer
			boolean hitNonEmpty = false;
			for(int k = apos; k < apos+alen; k++) {
				int aixk = aix[k];
				if( b.isEmpty(aixk) ) continue;
				vectMultiplyAdd(avals[k], b.values(aixk), tmp,
					b.indexes(aixk), b.pos(aixk), 0, b.size(aixk));
				hitNonEmpty = true;
			}
			//copy dense buffer into sparse output (CSR or MCSR)
			if( hitNonEmpty ) {
				int rnnz = UtilFunctions.computeNnz(tmp, 0, n);
				nnz += rnnz;
				c.allocate(i, rnnz);
				for(int j=0; j<n; j++)
					if( tmp[j] != 0 ) {
						c.append(i, j, tmp[j]);
						tmp[j] = 0;
					}
			}
		}
		return nnz;
	}
	
	private static void matrixMultSparseSparseMMSmallRHS(SparseBlock a, SparseBlock b, DenseBlock c, int rl, int ru) {
		for( int i=rl; i<Math.min(ru, a.numRows()); i++ ) {
			if( a.isEmpty(i) ) continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			double[] cvals = c.values(i);
			int cix = c.pos(i);
			for(int k = apos; k < apos+alen; k++) {
				int aixk = aix[k];
				if( b.isEmpty(aixk) ) continue;
				vectMultiplyAdd(avals[k], b.values(aixk), cvals,
					b.indexes(aixk), b.pos(aixk), cix, b.size(aixk));
			}
		}
	}
	
	private static void matrixMultSparseSparseMM(SparseBlock a, SparseBlock b, DenseBlock c, int m, int cd, long nnz1, int rl, int ru) {
		//block sizes for best-effort blocking w/ sufficient row reuse in B yet small overhead
		final int blocksizeI = 32;
		final int blocksizeK = Math.max(32,
			UtilFunctions.nextIntPow2((int)Math.pow((double)m*cd/nnz1, 2)));
		
		//temporary array of current sparse positions
		int[] curk = new int[Math.min(blocksizeI, ru-rl)];
		
		//blocked execution over IK 
		for( int bi = rl; bi < ru; bi+=blocksizeI ) {
			Arrays.fill(curk, 0); //reset positions
			for( int bk = 0, bimin = Math.min(ru, bi+blocksizeI); bk < cd; bk+=blocksizeK ) {
				final int bkmin = Math.min(cd, bk+blocksizeK); 
				//core sub block matrix multiplication
				for( int i=bi; i<bimin; i++ ) {
					if( a.isEmpty(i) ) continue;
					final int apos = a.pos(i);
					final int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					double[] cvals = c.values(i);
					int cix = c.pos(i);
					int k = curk[i-bi] + apos;
					for(; k < apos+alen && aix[k]<bkmin; k++) {
						if( b.isEmpty(aix[k]) ) continue;
						vectMultiplyAdd(avals[k], b.values(aix[k]), cvals,
							b.indexes(aix[k]), b.pos(aix[k]), cix, b.size(aix[k]));
					}
					curk[i-bi] = k - apos;
				}
			}
		}
	}
	
	@SuppressWarnings("unused")
	private static void matrixMultSparseSparseMMGeneric(SparseBlock a, SparseBlock b, DenseBlock c, int rl, int ru) {
		for( int i=rl; i<Math.min(ru, a.numRows()); i++ ) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			double[] cvals = c.values(i);
			int cix = c.pos(i);
			for(int k = apos; k < apos+alen; k++) {
				if( b.isEmpty(aix[k]) ) continue;
				double val = avals[k];
				int bpos = b.pos(aix[k]);
				int blen = b.size(aix[k]);
				int[] bix = b.indexes(aix[k]);
				double[] bvals = b.values(aix[k]);
				for(int j = bpos; j < bpos+blen; j++)
					cvals[cix+bix[j]] += val * bvals[j];
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
	 * @param m1 first matrix
	 * @param m2 second matrix
	 * @param ret result matrix
	 * @param rl row lower bound
	 * @param ru row upper bound
	 */
	private static void matrixMultUltraSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, boolean m1Perm, int rl, int ru) {
		final boolean leftUS = m1.isUltraSparse()
			|| (m1.isUltraSparse(false) && !m2.isUltraSparse())
			|| (m1.sparse && !m2.sparse);
		if( m1 == m2 ) //self-product
			matrixMultUltraSparseSelf(m1, ret, rl, ru);
		else if( leftUS || m1Perm )
			matrixMultUltraSparseLeft(m1, m2, ret, rl, ru);
		else
			matrixMultUltraSparseRight(m1, m2, ret, rl, ru);
	}
	
	private static void matrixMultUltraSparseSelf(MatrixBlock m1, MatrixBlock ret, int rl, int ru) {
		//common use case: self product G %*% G of graph resulting in dense but still sparse output
		int n = m1.clen; //m2.clen
		SparseBlock a = m1.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		double[] tmp = null;
		
		//IKJ with dense working row for lhs nnz/row > threshold
		for( int i=rl; i<ru; i++ ) {
			if( a.isEmpty(i) ) continue;
			int alen = a.size(i);
			int apos = a.pos(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			
			//compute number of aggregated non-zeros for input row
			int nnz1 = (int) Math.min(UtilFunctions.computeNnz(a, aix, apos, alen), n);
			boolean ldense = nnz1 > n / 128;
			
			//perform vector-matrix multiply w/ dense or sparse output
			if( ldense ) { //init dense tmp row
				tmp = (tmp == null) ? new double[n] : tmp;
				Arrays.fill(tmp, 0);
			}
			for( int k=apos; k<apos+alen; k++ ) {
				if( a.isEmpty(aix[k]) ) continue;
				int blen = a.size(aix[k]);
				int bpos = a.pos(aix[k]);
				int[] bix = a.indexes(aix[k]);
				double aval = avals[k];
				double[] bvals = a.values(aix[k]);
				if( ldense ) { //dense aggregation
					for( int j=bpos; j<bpos+blen; j++ )
						tmp[bix[j]] += aval * bvals[j];
				}
				else { //sparse aggregation
					c.allocate(i, nnz1);
					for( int j=bpos; j<bpos+blen; j++ )
						c.add(i, bix[j], aval * bvals[j]);
					c.compact(i); //conditional compaction
				}
			}
			if( ldense ) { //copy dense tmp row
				int nnz2 = UtilFunctions.computeNnz(tmp, 0, n);
				c.allocate(i, nnz2); //avoid reallocation
				for( int j=0; j<n; j++ )
					c.append(i, j, tmp[j]);
			}
		}
		//recompute non-zero for single-threaded
		if( rl == 0 && ru == m1.rlen ){
			ret.recomputeNonZeros();
		}
	}
	
	/**
	 * Ultra sparse kernel with guaranteed sparse output.
	 * 
	 * @param m1 Left side ultra sparse matrix
	 * @param m2 Right side Matrix Sparse or Dense
	 * @param ret Sparse output matrix
	 * @param rl Row start
	 * @param ru Row end
	 */
	private static void matrixMultUltraSparseLeft(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		final int m  = m1.rlen;
		final int n  = m2.clen;

		SparseBlock a = m1.sparseBlock;
		SparseBlockMCSR c = (SparseBlockMCSR) ret.sparseBlock;
		boolean rightSparse = m2.sparse;

		if(rightSparse)
			matrixMultUltraSparseSparseSparseLeft( a, m2.sparseBlock, c, m, n , rl, ru);
		else
			matrixMultUltraSparseDenseSparseLeftRow(a, m2.denseBlock, c, m, n, rl, ru);
		
		if( rl == 0 && ru == m ){
			ret.recomputeNonZeros();
		}
	}

	private static void matrixMultUltraSparseDenseSparseLeftRow(SparseBlock a, DenseBlock b, SparseBlockMCSR c, int m, int n,
		int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			if(a.isEmpty(i))
				continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aixs = a.indexes(i);
			final double[] avals = a.values(i);
			if(alen == 1) 
				matrixMultUltraSparseDenseSparseLeftRowOneNonZero(i, aixs[apos], avals[apos], b, c, m, n);
			else  
				matrixMultUltraSparseDenseSparseLeftRowGeneric(i, apos, alen, aixs, avals, b, c, m, n);
		}
	}

	private static void matrixMultUltraSparseDenseSparseLeftRowOneNonZero(int i, int aix, double aval, DenseBlock b,
		SparseBlockMCSR c, int m, int n) {
		final double[] bvals = b.values(aix);
		final int  bix = b.pos(aix);
		final int lnnz = UtilFunctions.computeNnz(bvals, bix, n);
		if(lnnz == 0)
			return;
		else if(lnnz == 1) {
			for(int j = 0; j < n; j++) {
				final double bv = bvals[bix + j];
				if(bv != 0) {
					SparseRowScalar s = new SparseRowScalar(j, bv * aval);
					c.set(i, s, false);
					break;
				}
			}
		}
		else {
			setSparseRowVector(lnnz, aval, bvals, bix, n, i, c);
		}
	}

	private static void setSparseRowVector(int lnnz, double aval, double[] bvals, int bix, int n, int i, SparseBlockMCSR c) {
		final double[] vals = new double[lnnz];
		final int[] idx = new int[lnnz];
		for(int j = 0, o = 0; j < n; j++) {
			final double bv = bvals[bix + j];
			if(bv != 0) {
				vals[o] = bv * aval;
				idx[o] = j;
				o++;
			}
		}
		SparseRowVector v = new SparseRowVector(vals, idx);
		c.set(i, v, false);
	}

	private static void matrixMultUltraSparseDenseSparseLeftRowGeneric(int i, int apos, int alen, int[] aixs,
		double[] avals, DenseBlock b, SparseBlock c, int m, int n){
			for(int k = apos; k < apos + alen; k++) {
					final double aval = avals[k];
					final int aix = aixs[k];
					for(int j = 0; j < n; j++) {
						double cvald = aval * b.get(aix, j);
						if(cvald != 0)
							c.add(i, j,  cvald);
					}
				}
		}

	private static void matrixMultUltraSparseSparseSparseLeft(SparseBlock a, SparseBlock b, SparseBlockMCSR c, int m,
		int n, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			if(a.isEmpty(i))
				continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aixs = a.indexes(i);
			final double[] avals = a.values(i);
			if(alen == 1)
				matrixMultUltraSparseSparseSparseLeftRowOneNonZero(i, aixs[apos], avals[apos], b, c, m, n);
			else // GENERAL CASE
				matrixMultUltraSparseSparseSparseLeftRowGeneric(i, apos, alen, aixs, avals, b, c, m, n);
		}
	}

	private static void matrixMultUltraSparseSparseSparseLeftRowOneNonZero(int i, int aix, double aval, SparseBlock b, SparseBlockMCSR c, int m, int n){
		if(!b.isEmpty(aix)) {
			c.set(i, b.get(aix), true);
			// optional scaling if not pure selection
			if(aval != 1) {
				if(c.get(i) instanceof SparseRowScalar) {
					SparseRowScalar sv = (SparseRowScalar) c.get(i);
					c.set(i, new SparseRowScalar(sv.getIndex(), sv.getValue() * aval), false);
				}
				else
					vectMultiplyInPlace(aval, c.values(i), c.pos(i), c.size(i));
			}
		}
	}

	private static void matrixMultUltraSparseSparseSparseLeftRowGeneric(int i, int apos, int alen, int[] aixs,
		double[] avals, SparseBlock b, SparseBlockMCSR c, int m, int n) {
		for(int k = apos; k < apos + alen; k++) {
			final int aix = aixs[k];
			if(b.isEmpty(aix))
				continue;
			final double aval = avals[k];
			final int bpos = b.pos(aix);
			final int blen = b.size(aix) + bpos;
			final int[] bix = b.indexes(aix);
			final double[] bvals = b.values(aix);
			if(!c.isAllocated(i))
				c.allocate(i, Math.max(blen - bpos, 2)); // guarantee a vector for efficiency
			final SparseRowVector v = (SparseRowVector) c.get(i);
			if(v.size() == n){ // If output row is dense already
				final double[] vvals = v.values();
				for(int bo = bpos; bo < blen; bo++) 
					vvals[bix[bo]] += aval * bvals[bo];
			}
			else
				for(int bo = bpos; bo < blen; bo++) 
					v.add(bix[bo], aval * bvals[bo]);
			
		}
	}

	
	private static void matrixMultUltraSparseRight(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		if(ret.isInSparseFormat()){
			if(m1.isInSparseFormat())
				matrixMultUltraSparseRightSparseMCSRLeftSparseOut(m1, m2, ret, rl, ru);
			else if (m2.isInSparseFormat())
				matrixMultUltraSparseRightDenseLeftSparseOut(m1, m2, ret, rl, ru);
			else 
				matrixMultUltraSparseDenseInput(m1, m2, ret, rl, ru);
		}
		else if(ret.getDenseBlock().isContiguous())
			matrixMultUltraSparseRightDenseOut(m1, m2, ret, rl, ru);
		else
			matrixMultUltraSparseRightGeneric(m1, m2, ret, rl, ru);
	}

	private static void matrixMultUltraSparseRightDenseOut(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		final int cd = m1.clen;
		final int kd = m2.clen;
		double[] retV = ret.getDenseBlockValues();

		// right is ultra-sparse (KJI)
		final SparseBlock b = m2.sparseBlock;
		for(int k = 0; k < cd; k++) {
			if(b.isEmpty(k))
				continue;
			final int bpos = b.pos(k);
			final int blen = b.size(k);
			final int[] bixs = b.indexes(k);
			final double[] bvals = b.values(k);
			for(int j = bpos; j < bpos + blen; j++) {
				double bval = bvals[j];
				int bix = bixs[j];
				for(int i = rl; i < ru; i++) 
					retV[i *kd + bix] += bval * m1.get(i, k);
			}
		}
	}

	private static void matrixMultUltraSparseRightSparseMCSRLeftSparseOut(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		final int cd = m1.clen;

		// right is ultra-sparse (KJI)
		final SparseBlock a = m1.sparseBlock;
		final SparseBlock b = m2.sparseBlock;
		final SparseBlockMCSR r = (SparseBlockMCSR) ret.sparseBlock;
		
		for(int k = 0; k < cd; k++) {
			if(b.isEmpty(k))
				continue;
			final int bpos = b.pos(k);
			final int blen = b.size(k);
			final int[] bixs = b.indexes(k);
			final double[] bvals = b.values(k);
			for(int i = rl; i < ru; i++) {
				if(a.isEmpty(i))
					continue;
				final double cvald = a.get(i, k);
				// since the left side is sparse as well, it is likely that this value is zero.
				// therefore we reorder the loop to access the value here.
				if(cvald != 0) {
					for(int j = bpos; j < bpos + blen; j++) {
						final int bix = bixs[j];
						final double bval = bvals[j];
						final double cval = r.get(i, bix);
						r.set(i, bix, cval + bval * cvald);
					}
				}
			}
		}
	}

	private static void matrixMultUltraSparseRightDenseLeftSparseOut(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		final int cd = m1.clen;
		final DenseBlock  a = m1.denseBlock;
		final SparseBlock b = m2.sparseBlock;
		final SparseBlockMCSR c = (SparseBlockMCSR) ret.sparseBlock;

		for(int k = 0; k < cd; k++){
			if(b.isEmpty(k))
				continue; // skip emptry rows right side.
			final int bpos = b.pos(k);
			final int blen = b.size(k);
			final int[] bixs = b.indexes(k);
			final double[] bvals = b.values(k);
			for(int i = rl; i < ru; i++) 
				mmDenseMatrixSparseRow(bpos, blen, bixs, bvals, k, i, a, c);
		}
	}

	private static void matrixMultUltraSparseDenseInput(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru){
		final int cd = m1.clen;
		final int rc = m2.clen;
		final DenseBlock a = m1.denseBlock;
		final DenseBlock b = m2.denseBlock;
		final SparseBlockMCSR c = (SparseBlockMCSR) ret.sparseBlock;

		for(int i = rl; i < ru; i++) {
			// it is known that the left matrix is most likely containing many zeros.
			final double[] av = a.values(i);
			final int pos = a.pos(i);
			for(int k = 0; k < cd; k++) {
				final double v = av[pos + k];
				if(v != 0) {
					final double[] bv = b.values(k);
					final int posb = b.pos(k);
					for(int j = 0; j < rc; j++) {
						c.add(i,j, bv[posb + j] * v);
					}
				}
			}
		}
	}

	private static void mmDenseMatrixSparseRow(int bpos, int blen, int[] bixs, double[] bvals, int k, int i,
		DenseBlock a, SparseBlockMCSR c) {
		final double[] aval = a.values(i);
		final int apos = a.pos(i);
		if(!c.isAllocated(i))
			c.allocate(i, Math.max(blen, 2));
		final SparseRowVector srv = (SparseRowVector) c.get(i); // guaranteed
		for(int j = bpos; j < bpos + blen; j++) { // right side columns
			final int bix = bixs[j];
			final double bval = bvals[j];
			srv.add(bix, bval * aval[apos + k]);
		}

	}

	private static void matrixMultUltraSparseRightGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int rl, int ru) {
		final int cd = m1.clen;

		// right is ultra-sparse (KJI)
		final SparseBlock b = m2.sparseBlock;
		for(int k = 0; k < cd; k++) {
			if(b.isEmpty(k))
				continue;
			final int bpos = b.pos(k);
			final int blen = b.size(k);
			final int[] bixs = b.indexes(k);
			final double[] bvals = b.values(k);
			for(int j = bpos; j < bpos + blen; j++) {
				double bval = bvals[j];
				int bix = bixs[j];
				for(int i = rl; i < ru; i++) {
					double cvald = bval * m1.get(i, k);
					if(cvald != 0) {
						double cval = ret.get(i, bix);
						ret.set(i, bix, cval + cvald);
					}
				}
			}
		}
	}

	private static void matrixMultChainDense(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int rl, int ru) 
	{
		DenseBlock a = mX.getDenseBlock();
		double[] b = mV.getDenseBlockValues();
		double[] w = (mW!=null) ? mW.getDenseBlockValues() : null;
		double[] c = ret.getDenseBlockValues();
		final int cd = mX.clen; //features in X
		boolean weights = (ct == ChainType.XtwXv);
		boolean weights2 = (ct == ChainType.XtXvy);
		
		//temporary array for cache blocking
		//(blocksize chosen to fit b+v in L2 (256KB) for default 1k blocks)
		final int blocksizeI = 24; // constraint: factor of 4
		final int blocksizeJ = 1024;
		double[] tmp = new double[blocksizeI];
		final int bn = (ru-rl) % blocksizeI;
		
		//compute rest (not aligned to blocksize)
		for( int i=rl; i < rl+bn; i++ ) {
			double[] avals = a.values(i);
			int aix = a.pos(i);
			double val = (b == null) ? 0 :
				dotProduct(avals, b, aix, 0, cd);
			val *= (weights) ? w[i] : 1;
			val -= (weights2) ? w[i] : 0;
			vectMultiplyAdd(val, avals, c, aix, 0, cd);
		}
		
		//blockwise mmchain computation
		for( int bi=rl+bn; bi < ru; bi+=blocksizeI ) 
		{
			//compute 1st matrix-vector for row block
			Arrays.fill(tmp, 0);
			if( b != null ) {
				for( int bj=0; bj<cd; bj+=blocksizeJ ) {
					int bjmin = Math.min(cd-bj, blocksizeJ);
					for( int i=0; i < blocksizeI; i++ )
						tmp[i] += dotProduct(a.values(bi+i), b, a.pos(bi+i,bj), bj, bjmin);
				}
			}
			
			//multiply/subtract weights (in-place), if required
			if( weights ) 
				vectMultiply(w, tmp, bi, 0, blocksizeI);
			else if( weights2 )
				vectSubtract(w, tmp, bi, 0, blocksizeI);
			
			//compute 2nd matrix vector for row block and aggregate
			for( int bj = 0; bj<cd; bj+=blocksizeJ ) {
				int bjmin = Math.min(cd-bj, blocksizeJ);
				if( a.isContiguous() ) {
					double[] avals = a.values(0);
					for( int i=0, aix=bi*cd+bj; i<blocksizeI; i+=4, aix+=4*cd )
						vectMultiplyAdd4(tmp[i], tmp[i+1], tmp[i+2], tmp[i+3],
							avals, c, aix, aix+cd, aix+2*cd, aix+3*cd, bj, bjmin);
				}
				else {
					for( int i=0; i<blocksizeI; i++ )
						vectMultiplyAdd(tmp[i], a.values(bi+i), c, a.pos(bi+i,bj), bj, bjmin);
				}
			}
		}
	}

	private static void matrixMultChainSparse(MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, ChainType ct, int rl, int ru) 
	{
		final SparseBlock a = mX.sparseBlock;
		final double[] b = mV.getDenseBlockValues();
		final double[] w = (mW != null) ? mW.getDenseBlockValues() : null;
		final double[] c = ret.getDenseBlockValues();
		
		//row-wise mmchain computation
		if(ct == ChainType.XtXvy)
			matrixMultChainSparseXtXvy(a, b, w, c, rl, ru);
		else if(b != null) {
			if(ct == ChainType.XtwXv)
				matrixMultChainSparseXtwXv(a, b, w, c, rl, ru);
			else // XtXv
				matrixMultChainSparseXtXv(a, b, c, rl, ru);
		}
	}

	private static final void matrixMultChainSparseXtXv(SparseBlock a, double[] b,  double[] c, int rl,
		int ru) {
		for(int i = rl; i < ru; i++) {
			if(a.isEmpty(i))
				continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);

			// compute 1st matrix-vector dot product
			final double val = dotProduct(avals, b, aix, apos, 0, alen);

			// compute 2nd matrix vector and aggregate
			if(val != 0)
				vectMultiplyAdd(val, avals, c, aix, apos, 0, alen);
		}
	}

	private static final void matrixMultChainSparseXtwXv(SparseBlock a, double[] b, double[] w, double[] c, int rl,
		int ru) {
		for(int i = rl; i < ru; i++) {
			if(w[i] == 0 || a.isEmpty(i))
				continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			// compute 1st matrix-vector dot product
			double val = dotProduct(avals, b, aix, apos, 0, alen) * w[i];

			// compute 2nd matrix vector and aggregate
			if(val != 0)
				vectMultiplyAdd(val, avals, c, aix, apos, 0, alen);
		}
	}


	private static final void matrixMultChainSparseXtXvy(SparseBlock a, double[] b, double[] w, double[] c, int rl,
		int ru) {
		if(b == null && w == null) {// early abort
			return;
		}
		else if(b == null && w != null) { // short case with empty B.
			for(int i = rl; i < ru; i++) {
				final double val = -w[i];
				if(val != 0 && !a.isEmpty(i)) {
					final int apos = a.pos(i);
					final int alen = a.size(i);
					final int[] aix = a.indexes(i);
					final double[] avals = a.values(i);
					vectMultiplyAdd(val, avals, c, aix, apos, 0, alen);
				}
			}
		}
		else { // case XtXvy
			// row-wise mmchain computation
			for(int i = rl; i < ru; i++) {
				if(a.isEmpty(i))
					continue;
				final int apos = a.pos(i);
				final int alen = a.size(i);
				final int[] aix = a.indexes(i);
				final double[] avals = a.values(i);

				// compute 1st matrix-vector dot product
				double val = dotProduct(avals, b, aix, apos, 0, alen);

				// multiply/subtract weights, if required
				if(w != null)
					val -= w[i];

				// compute 2nd matrix vector and aggregate
				if(val != 0)
					vectMultiplyAdd(val, avals, c, aix, apos, 0, alen);
			}
		}
	}

	private static void matrixMultTransposeSelfDense( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int rl, int ru ) {
		//2) transpose self matrix multiply dense
		// (compute only upper-triangular matrix due to symmetry)
		DenseBlock a = m1.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		int m = m1.rlen;
		int n = m1.clen;
		
		if( leftTranspose ) // t(X)%*%X
		{
			if( n==1 ) //VECTOR (col)
			{
				double[] avals = a.valuesAt(0);
				c.set(0, 0, dotProduct(avals, avals, m));
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
								double[] cvals = c.values(i);
								int cixj = c.pos(i, bj);
								
								if( a.isContiguous(bk, bkmin-1) ) {
									double[] avals = a.values(bk);
									int aixi = a.pos(bk, i);
									int bkpos = a.pos(bk, bj);
									
									//determine nnz of a (for sparsity-aware skipping of rows)
									int knnz = copyNonZeroElements(avals, aixi, bkpos, n, nx, ta, tbi, bklen);
									
									//rest not aligned to blocks of 4 rows
									final int bn = knnz % 4;
									switch( bn ){
										case 1: vectMultiplyAdd(ta[0], avals, cvals, tbi[0], cixj, bjlen); break;
										case 2: vectMultiplyAdd2(ta[0],ta[1], avals, cvals, tbi[0], tbi[1], cixj, bjlen); break;
										case 3: vectMultiplyAdd3(ta[0],ta[1],ta[2], avals, cvals, tbi[0], tbi[1],tbi[2], cixj, bjlen); break;
									}
									
									//compute blocks of 4 rows (core inner loop)
									for( int k = bn; k<knnz; k+=4 ){
										vectMultiplyAdd4( ta[k], ta[k+1], ta[k+2], ta[k+3], avals, cvals,
											tbi[k], tbi[k+1], tbi[k+2], tbi[k+3], cixj, bjlen );
									}
								}
								else {
									for( int k = bk; k<bkmin; k++ ) {
										double[] avals = a.values(bk);
										int aix = a.pos(bk, i);
										if( avals[aix] != 0 )
											vectMultiplyAdd( avals[aix], a.values(k),
												cvals, a.pos(k, bj), cixj, bjlen );
									}
								}
							}
						}
			}
		}
		else // X%*%t(X)
		{
			if( m==1 ) //VECTOR
			{
				double[] avals = a.valuesAt(0);
				c.set(0, 0, dotProduct(avals, avals, n));
			}
			else //MATRIX
			{
				//algorithm: scan c, foreach ci,j: scan row of a and t(a) (IJK)
			
				//1) Unrolled inner loop, for better ILP
				//2) Blocked execution, for less cache trashing in parallel exec 
				//   (we block such that lhs, rhs, and output roughly fit into L2, output in L1)
				//3) Asymmetric block sizes and exploitation of result symmetry
				int blocksizeK = 1024; //two memory pages for sufficiently long scans
				int blocksizeIJ = L2_CACHESIZE / 8 / blocksizeK / 2 - 1; //15
			
				//blocked execution over IKJ (lhs/rhs in L2, output in L1)
				for( int bi = rl; bi<ru; bi+=blocksizeIJ ) 
					for( int bk = 0, bimin = Math.min(ru, bi+blocksizeIJ); bk<n; bk+=blocksizeK )
						for( int bj = bi, bklen = Math.min(blocksizeK, n-bk); bj<m; bj+=blocksizeIJ ) {
							//core tsmm block operation (15x15 vectors of length 1K elements)
							int bjmin = Math.min(m, bj+blocksizeIJ);
							for( int i=bi; i<bimin; i++ ) {
								final int bjmax = Math.max(i,bj); //from i due to symmetry
								double[] avals = a.values(i), cvals = c.values(i);
								int aix = a.pos(i, bk), cix = c.pos(i);
								for(int j=bjmax; j <bjmin; j++) 
									cvals[ cix+j ] += dotProduct(avals, a.values(j), aix, a.pos(j, bk), bklen);
							}
						}
			}
		}
	}

	private static void matrixMultTransposeSelf(MatrixBlock m1, MatrixBlock m1t, MatrixBlock ret, boolean leftTranspose, int rl, int ru) {
		if(m1.sparse && ret.sparse) {
			if( m1t == null )
				matrixMultTransposeSelfUltraSparse(m1, ret, leftTranspose, rl, ru);
			else
				matrixMultTransposeSelfUltraSparse2(m1, m1t, ret, leftTranspose, rl, ru);
		}
		else if( m1.sparse )
			matrixMultTransposeSelfSparse(m1, ret, leftTranspose, rl, ru);
		else 
			matrixMultTransposeSelfDense(m1, ret, leftTranspose, rl, ru );
	}
	
	private static void matrixMultTransposeSelfSparse( MatrixBlock m1, MatrixBlock ret, boolean leftTranspose, int rl, int ru ) {
		//2) transpose self matrix multiply sparse
		// (compute only upper-triangular matrix due to symmetry)
		SparseBlock a = m1.sparseBlock;
		DenseBlock c = ret.getDenseBlock();
		int m = m1.rlen;
		
		if( leftTranspose ) // t(X)%*%X 
		{
			//only general case (because vectors always dense)
			//algorithm: scan rows, foreach row self join (KIJ)
			final int n = m1.clen;
			final int arlen = a.numRows();
			for( int r=0; r<arlen; r++ ) {
				if( a.isEmpty(r) ) continue;
				final int alen = a.size(r);
				final double[] avals = a.values(r);
				final int apos = a.pos(r);
				if( alen == n ) { //dense row
					for (int i = rl; i < ru; i++){
						double[] cvals = c.values(i);
						int cix = c.pos(i);
						double val = avals[i + apos];
						for(int j = i; j < m1.clen; j++)
							cvals[cix + j] +=val * avals[j + apos];
					}
				}
				else { //non-full sparse row
					int[] aix = a.indexes(r);
					int rlix = (rl==0) ? 0 : a.posFIndexGTE(r, rl);
					rlix = (rlix>=0) ? apos+rlix : apos+alen;
					int len = apos + alen;
					for(int i = rlix; i < len && aix[i] < ru; i++)
						vectMultiplyAdd(avals[i], avals, c.values(aix[i]), aix, i, c.pos(aix[i]), len - i);
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
					c.set(0, 0, dotProduct(avals, avals, alen));
				}
			}
			else //MATRIX
			{
				//note: reorg to similar layout as t(X)%*%X because faster than 
				//direct computation with IJK (no dependencies/branches in inner loop)
				//see preprocessMatrixMultTransposeSelf m1<-tmpBlock
				m = m1.clen;
				
				//algorithm: scan rows, foreach row self join (KIJ)
				int arlen = a.numRows();
				for( int r=0; r<arlen; r++ ) {
					if( a.isEmpty(r) ) continue;
					int apos = a.pos(r);
					int alen = a.size(r);
					int[] aix = a.indexes(r);
					double[] avals = a.values(r);
					int rlix = (rl==0) ? 0 : a.posFIndexGTE(r, rl);
					rlix = (rlix>=0) ? apos+rlix : apos+alen;
					for(int i = rlix; i < apos+alen && aix[i]<ru; i++) {
						double val = avals[i];
						if( val != 0 )
							vectMultiplyAdd(val, avals, c.values(aix[i]),
								aix, i, c.pos(aix[i]), alen-i);
					}
				}
			}
		}
	}
	
	private static void matrixMultTransposeSelfUltraSparse(MatrixBlock m1, MatrixBlock ret, boolean leftTranspose,
		int rl, int ru) {
		SparseBlock a = m1.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		int m = m1.rlen;

		if(leftTranspose) {
			// Operation t(X)%*%X, sparse input and output
			for(int i=0; i<m; i++)
				c.allocate(i, 8*SparseRowVector.initialCapacity);
			SparseRow[] sr = ((SparseBlockMCSR) c).getRows();
			for( int r=0; r<a.numRows(); r++ ) {
				if( a.isEmpty(r) ) continue;
				final int alen = a.size(r);
				final double[] avals = a.values(r);
				final int apos = a.pos(r);
				int[] aix = a.indexes(r);
				int rlix = (rl==0) ? 0 : a.posFIndexGTE(r, rl);
				if(rlix>=0) {
					int len = apos + alen;
					for(int i = rlix; i < len && aix[i] < ru; i++) {
						for (int k = a.posFIndexGTE(r, aix[i]); k < len; k++) {
							sr[aix[i]].add(c.pos(k) + aix[k], avals[i] * avals[k]);
						}
					}
				}
			}
		}
		else {
			// Operation X%*%t(X), sparse input and output
			final int blocksize = 256;
			for(int bi=rl; bi<ru; bi+=blocksize) { //blocking rows in X
				int bimin = Math.min(bi+blocksize, ru);
				for(int i=bi; i<bimin; i++) //preallocation
					if( !a.isEmpty(i) )
						c.allocate(i, 8*SparseRowVector.initialCapacity); //heuristic
				for(int bj=bi; bj<m; bj+=blocksize ) { //blocking cols in t(X)
					int bjmin = Math.min(bj+blocksize, m);
					for(int i=bi; i<bimin; i++) { //rows in X
						if( a.isEmpty(i) ) continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for(int j=Math.max(bj,i); j<bjmin; j++) { //cols in t(X)
							if( a.isEmpty(j) ) continue;
							int bpos = a.pos(j);
							int blen = a.size(j);
							int[] bix = a.indexes(j);
							double[] bvals = a.values(j);

							//compute sparse dot product and append
							double v = dotProduct(avals, aix, apos, alen, bvals, bix, bpos, blen);
							if( v != 0 )
								c.append(i, j, v);
						}
					}
				}
			}
		}
	}
	
	//alternative matrixMultTransposeSelfUltraSparse2 w/ IKJ iteration order and sparse updates
	private static void matrixMultTransposeSelfUltraSparse2( MatrixBlock m1, MatrixBlock m1t, MatrixBlock ret, boolean leftTranspose, int rl, int ru ) {
		SparseBlock a;
		SparseBlock b;
		if( leftTranspose ) {
			a = m1t.sparseBlock;
			b = m1.sparseBlock;
		}
		else {
			a = m1.sparseBlock;
			b = m1t.sparseBlock;
		}

		// Operation X%*%t(X), sparse input and output
		SparseBlock c = ret.sparseBlock;
		for(int i=rl; i<ru; i++) { //rows in X
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			//aggregate arow %*% B into output
			for(int k=apos; k<apos+alen; k++) {
				int aixk = aix[k];
				double aval = avals[k];
				if( b.isEmpty(aixk) ) continue;
				int bpos = b.pos(aixk);
				int bpos2 = b.posFIndexGTE(aixk, i);
				if( bpos2 < 0 ) continue;
				int blen = b.size(aixk);
				int[] bix = b.indexes(aixk);
				double[] bvals = b.values(aixk);
				//sparse updates for ultra-sparse output
				for(int k2 = bpos+bpos2; k2<bpos+blen; k2++) {
					c.add(i, bix[k2], aval*bvals[k2]);
				}
			}
		}
	}

	private static void matrixMultPermuteDense(MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru) {
		double[] a = pm1.getDenseBlockValues();
		DenseBlock b = m2.getDenseBlock();
		DenseBlock c = ret1.getDenseBlock();

		final int n = m2.clen;
		final int blen = ret1.getNumRows();
		int lastblk = -1;
		
		for( int i=rl; i<ru; i++ ) {
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast
			if( pos > 0 ) { //selected row
				int bpos = (pos-1) % blen;
				int blk = (pos-1) / blen;
				//allocate and switch to second output block
				//(never happens in cp, correct for multi-threaded usage)
				if( lastblk!=-1 && lastblk<blk ) {
					ret2.sparse = false;
					ret2.allocateDenseBlock();
					c = ret2.getDenseBlock();
				}
				//memcopy entire dense row into target position
				System.arraycopy(b.values(i), b.pos(i), c.values(bpos), c.pos(bpos), n);
				lastblk = blk;
			}
		}
	}

	private static void matrixMultPermuteDenseSparse( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
	{
		double[] a = pm1.getDenseBlockValues(); //vector
		DenseBlock b = m2.getDenseBlock();
		SparseBlock c = ret1.sparseBlock;

		final int n = m2.clen;
		final int blen = ret1.getNumRows();
		
		int lastblk = -1;
		for( int i=rl; i<ru; i++ ) {
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast
			if( pos > 0 ) { //selected row
				double[] bvals = b.values(i);
				int bix = b.pos(i);
				int bpos = (pos-1) % blen;
				int blk = (pos-1) / blen;
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
					c.append(bpos, j, bvals[bix+j]);
				lastblk = blk;
			}
		}
	}

	private static void matrixMultPermuteSparse( MatrixBlock pm1, MatrixBlock m2, MatrixBlock ret1, MatrixBlock ret2, int rl, int ru)
	{
		double[] a = pm1.getDenseBlockValues(); //vector
		SparseBlock b = m2.sparseBlock;
		SparseBlock c = ret1.sparseBlock;

		final int blen = ret1.getNumRows();
		
		int lastblk = -1;
		for( int i=rl; i<ru; i++ )  {
			//compute block index and in-block indexes
			int pos = UtilFunctions.toInt( a[ i ]); //safe cast
			if( pos > 0 ) { //selected row
				int bpos = (pos-1) % blen;
				int blk = (pos-1) / blen;
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

	private static void matrixMultWSLossDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		DenseBlock x = mX.getDenseBlock();
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		DenseBlock w = (mW!=null)? mW.getDenseBlock() : null;
		final int n = mX.clen;
		final int cd = mU.clen;
		double wsloss = 0;
		
		// approach: iterate over all cells of X 
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
					
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		

		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			for( int bj = 0; bj < n; bj+=blocksizeIJ ){
				int bjmin = Math.min(n, bj+blocksizeIJ);
				
				// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
				if( wt==WeightsType.POST ) {
					for( int i=bi; i<bimin; i++ ) {
						double[] wvals = w.values(i), xvals = x.values(i), uvals = u.values(i);
						int xix = x.pos(i), uix = u.pos(i);
						for( int j=bj; j<bjmin; j++ ) {
							double wij = wvals[xix+j];
							if( wij != 0 ) {
								double uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
								wsloss += wij*(xvals[xix+j]-uvij)*(xvals[xix+j]-uvij); //^2
							}
						}
					}
				}
				// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post_nz weighting)
				else if( wt==WeightsType.POST_NZ ) {
					for( int i=bi; i<bimin; i++ ) {
						double[] xvals = x.values(i), uvals = u.values(i);
						int xix = x.pos(i), uix = u.pos(i);
						for( int j=bj; j<bjmin; j++ ) {
							double xij = xvals[xix+j];
							if( xij != 0 ) {
								double uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
								wsloss += (xij-uvij)*(xij-uvij); //^2
							}
						}
					}
				}
				// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
				else if( wt==WeightsType.PRE ) {
					for( int i=bi; i<bimin; i++ ) {
						double[] wvals = w.values(i), xvals = x.values(i), uvals = u.values(i);
						int xix = x.pos(i), uix = u.pos(i);
						for( int j=bj; j<bjmin; j++ ) {
							double wij = wvals[xix+j];
							double uvij = 0;
							if( wij != 0 )
								uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
							wsloss += (xvals[xix+j]-wij*uvij)*(xvals[xix+j]-wij*uvij); //^2
						}
					}
				}
				// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
				else if( wt==WeightsType.NONE ) {
					for( int i=bi; i<bimin; i++ ) {
						double[] xvals = x.values(i), uvals = u.values(i);
						int xix = x.pos(i), uix = u.pos(i);
						for( int j=bj; j<bjmin; j++) {
							double uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
							wsloss += (xvals[xix+j]-uvij)*(xvals[xix+j]-uvij); //^2
						}
					}
				}
			}
		}
		ret.set(0, 0, wsloss);
	}

	private static void matrixMultWSLossSparseDense(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, MatrixBlock ret, WeightsType wt, int rl, int ru)
	{
		SparseBlock x = mX.sparseBlock;
		SparseBlock w = (mW!=null)? mW.sparseBlock : null;
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		final int n = mX.clen; 
		final int cd = mU.clen;
		double wsloss = 0; 
		
		// Pattern 1) sum (W * (X - U %*% t(V)) ^ 2) (post weighting)
		if( wt==WeightsType.POST ) {
			// approach: iterate over W, point-wise in order to exploit sparsity
			for( int i=rl; i<ru; i++ ) {
				if( w.isEmpty(i) ) continue;
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				double[] uvals = u.values(i);
				int uix = u.pos(i);
				if( w.isAligned(i, x) ) {
					//O(n) where n is nnz in w/x 
					double[] xval = x.values(i);
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double uvij = dotProduct(uvals, v.values(wix[k]), uix, v.pos(wix[k]), cd);
						wsloss += wval[k]*(xval[k]-uvij)*(xval[k]-uvij);
					}
				}
				else {
					//O(n log m) where n/m is nnz in w/x 
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double xi = mX.get(i, wix[k]);
						double uvij = dotProduct(uvals, v.values(wix[k]), uix, v.pos(wix[k]), cd);
						wsloss += wval[k]*(xi-uvij)*(xi-uvij);
					}
				}
			}
		}
		// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post weighting)
		else if( wt==WeightsType.POST_NZ ) {
			// approach: iterate over W, point-wise in order to exploit sparsity
			// blocked over ij, while maintaining front of column indexes, where the
			// blocksize is chosen such that we reuse each vector on average 8 times.
			final int blocksizeIJ = (int) (8L*mX.rlen*mX.clen/mX.nonZeros); 
			int[] curk = new int[blocksizeIJ];
			
			for( int bi=rl; bi<ru; bi+=blocksizeIJ ) {
				int bimin = Math.min(ru, bi+blocksizeIJ);
				//prepare starting indexes for block row
				Arrays.fill(curk, 0); 
				//blocked execution over column blocks
				for( int bj=0; bj<n; bj+=blocksizeIJ ) {
					int bjmin = Math.min(n, bj+blocksizeIJ);
					for( int i=bi; i<bimin; i++ ) {
						if( x.isEmpty(i) ) continue;
						int xpos = x.pos(i);
						int xlen = x.size(i);
						int[] xix = x.indexes(i);
						double[] xval = x.values(i), uvals = u.values(i);
						int uix = u.pos(i);
						int k = xpos + curk[i-bi];
						for( ; k<xpos+xlen && xix[k]<bjmin; k++ ) {
							double uvij = dotProduct(uvals, v.values(xix[k]), uix, v.pos(xix[k]), cd);
							wsloss += (xval[k]-uvij)*(xval[k]-uvij);
						}
						curk[i-bi] = k - xpos;
					}
				}
			}
		}
		// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
		else if( wt==WeightsType.PRE ) {
			// approach: iterate over all cells of X maybe sparse and dense
			// (note: tuning similar to pattern 3 possible but more complex)
			for( int i=rl; i<ru; i++ ) {
				double[] uvals = u.values(i);
				int uix = u.pos(i);
				for( int j=0; j<n; j++ ) {
					double xij = mX.get(i, j);
					double wij = mW.get(i, j);
					double uvij = 0;
					if( wij != 0 )
						uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
					wsloss += (xij-wij*uvij)*(xij-wij*uvij);
				}
			}
		}
		// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
		else if( wt==WeightsType.NONE ) {
			//approach: use sparsity-exploiting pattern rewrite sum((X-(U%*%t(V)))^2) 
			//-> sum(X^2)-sum(2*X*(U%*%t(V))))+sum((t(U)%*%U)*(t(V)%*%V)), where each
			//parallel task computes sum(X^2)-sum(2*X*(U%*%t(V)))) and the last term
			//sum((t(U)%*%U)*(t(V)%*%V)) is computed once via two tsmm operations.
			
			final int blocksizeIJ = (int) (8L*mX.rlen*mX.clen/mX.nonZeros); 
			int[] curk = new int[blocksizeIJ];
			
			for( int bi=rl; bi<ru; bi+=blocksizeIJ ) {
				int bimin = Math.min(ru, bi+blocksizeIJ);
				//prepare starting indexes for block row
				Arrays.fill(curk, 0); 
				//blocked execution over column blocks
				for( int bj=0; bj<n; bj+=blocksizeIJ ) {
					int bjmin = Math.min(n, bj+blocksizeIJ);
					for( int i=bi; i<bimin; i++ ) {
						if( x.isEmpty(i) ) continue; 
						int xpos = x.pos(i);
						int xlen = x.size(i);
						int[] xix = x.indexes(i);
						double[] xval = x.values(i);
						double[] uvals = u.values(i);
						int uix = u.pos(i);
						int k = xpos + curk[i-bi];
						for( ; k<xpos+xlen && xix[k]<bjmin; k++ ) {
							double xij = xval[k];
							double uvij = dotProduct(uvals, v.values(xix[k]), uix, v.pos(xix[k]), cd);
							wsloss += xij * xij - 2 * xij * uvij;
						}
						curk[i-bi] = k - xpos;
					}
				}
			}
		}
		
		ret.set(0, 0, wsloss);
	}

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
				for( int i=rl; i<ru; i++ ) {
					if( w.isEmpty(i) ) continue;
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					for( int k=wpos; k<wpos+wlen; k++ ) {
						double uvij = dotProductGeneric(mU, mV, i, wix[k], cd);
						double xi = mX.get(i, wix[k]);
						wsloss += wval[k]*(xi-uvij)*(xi-uvij);
					}
				}
			}
			else //DENSE
			{
				DenseBlock w = mW.getDenseBlock();
				for( int i=rl; i<ru; i++ ) {
					double[] wvals = w.values(i);
					int wix = w.pos(i);
					for( int j=0; j<n; j++)
						if( wvals[wix+j] != 0 ) {
							double uvij = dotProductGeneric(mU, mV, i, j, cd);
							double xij = mX.get(i, j);
							wsloss += wvals[wix+j]*(xij-uvij)*(xij-uvij);
						}
				}
			}
		}
		// Pattern 1b) sum ((X!=0) * (X - U %*% t(V)) ^ 2) (post weighting)
		else if( wt==WeightsType.POST_NZ )
		{
			// approach: iterate over W, point-wise in order to exploit sparsity
			if( mX.sparse ) //SPARSE
			{
				SparseBlock x = mX.sparseBlock;
				for( int i=rl; i<ru; i++ ) {
					if( x.isEmpty(i) ) continue;
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
				DenseBlock x = mX.getDenseBlock();
				for( int i=rl; i<ru; i++ ) {
					double[] xvals = x.values(i);
					int xix = x.pos(i);
					for( int j=0; j<n; j++) {
						double xij = xvals[xix+j];
						if( xij != 0 ) {
							double uvij = dotProductGeneric(mU, mV, i, j, cd);
							wsloss += (xij-uvij)*(xij-uvij);
						}
					}
				}
			}
		}
		// Pattern 2) sum ((X - W * (U %*% t(V))) ^ 2) (pre weighting)
		else if( wt==WeightsType.PRE )
		{
			// approach: iterate over all cells of X maybe sparse and dense
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++) {
					double xij = mX.get(i, j);
					double wij = mW.get(i, j);
					double uvij = 0;
					if( wij != 0 )
						uvij = dotProductGeneric(mU, mV, i, j, cd);
					wsloss += (xij-wij*uvij)*(xij-wij*uvij);
				}
		}
		// Pattern 3) sum ((X - (U %*% t(V))) ^ 2) (no weighting)
		else if( wt==WeightsType.NONE )
		{
			//approach: use sparsity-exploiting pattern rewrite sum((X-(U%*%t(V)))^2) 
			//-> sum(X^2)-sum(2*X*(U%*%t(V))))+sum((t(U)%*%U)*(t(V)%*%V)), where each
			//parallel task computes sum(X^2)-sum(2*X*(U%*%t(V)))) and the last term
			//sum((t(U)%*%U)*(t(V)%*%V)) is computed once via two tsmm operations.
			
			if( mX.sparse ) { //SPARSE
				SparseBlock x = mX.sparseBlock;
				for( int i=rl; i<ru; i++ ) {
					if( x.isEmpty(i) ) continue;
					int xpos = x.pos(i);
					int xlen = x.size(i);
					int[] xix = x.indexes(i);
					double[] xval = x.values(i);
					for( int k=xpos; k<xpos+xlen; k++ ) {
						double xij = xval[k];
						double uvij = dotProductGeneric(mU, mV, i, xix[k], cd);
						wsloss += xij * xij - 2 * xij * uvij;
					}
				}
			}
			else { //DENSE
				DenseBlock x = mX.getDenseBlock();
				for( int i=rl; i<ru; i++ ) {
					double[] xvals = x.values(i);
					int xix = x.pos(i);
					for( int j=0; j<n; j++)
						if( xvals[xix+j] != 0 ) {
							double xij = xvals[xix+j];
							double uvij = dotProductGeneric(mU, mV, i, j, cd);
							wsloss += xij * xij - 2 * xij * uvij;
						}
				}
			}
		}

		ret.set(0, 0, wsloss);
	}
	
	private static void addMatrixMultWSLossNoWeightCorrection(MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, int k) {
		MatrixBlock tmp1 = new MatrixBlock(mU.clen, mU.clen, false);
		MatrixBlock tmp2 = new MatrixBlock(mU.clen, mU.clen, false);
		matrixMultTransposeSelf(mU, tmp1, true, k);
		matrixMultTransposeSelf(mV, tmp2, true, k);
		ret.set(0, 0, ret.get(0, 0) + 
			((tmp1.sparse || tmp2.sparse) ? dotProductGeneric(tmp1, tmp2) :
			dotProduct(tmp1.getDenseBlockValues(), tmp2.getDenseBlockValues(), mU.clen*mU.clen)));
	}

	private static void matrixMultWSigmoidDenseNative(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt) {
		double[] w = mW.getDenseBlockValues();
		double[] c = ret.getDenseBlockValues();
		final int m = mW.rlen, n = mW.clen;
		final int cd = mU.clen;
		boolean flagminus = (wt==WSigmoidType.MINUS || wt==WSigmoidType.LOG_MINUS); 
		boolean flaglog = (wt==WSigmoidType.LOG || wt==WSigmoidType.LOG_MINUS);
		
		//note: experiments with a fully native implementation of this method (even with #pragma omp simd)
		//showed performance regressions compared to this version because we benefit from FastMath.exp 
		
		//call native matrix multiplication (only called for single-threaded and matrix-vector
		//because this ensures that we can deal with the transpose mV without additional transpose)
		long nnz =NativeHelper.dmmdd(((m==1)?mV:mU).getDenseBlockValues(),
			((m==1)?mU:mV).getDenseBlockValues(), c, (m==1)?n:m, cd, 1, 1);
		if(nnz < 0) {
			//fallback to default java implementation
			LOG.warn("matrixMultWSigmoidDenseNative: Native mat mult failed. Falling back to java version.");
			matrixMult(((m==1)?mV:mU), ((m==1)?mU:mV), ret, false);
		}
		
		//compute remaining wsigmoid for all relevant outputs
		for( int i=0; i<m*n; i++ ) {
			//compute core sigmoid function
			double cval = flagminus ?
				1 / (1 + FastMath.exp(c[i])) :
				1 / (1 + FastMath.exp(-c[i]));
			//compute weighted output
			c[i] = w[i] * ((flaglog) ? Math.log(cval) : cval);
		}
	}
	
	private static void matrixMultWSigmoidDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) {
		DenseBlock w = mW.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
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
		for( int bi = rl; bi < ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			for( int bj = 0; bj < n; bj+=blocksizeIJ ) {
				int bjmin = Math.min(n, bj+blocksizeIJ);
				//core wsigmoid computation
				for( int i=bi; i<bimin; i++ ) {
					double[] wvals = w.values(i), uvals = u.values(i), cvals = c.values(i);
					int wix = w.pos(i), uix = u.pos(i);
					for( int j=bj; j<bjmin; j++) {
						double wij = wvals[wix+j];
						if( wij != 0 )
							cvals[wix+j] = wsigmoid(wij, uvals, v.values(j),
								uix, v.pos(j), flagminus, flaglog, cd);
					}
				}
			}
		}
	}

	private static void matrixMultWSigmoidSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) {
		SparseBlock w = mW.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		final int cd = mU.clen;
		
		boolean flagminus = (wt==WSigmoidType.MINUS || wt==WSigmoidType.LOG_MINUS);
		boolean flaglog = (wt==WSigmoidType.LOG || wt==WSigmoidType.LOG_MINUS);
	
		//approach: iterate over non-zeros of w, selective mm computation
		for( int i=rl; i<ru; i++ ) {
			if( w.isEmpty(i) ) continue;
			int wpos = w.pos(i);
			int wlen = w.size(i);
			int[] wix = w.indexes(i);
			double[] wval = w.values(i);
			double[] uvals = u.values(i);
			int uix = u.pos(i);
			c.allocate(i, wlen);
			for( int k=wpos; k<wpos+wlen; k++ ) {
				double cval = wsigmoid(wval[k], uvals, v.values(wix[k]),
					uix, v.pos(wix[k]), flagminus, flaglog, cd);
				c.append(i, wix[k], cval);
			}
		}
	}

	private static void matrixMultWSigmoidGeneric (MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) {
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
			for( int i=rl; i<ru; i++ ) {
				if( w.isEmpty(i) ) continue;
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
			DenseBlock w = mW.getDenseBlock();
			DenseBlock c = ret.getDenseBlock();
			for( int i=rl; i<ru; i++ ) {
				double[] wvals = w.values(i), cvals = c.values(i);
				int ix = w.pos(i);
				for( int j=0; j<n; j++ ) {
					double wij = wvals[ix+j];
					if( wij != 0 )
						cvals[ix+j] = wsigmoid(wij, mU, mV, i, j, flagminus, flaglog, cd);
				}
			}
		}
	}

	private static void matrixMultWDivMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) {
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft();
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final double eps = scalar ? mX.get(0, 0) : 0;
		final int cd = mU.clen;
		
		DenseBlock w = mW.getDenseBlock();
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		DenseBlock x = (mX==null) ? null : mX.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		
		//approach: iterate over non-zeros of w, selective mm computation
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 
		
		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			for( int bj = cl; bj < cu; bj+=blocksizeIJ ) {
				int bjmin = Math.min(cu, bj+blocksizeIJ);
				//core wsigmoid computation
				for( int i=bi; i<bimin; i++ ) {
					double[] wvals = w.values(i), uvals = u.values(i);
					double[] xvals = four ? x.values(i) : null; 
					int wix = w.pos(i), uix = u.pos(i);
					for( int j=bj; j<bjmin; j++ )
						if( wvals[wix+j] != 0 ) {
							double[] cvals = c.values((basic||!left) ? i : j);
							if( basic ) 
								cvals[wix+j] = wvals[wix+j] * dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
							else if( four ) { //left/right 
								if (scalar)
									wdivmm(wvals[wix+j], eps, uvals, v.values(j), cvals, uix, v.pos(j), left, scalar, cd);
								else
									wdivmm(wvals[wix+j], xvals[wix+j], uvals, v.values(j), cvals, uix, v.pos(j), left, scalar, cd);
							}
							else //left/right minus/default
								wdivmm(wvals[wix+j], uvals, v.values(j), cvals, uix, v.pos(j), left, mult, minus, cd);
						}
				}
			}
		}
	}

	private static void matrixMultWDivMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) {
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft();
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final double eps = scalar ? mX.get(0, 0) : 0;
		final int cd = mU.clen;
		
		SparseBlock w = mW.sparseBlock;
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		SparseBlock x = (mX==null) ? null : mX.sparseBlock;
		
		//approach: iterate over non-zeros of w, selective mm computation
		//blocked over ij, while maintaining front of column indexes, where the
		//blocksize is chosen such that we reuse each  Ui/Vj vector on average 8 times,
		//with custom blocksizeJ for wdivmm_left to avoid LLC misses on output.
		final int blocksizeI = (int) (8L*mW.rlen*mW.clen/mW.nonZeros);
		final int blocksizeJ = left ? Math.max(8,Math.min(L2_CACHESIZE/(mU.clen*8), blocksizeI)) : blocksizeI;
		
		int[] curk = new int[blocksizeI];
		boolean[] aligned = (four&&!scalar) ? new boolean[blocksizeI] : null;
		
		//blocked execution over row blocks
		for( int bi=rl; bi<ru; bi+=blocksizeI ) 
		{
			int bimin = Math.min(ru, bi+blocksizeI);
			//prepare starting indexes for block row
			for( int i=bi; i<bimin; i++ ) {
				int k = (cl==0||w.isEmpty(i)) ? 0 : w.posFIndexGTE(i,cl);
				curk[i-bi] = (k>=0) ? k : mW.clen;
			}
			//prepare alignment info if necessary
			if( four && !scalar )
				for( int i=bi; i<bimin; i++ )
					aligned[i-bi] = w.isAligned(i-bi, x);
			
			//blocked execution over column blocks
			for( int bj=cl; bj<cu; bj+=blocksizeJ )  
			{
				int bjmin = Math.min(cu, bj+blocksizeJ);
				//core wdivmm block matrix mult
				for( int i=bi; i<bimin; i++ ) {
					if( w.isEmpty(i) ) continue;
					
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wval = w.values(i);
					double[] uvals = u.values(i);
					int uix = u.pos(i);
					
					int k = wpos + curk[i-bi];
					if( basic ) {
						for( ; k<wpos+wlen && wix[k]<bjmin; k++ )
							ret.appendValue( i, wix[k], wval[k] *dotProduct(
								uvals, v.values(wix[k]), uix, v.pos(wix[k]), cd));
					}
					else if( four ) { //left/right
						//checking alignment per row is ok because early abort if false, 
						//row nnz likely fit in L1/L2 cache, and asymptotically better if aligned
						if( !scalar && w.isAligned(i, x) ) {
							//O(n) where n is nnz in w/x 
							double[] xvals = x.values(i);
							for( ; k<wpos+wlen && wix[k]<bjmin; k++ ) {
								double[] cvals = c.values(left ? wix[k] : i);
								wdivmm(wval[k], xvals[k], uvals, v.values(wix[k]),
									cvals, uix, v.pos(wix[k]), left, scalar, cd);
							}
						}
						else {
							//scalar or O(n log m) where n/m are nnz in w/x
							for( ; k<wpos+wlen && wix[k]<bjmin; k++ ) {
								double[] cvals = c.values(left ? wix[k] : i);
								if (scalar)
									wdivmm(wval[k], eps, uvals, v.values(wix[k]),
										cvals, uix, v.pos(wix[k]), left, scalar, cd);
								else
									wdivmm(wval[k], x.get(i, wix[k]), uvals,
										v.values(wix[k]), cvals, uix, v.pos(wix[k]), left, scalar, cd);
							}
						}
					}
					else { //left/right minus default
						for( ; k<wpos+wlen && wix[k]<bjmin; k++ ) {
							double[] cvals = c.values(left ? wix[k] : i);
							wdivmm(wval[k], uvals, v.values(wix[k]), cvals,
								uix, v.pos(wix[k]), left, mult, minus, cd);
						}
					}
					curk[i-bi] = k - wpos;
				}
			}
		}
	}

	private static void matrixMultWDivMMGeneric(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) {
		final boolean basic = wt.isBasic();
		final boolean left = wt.isLeft(); 
		final boolean mult = wt.isMult();
		final boolean minus = wt.isMinus();
		final boolean four = wt.hasFourInputs();
		final boolean scalar = wt.hasScalar();
		final double eps = scalar ? mX.get(0, 0) : 0;
		final int cd = mU.clen;

		//output always in dense representation
		DenseBlock c = ret.getDenseBlock();
		
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			SparseBlock w = mW.sparseBlock;
			
			for( int i=rl; i<ru; i++ ) {
				if( w.isEmpty(i) ) continue;
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				int k = (cl==0) ? 0 : w.posFIndexGTE(i,cl);
				k = (k>=0) ? wpos+k : wpos+wlen;
				for( ; k<wpos+wlen && wix[k]<cu; k++ ) {
					double[] cvals = c.values((basic||!left) ? i : wix[k]);
					if( basic ) {
						double uvij = dotProductGeneric(mU,mV, i, wix[k], cd);
						ret.appendValue(i, wix[k], uvij);
					}
					else if( four ) { //left/right
						double xij = scalar ? eps : mX.get(i, wix[k]);
						wdivmm(wval[k], xij, mU, mV, cvals, i, wix[k], left, scalar, cd);
					}
					else { //left/right minus/default
						wdivmm(wval[k], mU, mV, cvals, i, wix[k], left, mult, minus, cd);
					}
				}
			}
		}
		else //DENSE
		{
			DenseBlock w = mW.getDenseBlock();
			for( int i=rl; i<ru; i++ ) {
				double[] wvals = w.values(i);
				int ix = w.pos(i);
				for( int j=cl; j<cu; j++)
					if( wvals[ix+j] != 0 ) {
						double[] cvals = c.values((basic||!left) ? i : j);
						if( basic ) {
							cvals[ix+j] = dotProductGeneric(mU,mV, i, j, cd);
						}
						else if( four ) { //left/right
							double xij = scalar ? eps : mX.get(i, j);
							wdivmm(wvals[ix+j], xij, mU, mV, cvals, i, j, left, scalar, cd);
						}
						else { //left/right minus/default
							wdivmm(wvals[ix+j], mU, mV, cvals, i, j, left, mult, minus, cd);
						}
					}
			}
		}
	}

	private static void matrixMultWCeMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		DenseBlock w = mW.getDenseBlock();
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		final int n = mW.clen;
		final int cd = mU.clen;
		double wceval = 0;
		
		// approach: iterate over all cells of X 
		//cache-conscious blocking: due to blocksize constraint (default 1000),
		//a blocksize of 16 allows to fit blocks of UV into L2 cache (256KB) 
		final int blocksizeIJ = 16; //u/v block (max at typical L2 size) 

		//blocked execution
		for( int bi = rl; bi < ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			for( int bj = 0; bj < n; bj+=blocksizeIJ ) {
				int bjmin = Math.min(n, bj+blocksizeIJ);
				for( int i=bi; i<bimin; i++ ) {
					double[] wvals = w.values(i), uvals = u.values(i);
					int wix = w.pos(i), uix = u.pos(i);
					for( int j=bj; j<bjmin; j++ ) {
						double wij = wvals[wix+j];
						if( wij != 0 ) {
							double uvij = dotProduct(uvals, v.values(j), uix, v.pos(j), cd);
							wceval += wij * Math.log(uvij + eps);
						}
					}
				}
			}
		}
		ret.set(0, 0, wceval);
	}

	private static void matrixMultWCeMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		SparseBlock w = mW.sparseBlock;
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		final int n = mW.clen;
		final int cd = mU.clen;
		double wceval = 0;
		
		// approach: iterate over W, point-wise in order to exploit sparsity
		// blocked over ij, while maintaining front of column indexes, where the
		// blocksize is chosen such that we reuse each vector on average 8 times.
		final int blocksizeIJ = (int) (8L*mW.rlen*mW.clen/mW.nonZeros); 
		int[] curk = new int[blocksizeIJ];
		
		for( int bi=rl; bi<ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			//prepare starting indexes for block row
			Arrays.fill(curk, 0); 
			//blocked execution over column blocks
			for( int bj=0; bj<n; bj+=blocksizeIJ ) {
				int bjmin = Math.min(n, bj+blocksizeIJ);
				for( int i=bi; i<bimin; i++ ) {
					if( w.isEmpty(i) ) continue;
					int wpos = w.pos(i);
					int wlen = w.size(i);
					int[] wix = w.indexes(i);
					double[] wvals = w.values(i);
					double[] uvals = u.values(i);
					int uix = u.pos(i);
					int k = wpos + curk[i-bi];
					for( ; k<wpos+wlen && wix[k]<bjmin; k++ ) {
						double uvij = dotProduct(uvals, v.values(wix[k]), uix, v.pos(wix[k]), cd);
						wceval += wvals[k] * Math.log(uvij + eps);
					}
					curk[i-bi] = k - wpos;
				}
			}
		}
		ret.set(0, 0, wceval);
	}

	private static void matrixMultWCeMMGeneric(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, MatrixBlock ret, WCeMMType wt, int rl, int ru)
	{
		final int n = mW.clen; 
		final int cd = mU.clen;
		double wceval = 0; 

		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) //SPARSE
		{
			SparseBlock w = mW.sparseBlock;
			for( int i=rl; i<ru; i++ ) {
				if( w.isEmpty(i) ) continue;
				int wpos = w.pos(i);
				int wlen = w.size(i);
				int[] wix = w.indexes(i);
				double[] wval = w.values(i);
				for( int k=wpos; k<wpos+wlen; k++ ) {
					double uvij = dotProductGeneric(mU, mV, i, wix[k], cd);
					wceval += wval[k] * Math.log(uvij + eps);
				}
			}
		}
		else //DENSE
		{
			DenseBlock w = mW.getDenseBlock();
			for( int i=rl; i<ru; i++ ) {
				double[] wvals = w.values(i);
				int wix = w.pos(i);
				for( int j=0; j<n; j++ ) {
					double wij = wvals[wix+j];
					if( wij != 0 ) {
						double uvij = dotProductGeneric(mU, mV, i, j, cd);
						wceval += wij * Math.log(uvij + eps);
					}
				}
			}
		}

		ret.set(0, 0, wceval);
	}

	private static void matrixMultWuMMDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) {
		DenseBlock w = mW.getDenseBlock();
		DenseBlock c = ret.getDenseBlock();
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
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
		for( int bi = rl; bi < ru; bi+=blocksizeIJ ) {
			int bimin = Math.min(ru, bi+blocksizeIJ);
			for( int bj = 0; bj < n; bj+=blocksizeIJ ) {
				int bjmin = Math.min(n, bj+blocksizeIJ);
				//core wsigmoid computation
				for( int i=bi; i<bimin; i++ ) {
					double[] wvals = w.values(i), uvals = u.values(i), cvals = c.values(i);
					int wix = w.pos(i), uix = u.pos(i);
					for( int j=bj; j<bjmin; j++ ) {
						double wij = wvals[wix+j];
						if( wij != 0 )
							cvals[wix+j] = wumm(wij, uvals, v.values(j),
								uix, v.pos(j), flagmult, fn, cd);
					}
				}
			}
		}
	}

	private static void matrixMultWuMMSparseDense(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) {
		SparseBlock w = mW.sparseBlock;
		SparseBlock c = ret.sparseBlock;
		DenseBlock u = mU.getDenseBlock();
		DenseBlock v = mV.getDenseBlock();
		final int cd = mU.clen;
		boolean flagmult = (wt==WUMMType.MULT);
		
		//approach: iterate over non-zeros of w, selective mm computation
		for( int i=rl; i<ru; i++ ) {
			if( w.isEmpty(i) ) continue;
			int wpos = w.pos(i);
			int wlen = w.size(i);
			int[] wix = w.indexes(i);
			double[] wvals = w.values(i);
			double[] uvals = u.values(i);
			int uix = u.pos(i);
			c.allocate(i, wlen);
			for( int k=wpos; k<wpos+wlen; k++ ) {
				double cval = wumm(wvals[k], uvals, v.values(wix[k]),
					uix, v.pos(wix[k]), flagmult, fn, cd);
				c.append(i, wix[k], cval);
			}
		}
	}

	private static void matrixMultWuMMGeneric (MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) {
		final int n = mW.clen;
		final int cd = mU.clen;
		boolean flagmult = (wt==WUMMType.MULT);
		
		//approach: iterate over non-zeros of w, selective mm computation
		if( mW.sparse ) { //SPARSE
			//w and c always in same representation
			SparseBlock w = mW.sparseBlock;
			SparseBlock c = ret.sparseBlock;
			for( int i=rl; i<ru; i++ ) {
				if( w.isEmpty(i) ) continue;
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
		else { //DENSE
			//w and c always in same representation
			DenseBlock w = mW.getDenseBlock();
			DenseBlock c = ret.getDenseBlock();
			for( int i=rl; i<ru; i++ ) {
				double[] wvals = w.values(i), cvals = c.values(i);
				int ix = w.pos(i);
				for( int j=0; j<n; j++) {
					double wij = wvals[ix+j];
					if( wij != 0 )
						cvals[ix+j] = wumm(wij, mU, mV, i, j, flagmult, fn, cd);
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
	 * @param a first vector
	 * @param b second vector
	 * @param len length
	 * @return dot product of the two input vectors
	 */
	private static double dotProduct( double[] a, double[] b, final int len )
	{
		double val = 0;

		final int bn = len%vLen;
				
		//compute rest
		for( int i = 0; i < bn; i++ )
			val += a[ i ] * b[ i ];
		
		//unrolled vLen-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=vLen ){
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, i);
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, i);
			val += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);
		}
		
		//scalar result
		return val; 
	}

	//note: public for use by codegen for consistency
	public static double dotProduct( double[] a, double[] b, int ai, int bi, final int len )
	{
		double val = 0;
		final int bn = len%vLen;
		
		//compute rest
		for( int i = 0; i < bn; i++, ai++, bi++ )
			val += a[ ai ] * b[ bi ];
		
		//unrolled vLen-block (for better instruction-level parallelism)
		for( int i = bn; i < len; i+=vLen, ai+=vLen, bi+=vLen )
		{
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			val += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);
		}
		
		//scalar result
		return val; 
	}
	
	//note: public for use by codegen for consistency
	public static double dotProduct( double[] a, double[] b, int[] aix, int ai, final int bi, final int len )
	{
		double val = 0;
		final int bn = len%vLen;
				
		//compute rest
		for( int i = ai; i < ai+bn; i++ )
			val += a[ i ] * b[ bi+aix[i] ];
		
		//unrolled vLen-block (for better instruction-level parallelism)
		for( int i = ai+bn; i < ai+len; i+=vLen)
		{
			//read 64B cacheline of a
			//read 64B of b via 'gather'
			//compute cval' = sum(a * b) + cval
			var aVec = DoubleVector.fromArray(SPECIES, a, i);
			var bVec = DoubleVector.fromArray(SPECIES, b, bi, aix, i);
			val += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);

		}
		
		//scalar result
		return val; 
	}
	
	private static double dotProduct(double[] a, int[] aix, final int apos, final int alen, double[] b, int bix[], final int bpos, final int blen) {
		final int asize = apos+alen;
		final int bsize = bpos+blen;
		int k = apos, k2 = bpos;
		
		//pruning filter
		if(aix[apos]>bix[bsize-1] || aix[asize-1]<bix[bpos] )
			return 0;
		
		//sorted set intersection
		double v = 0;
		while( k<asize & k2<bsize ) {
			int aixk = aix[k];
			int bixk = bix[k2];
			if( aixk < bixk )
				k++;
			else if( aixk > bixk )
				k2++;
			else { // ===
				v += a[k] * b[k2];
				k++; k2++;
			}
			//note: branchless version slower
			//v += (aixk==bixk) ? a[k] * b[k2] : 0;
			//k += (aixk <= bixk) ? 1 : 0;
			//k2 += (aixk >= bixk) ? 1 : 0;
		}
		return v;
	}

	//note: public for use by codegen for consistency
	public static void vectMultiplyAdd(final double aval, double[] b, double[] c, int bi, int ci, final int len) {
		final int bn = len%vLen;
		
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, bi++, ci++)
			c[ ci ] += aval * b[ bi ];
		
		DoubleVector aVec = DoubleVector.broadcast(SPECIES, aval);
		//unrolled vLen-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, bi+=vLen, ci+=vLen) 
		{
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			cVec = aVec.fma(bVec, cVec);
			cVec.intoArray(c, ci);
		}
	}

	private static void vectMultiplyAdd2( final double aval1, final double aval2, double[] b, double[] c, int bi1, int bi2, int ci, final int len )
	{
		final int bn = len%vLen;
		
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ];
		
		DoubleVector aVec1 = DoubleVector.broadcast(SPECIES, aval1);
		DoubleVector aVec2 = DoubleVector.broadcast(SPECIES, aval2);
		//unrolled vLen-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, bi1+=vLen, bi2+=vLen, ci+=vLen ) 		{
			DoubleVector bVec1 = DoubleVector.fromArray(SPECIES, b, bi1);
			DoubleVector bVec2 = DoubleVector.fromArray(SPECIES, b, bi2);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			cVec = aVec1.fma(bVec1, cVec);
			cVec = aVec2.fma(bVec2, cVec);
			cVec.intoArray(c, ci);
		}
	}

	private static void vectMultiplyAdd3( final double aval1, final double aval2, final double aval3, double[] b, double[] c, int bi1, int bi2, int bi3, int ci, final int len )
	{
		final int bn = len%vLen;
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, bi3++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ] + aval3 * b[ bi3 ];
		
		DoubleVector aVec1 = DoubleVector.broadcast(SPECIES, aval1);
		DoubleVector aVec2 = DoubleVector.broadcast(SPECIES, aval2);
		DoubleVector aVec3 = DoubleVector.broadcast(SPECIES, aval3);
		//unrolled vLen-block (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, bi1+=vLen, bi2+=vLen, bi3+=vLen, ci+=vLen ) 
		{	
			DoubleVector bVec1 = DoubleVector.fromArray(SPECIES, b, bi1);
			DoubleVector bVec2 = DoubleVector.fromArray(SPECIES, b, bi2);
			DoubleVector bVec3 = DoubleVector.fromArray(SPECIES, b, bi3);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			cVec = aVec1.fma(bVec1, cVec);
			cVec = aVec2.fma(bVec2, cVec);
			cVec = aVec3.fma(bVec3, cVec);
			cVec.intoArray(c, ci);
		}
	}

	private static void vectMultiplyAdd4( final double aval1, final double aval2, final double aval3, final double aval4, double[] b, double[] c, int bi1, int bi2, int bi3, int bi4, int ci, final int len )
	{
		final int bn = len%vLen;
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, bi1++, bi2++, bi3++, bi4++, ci++ )
			c[ ci ] += aval1 * b[ bi1 ] + aval2 * b[ bi2 ] + aval3 * b[ bi3 ] + aval4 * b[ bi4 ];
		
		DoubleVector aVec1 = DoubleVector.broadcast(SPECIES, aval1);
		DoubleVector aVec2 = DoubleVector.broadcast(SPECIES, aval2);
		DoubleVector aVec3 = DoubleVector.broadcast(SPECIES, aval3);
		DoubleVector aVec4 = DoubleVector.broadcast(SPECIES, aval4);
		//unrolled vLen-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, bi1+=vLen, bi2+=vLen, bi3+=vLen, bi4+=vLen, ci+=vLen) 
		{
			DoubleVector bVec1 = DoubleVector.fromArray(SPECIES, b, bi1);
			DoubleVector bVec2 = DoubleVector.fromArray(SPECIES, b, bi2);
			DoubleVector bVec3 = DoubleVector.fromArray(SPECIES, b, bi3);
			DoubleVector bVec4 = DoubleVector.fromArray(SPECIES, b, bi4);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci);
			cVec = aVec1.fma(bVec1, cVec);
			cVec = aVec2.fma(bVec2, cVec);
			cVec = aVec3.fma(bVec3, cVec);
			cVec = aVec4.fma(bVec4, cVec);
			cVec.intoArray(c, ci);
		}
	}
	
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

	//note: public for use by codegen for consistency
	public static void vectMultiplyAdd( final double aval, double[] b, double[] c, int[] bix, final int bi, final int ci, final int len )
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

	//note: public for use by codegen for consistency
	public static void vectMultiplyWrite( final double aval, double[] b, double[] c, int bi, int ci, final int len )
	{
		final int bn = len%vLen;
		
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, bi++, ci++)
			c[ ci ] = aval * b[ bi ];
		
		//unrolled vLen-block (for better instruction-level parallelism)
		DoubleVector aVec = DoubleVector.broadcast(SPECIES, aval);
		for( int j = bn; j < len; j+=vLen, bi+=vLen, ci+=vLen) 
		{
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			aVec.mul(bVec).intoArray(c, ci);
		}
	}
	
	public static void vectMultiplyInPlace( final double aval, double[] c, int ci, final int len ) {
		final int bn = len%8;
		//rest, not aligned to 8-blocks
		for( int j = 0; j < bn; j++, ci++)
			c[ ci ] *= aval;
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=8, ci+=8) {
			c[ ci+0 ] *= aval; c[ ci+1 ] *= aval;
			c[ ci+2 ] *= aval; c[ ci+3 ] *= aval;
			c[ ci+4 ] *= aval; c[ ci+5 ] *= aval;
			c[ ci+6 ] *= aval; c[ ci+7 ] *= aval;
		}
	}
	
	public static void vectMultiplyInPlace(final double[] a, double[] c, int[] cix, final int ai, final int ci, final int len) {
		final int bn = len%8;
		//rest, not aligned to 8-blocks
		for( int j = ci; j < ci+bn; j++ )
			c[ j ] *= a[ ai+cix[j] ];
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = ci+bn; j < ci+len; j+=8 ) {
			c[ j+0 ] *= a[ ai+cix[j+0] ];
			c[ j+1 ] *= a[ ai+cix[j+1] ];
			c[ j+2 ] *= a[ ai+cix[j+2] ];
			c[ j+3 ] *= a[ ai+cix[j+3] ];
			c[ j+4 ] *= a[ ai+cix[j+4] ];
			c[ j+5 ] *= a[ ai+cix[j+5] ];
			c[ j+6 ] *= a[ ai+cix[j+6] ];
			c[ j+7 ] *= a[ ai+cix[j+7] ];
		}
	}

	//note: public for use by codegen for consistency
	public static void vectMultiplyWrite( double[] a, double[] b, double[] c, int ai, int bi, int ci, final int len ){

		final int bn = len%vLen;
		
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, ai++, bi++, ci++)
			c[ ci ] = a[ ai ] * b[ bi ];
		
		//unrolled vLen-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, ai+=vLen, bi+=vLen, ci+=vLen) 
		{
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi);
			aVec.mul(bVec).intoArray(c, ci);
		}
	}
	
	public static void vectMultiplyWrite( final double[] a, double[] b, double[] c, int[] bix, final int ai, final int bi, final int ci, final int len ) {
		final int bn = len%8;
		//rest, not aligned to 8-blocks
		for( int j = bi; j < bi+bn; j++ )
			c[ ci+bix[j] ] = a[ ai+bix[j] ] * b[ j ];
		//unrolled 8-block (for better instruction-level parallelism)
		for( int j = bi+bn; j < bi+len; j+=8 ) {
			c[ ci+bix[j+0] ] = a[ ai+bix[j+0] ] * b[ j+0 ];
			c[ ci+bix[j+1] ] = a[ ai+bix[j+1] ] * b[ j+1 ];
			c[ ci+bix[j+2] ] = a[ ai+bix[j+2] ] * b[ j+2 ];
			c[ ci+bix[j+3] ] = a[ ai+bix[j+3] ] * b[ j+3 ];
			c[ ci+bix[j+4] ] = a[ ai+bix[j+4] ] * b[ j+4 ];
			c[ ci+bix[j+5] ] = a[ ai+bix[j+5] ] * b[ j+5 ];
			c[ ci+bix[j+6] ] = a[ ai+bix[j+6] ] * b[ j+6 ];
			c[ ci+bix[j+7] ] = a[ ai+bix[j+7] ] * b[ j+7 ];
		}
	}

	public static void vectMultiply(double[] a, double[] c, int ai, int ci, final int len){

		final int bn = len%vLen;
		
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] *= a[ ai ];
		
		//unrolled vLen-block  (for better instruction-level parallelism)
		for( int j = bn; j < len; j+=vLen, ai+=vLen, ci+=vLen) 
		{
			DoubleVector res = DoubleVector.fromArray(SPECIES, c, ci);
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			
			res = aVec.mul(res);
			res.intoArray(c, ci);
		}
	}

	//note: public for use by codegen for consistency
	public static void vectAdd( double[] a, double bval, double[] c, int ai, int ci, final int len ) {
		final int bn = len%vLen;
		//rest, not aligned to vLen-blocks
		for( int j = 0; j < bn; j++, ai++, ci++)
			c[ ci ] += a[ ai ] + bval;

		//unrolled vLen-block  (for better ILP)
		DoubleVector bVec = DoubleVector.broadcast(SPECIES, bval);
		for( int j = bn; j < len; j+=vLen, ai+=vLen, ci+=vLen) {
			DoubleVector res = DoubleVector.fromArray(SPECIES, c, ci);
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai);
			res = aVec.add(bVec).add(res);
			res.intoArray(c, ci);
		}
	}
	
	//note: public for use by codegen for consistency
	public static void vectAdd( double[] a, double[] c, int ai, int ci, final int len )
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

	public static void vectAdd( double[] a, double[] c, int[] aix, int ai, int ci, final int alen ) {
		final int bn = alen%8;
		//rest, not aligned to 8-blocks
		for( int j = ai; j < ai+bn; j++ )
			c[ ci+aix[j] ] += a[ j ];
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = ai+bn; j < ai+alen; j+=8 ) {
			c[ ci+aix[j+0] ] += a[ j+0 ];
			c[ ci+aix[j+1] ] += a[ j+1 ];
			c[ ci+aix[j+2] ] += a[ j+2 ];
			c[ ci+aix[j+3] ] += a[ j+3 ];
			c[ ci+aix[j+4] ] += a[ j+4 ];
			c[ ci+aix[j+5] ] += a[ j+5 ];
			c[ ci+aix[j+6] ] += a[ j+6 ];
			c[ ci+aix[j+7] ] += a[ j+7 ];
		}
	}
	
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
	
	private static void vectAddAll(double[][] a, double[] c, int ai, int ci, final int len) {
		int bi = a.length % 4;
		//process stride for remaining blocks
		for(int i=0; i<bi; i++)
			vectAdd(a[i], c, ai, ci, len);
		//process stride in 4 blocks at a time
		for(int i=bi; i<a.length; i+=4)
			vectAdd4(a[i], a[i+1], a[i+2], a[i+3], c, ai, ci, len);
	}
	
	public static void vectAddInPlace(double aval, double[] c, final int ci, final int len) {
		final int bn = len%8;
		//rest, not aligned to 8-blocks
		for( int j = ci; j < ci+bn; j++)
			c[ j ] += aval;
		//unrolled 8-block  (for better instruction-level parallelism)
		for( int j = ci+bn; j < ci+len; j+=8) {
			c[ j+0 ] += aval; c[ j+1 ] += aval; 
			c[ j+2 ] += aval; c[ j+3 ] += aval;
			c[ j+4 ] += aval; c[ j+5 ] += aval;
			c[ j+6 ] += aval; c[ j+7 ] += aval;
		}
	}

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

	private static double wsigmoid( final double wij, double[] u, double[] v, final int uix, final int vix, final boolean flagminus, final boolean flaglog, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute core sigmoid function  
		double cval = flagminus ?
				1 / (1 + FastMath.exp(uvij)) :
				1 / (1 + FastMath.exp(-uvij));
				
		//compute weighted output
		return wij * ((flaglog) ? Math.log(cval) : cval);
	}

	private static double wsigmoid( final double wij, MatrixBlock u, MatrixBlock v, final int uix, final int vix, final boolean flagminus, final boolean flaglog, final int len )
	{
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);
		
		//compute core sigmoid function  
		double cval = flagminus ?
				1 / (1 + FastMath.exp(uvij)) :
				1 / (1 + FastMath.exp(-uvij));
				
		//compute weighted output
		return wij * ((flaglog) ? Math.log(cval) : cval);
	}
	
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
			c[cix+k2] += b.get(bix, k2) * wtmp;
	}

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
			c[cix+k2] += b.get(bix, k2) * wtmp;
	}

	private static double wumm( final double wij, double[] u, double[] v, final int uix, final int vix, final boolean flagmult, ValueFunction fn, final int len ) {
		//compute dot product over ui vj 
		double uvij = dotProduct(u, v, uix, vix, len);
		
		//compute unary operations
		double cval = fn.execute(uvij);
		
		//compute weighted output
		return flagmult ? wij * cval : wij / cval;
	}

	private static double wumm( final double wij, MatrixBlock u, MatrixBlock v, final int uix, final int vix, final boolean flagmult, ValueFunction fn, final int len ) {
		//compute dot product over ui vj 
		double uvij = dotProductGeneric(u, v, uix, vix, len);

		//compute unary operations
		double cval = fn.execute(uvij);
		
		//compute weighted output
		return flagmult ? wij * cval : wij / cval;
	}

	private static double dotProductGeneric(MatrixBlock a, MatrixBlock b, final int ai, final int bi, int len)
	{
		double val = 0;
		for( int k2=0; k2<len; k2++ )
			val += a.get(ai, k2) * b.get(bi, k2);
		
		return val;
	}
	
	private static double dotProductGeneric(MatrixBlock a, MatrixBlock b)
	{
		double val = 0;
		for( int i=0; i<a.getNumRows(); i++ )
			for( int j=0; j<a.getNumColumns(); j++ )
				val += a.get(i, j) * b.get(i, j);
		
		return val;
	}
	
	public static long copyUpperToLowerTriangle( MatrixBlock ret ) {
		return ret.sparse ?
			copyUpperToLowerTriangleSparse(ret) :
			copyUpperToLowerTriangleDense(ret);
	}
	
	
	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 * 
	 * @param ret matrix
	 * @return number of non zeros
	 */
	public static long copyUpperToLowerTriangleDense( MatrixBlock ret )
	{
		//ret is guaranteed to be a squared, symmetric matrix
		if( ret.rlen != ret.clen )
			throw new RuntimeException("Invalid non-squared input matrix.");
		
		final double[] c = ret.getDenseBlockValues();
		final int n = ret.rlen;
		long nnz = 0;
		
		//blocked execution (2x128KB for L2 blocking)
		final int blocksizeIJ = 128; 
		
		//handle blocks on diagonal
		for( int bi = 0; bi<n; bi+=blocksizeIJ ) {
			int bimin = Math.min(bi+blocksizeIJ, n);
			for( int i=bi, rix=bi*n; i<bimin; i++, rix+=n ) {
				LibMatrixReorg.transposeRow(c, c, rix+bi, bi*n+i, n, bimin-bi);
				nnz += (c[rix+i] != 0) ? 1 : 0; //for diagonal element
				for( int j=rix+i+1; j<rix+bimin; j++ )
					nnz += (c[j] != 0) ? 2 : 0;
			}
		}
		
		//handle non-diagonal blocks (full block copies)
		for( int bi = 0; bi<n; bi+=blocksizeIJ ) {
			int bimin = Math.min(bi+blocksizeIJ, n);
			for( int bj = bi; bj<n; bj+=blocksizeIJ ) 
				if( bi != bj ) { //not on diagonal
					int bjmin = Math.min(bj+blocksizeIJ, n);
					for( int i=bi, rix=bi*n; i<bimin; i++, rix+=n ) {
						LibMatrixReorg.transposeRow(c, c, rix+bj, bj*n+i, n, bjmin-bj);
						for( int j=rix+bj; j<rix+bjmin; j++ )
							nnz += (c[j] != 0) ? 2 : 0;
					}
				}
		}
		
		return nnz;
	}

	public static long copyUpperToLowerTriangleSparse( MatrixBlock ret )
	{
		//ret is guaranteed to be a squared, symmetric matrix
		if( ret.rlen != ret.clen )
			throw new RuntimeException("Invalid non-squared input matrix.");
		
		SparseBlock c = ret.getSparseBlock();
		int n = ret.rlen;
		long nnz = 0;
		
		//copy non-diagonal values from upper-triangular matrix
		for(int i=0; i<n; i++) {
			if(c.isEmpty(i)) continue;
			int cpos = c.pos(i);
			//int cpos2 = c.posFIndexGTE(i, i);
			//if( cpos2 < 0 ) continue;
			int clen = c.size(i);
			int[] cix = c.indexes(i);
			double[] cvals = c.values(i);
			for(int k=cpos; k<cpos+clen; k++) {
				if( cix[k] == i )
					nnz ++;
				else if( cix[k] > i ) {
					c.append(cix[k], i, cvals[k]);
					nnz += 2;
				}
			}
		}
		
		//sort sparse rows (because append out of order)
		c.sort();
		
		return nnz;
	}
	
	public static MatrixBlock prepMatrixMultTransposeSelfInput( MatrixBlock m1, boolean leftTranspose, boolean par ) {
		MatrixBlock ret = m1;
		final int rlen = m1.rlen;
		final int clen = m1.clen;
		boolean retSparse = isSparseOutputTSMM(m1);
		
		if( !leftTranspose && !retSparse && m1.sparse && rlen > 1) { //X%*%t(X) SPARSE MATRIX
			//directly via LibMatrixReorg in order to prevent sparsity change
			MatrixBlock tmpBlock = new MatrixBlock(clen, rlen, m1.sparse);
			LibMatrixReorg.reorg(m1, tmpBlock, new ReorgOperator(SwapIndex.getSwapIndexFnObject()));
			ret = tmpBlock;
		}
		else if( leftTranspose && !retSparse && m1.sparse && m1.sparseBlock instanceof SparseBlockCSR ) {
			//for a special case of CSR inputs where all non-empty rows are dense, we can
			//create a shallow copy of the values arrays to a "dense" block and perform
			//tsmm with the existing dense block operations w/o unnecessary gather/scatter
			SparseBlockCSR sblock = (SparseBlockCSR)m1.sparseBlock;
			boolean convertDense = (par ?
				IntStream.range(0, rlen).parallel() : IntStream.range(0, rlen))
				.allMatch(i -> sblock.isEmpty(i) || sblock.size(i)==clen );
			if( convertDense ) {
				int rows = (int) sblock.size() / clen;
				MatrixBlock tmpBlock = new MatrixBlock(rows, clen, false);
				tmpBlock.denseBlock = DenseBlockFactory
					.createDenseBlock(sblock.values(), rows, clen);
				tmpBlock.setNonZeros(m1.nonZeros);
				ret = tmpBlock;
			}
		}
		
		return ret;
	}

	private static boolean checkPrepMatrixMultRightInput( MatrixBlock m1, MatrixBlock m2 ) {
		//transpose if dense-dense, skinny rhs matrix (not vector), and memory guarded by output 
		return (!m1.sparse && !m2.sparse 
			&& isSkinnyRightHandSide(m1.rlen, m1.clen, m2.rlen, m2.clen, true));
	}
	
	//note: public for use by codegen for consistency
	public static boolean isSkinnyRightHandSide(long m1rlen, long m1clen, long m2rlen, long m2clen, boolean inclCacheSize) {
		return m1rlen > m2clen && m2rlen > m2clen && m2clen > 1 
			&& m2clen < 64 && (!inclCacheSize || 8*m2rlen*m2clen < L2_CACHESIZE);
	}
	
	private static boolean checkParMatrixMultRightInputRows( MatrixBlock m1, MatrixBlock m2, int k ) {
		//parallelize over rows in rhs matrix if number of rows in lhs/output is very small
		double jvmMem = InfrastructureAnalyzer.getLocalMaxMemory();
		return (m1.rlen==1 && !(m1.sparse && m2.clen==1) && !(m1.isUltraSparse()||m2.isUltraSparse()))
			|| (m1.rlen<=16 && m2.rlen > m1.rlen && (!m1.sparse | m2.clen > 1)
			   && ( !m1.isUltraSparse() && !(m1.sparse & m2.sparse) ) //dense-dense / sparse-dense / dense-sparse
			   && (long)k * 8 * m1.rlen * m2.clen < Math.max(MEM_OVERHEAD_THRESHOLD,0.01*jvmMem) );
	}

	private static boolean checkParMatrixMultRightInputCols( MatrixBlock m1, MatrixBlock m2, int k, boolean pm2r ) {
		//parallelize over cols in rhs matrix if dense, number of cols in rhs is large, and lhs fits in l2
		return (!m1.sparse && !m2.sparse 
				&& m2.clen > k * 1024 && m1.rlen < k * 32 && !pm2r
				&& 8*m1.rlen*m1.clen < 256*1024 ); //lhs fits in L2 cache
	}
	
	public static boolean satisfiesMultiThreadingConstraints(MatrixBlock m1, int k) {
		return satisfiesMultiThreadingConstraints(m1, true, false, -1, k);
	}
	
	public static boolean satisfiesMultiThreadingConstraints(MatrixBlock m1, boolean checkMem, boolean checkFLOPs, long FPfactor, int k) {
		boolean sharedTP = (InfrastructureAnalyzer.getLocalParallelism() == k);
		double jvmMem = InfrastructureAnalyzer.getLocalMaxMemory();
		return k > 1
			&& (!checkMem || 8L * m1.clen * k < Math.max(MEM_OVERHEAD_THRESHOLD,0.01*jvmMem))
			&& (!checkFLOPs || FPfactor * m1.rlen * m1.clen >
			(sharedTP ? PAR_MINFLOP_THRESHOLD2 : PAR_MINFLOP_THRESHOLD1));
	}
	
	public static boolean satisfiesMultiThreadingConstraints(MatrixBlock m1, MatrixBlock m2, boolean checkMem, boolean checkFLOPs, long FPfactor, int k) {
		boolean sharedTP = (InfrastructureAnalyzer.getLocalParallelism() == k);
		double jvmMem = InfrastructureAnalyzer.getLocalMaxMemory();
		return k > 1
			&& (!checkMem || 8L * m2.clen * k < Math.max(MEM_OVERHEAD_THRESHOLD,0.01*jvmMem))
			//note: cast to double to avoid long overflows on ultra-sparse matrices
			//due to FLOP computation based on number of cells not non-zeros
			&& (!checkFLOPs || (double)FPfactor * m1.rlen * m1.clen * m2.clen >
			(sharedTP ? PAR_MINFLOP_THRESHOLD2 : PAR_MINFLOP_THRESHOLD1));
	}
	
	private static boolean satisfiesMultiThreadingConstraintsTSMM(MatrixBlock m1, boolean leftTranspose, double FPfactor, int k) {
		boolean sharedTP = (InfrastructureAnalyzer.getLocalParallelism() == k);
		double threshold = sharedTP ? PAR_MINFLOP_THRESHOLD2 : PAR_MINFLOP_THRESHOLD1;
		return k > 1 && (leftTranspose?m1.clen:m1.rlen)!=1
			&& ((leftTranspose && FPfactor * m1.rlen * m1.clen * m1.clen > threshold)
			||(!leftTranspose && FPfactor * m1.clen * m1.rlen * m1.rlen > threshold));
	}
	
	public static boolean isUltraSparseMatrixMult(MatrixBlock m1, MatrixBlock m2, boolean m1Perm) {
		if( m2.clen == 1 ) //mv always dense
			return false;
		//note: ultra-sparse matrix mult implies also sparse outputs, hence we need
		//to be conservative an cannot use this for all ultra-sparse matrices.
		double outSp = OptimizerUtils.getMatMultSparsity(
			m1.getSparsity(), m2.getSparsity(), m1.rlen, m1.clen, m2.clen, true);
		return (m1.isUltraSparse() || m2.isUltraSparse()) //base case
			|| (m1.isUltraSparse(false) && m1 == m2) //ultra-sparse self product
			|| (m1Perm && OptimizerUtils.getSparsity(m2.rlen, m2.clen, m2.nonZeros)<1.0)
			|| ((m1.isUltraSparse(false) || m2.isUltraSparse(false)) 
				&& outSp < MatrixBlock.ULTRA_SPARSITY_TURN_POINT2)
			|| (m1.isInSparseFormat() // otherwise no matching branch
				&& m1.getSparsity() < MatrixBlock.ULTRA_SPARSITY_TURN_POINT2
				&& m1.getNonZeros() < MatrixBlock.ULTRA_SPARSE_BLOCK_NNZ
				&& m1.getLength()+m2.getLength() < (long)m1.rlen*m2.clen
				&& outSp < MatrixBlock.SPARSITY_TURN_POINT);
	}
	
	public static boolean isSparseOutputMatrixMult(MatrixBlock m1, MatrixBlock m2) {
		if(m2.rlen == 1 && m2.nonZeros < m2.clen / 4) // vector right ... that is sparse.
			return true;
		//output is a matrix (not vector), very likely sparse, and output rows fit into L1 cache
		if( !(m1.sparse && m2.sparse && m1.rlen > 1 && m2.clen > 1) )
			return false;
		double estSp = OptimizerUtils.getMatMultSparsity(
			m1.getSparsity(), m2.getSparsity(), m1.rlen, m1.clen, m2.clen, false);
		long estNnz = (long)(estSp * m1.rlen * m2.clen);
		boolean sparseOut = MatrixBlock.evalSparseFormatInMemory(m1.rlen, m2.clen, estNnz);
		return m2.clen < 4*1024 && sparseOut;
	}
	
	public static boolean isSparseOutputTSMM(MatrixBlock m1) {
		return isSparseOutputTSMM(m1, false);
	}
	
	public static boolean isSparseOutputTSMM(MatrixBlock m1, boolean ultraSparse) {
		double sp = m1.getSparsity();
		double osp = OptimizerUtils.getMatMultSparsity(sp, sp, m1.rlen, m1.clen, m1.rlen, false);
		double sp_threshold = ultraSparse ?
			MatrixBlock.ULTRA_SPARSITY_TURN_POINT : MatrixBlock.ULTRA_SPARSITY_TURN_POINT2;
		return m1.sparse && osp < sp_threshold;
	}

	public static boolean isOuterProductTSMM(int rlen, int clen, boolean left) {
		return left ? rlen == 1 & clen > 1 : rlen > 1 & clen == 1;
	}

	private static MatrixBlock prepMatrixMultRightInput( MatrixBlock m1, MatrixBlock m2, boolean tm2 ) {
		MatrixBlock ret = m2;
		
		//transpose if dense-dense, skinny rhs matrix (not vector), and memory guarded by output 
		if( tm2 ) {
			MatrixBlock tmpBlock = new MatrixBlock(m2.clen, m2.rlen, m2.sparse);
			ret = LibMatrixReorg.reorg(m2, tmpBlock, new ReorgOperator(SwapIndex.getSwapIndexFnObject()));
		}
		
		return ret;
	}

	//cp non-zeros for dense-dense mm
	private static int copyNonZeroElements( double[] a, final int aixi, final int bixk, final int n, double[] tmpa, int[] tmpbi, final int bklen ) {
		int knnz = 0;
		for( int k = 0; k < bklen; k++ )
			if( a[ aixi+k ] != 0 ) {
				tmpa[ knnz ] = a[ aixi+k ];
				tmpbi[ knnz ] = bixk + k*n;
				knnz ++;
			}
		return knnz;
	}

	//cp non-zeros for dense tsmm
	private static int copyNonZeroElements( double[] a, int aixi, int bixk, final int n, final int nx, double[] tmpa, int[] tmpbi, final int bklen ) {
		int knnz = 0;
		for( int k = 0; k < bklen; k++, aixi+=n, bixk+=nx )
			if( a[ aixi ] != 0 ) {
				tmpa[ knnz ] = a[ aixi ];
				tmpbi[ knnz ] = bixk;
				knnz ++;
			}
		return knnz;
	}
	
	@SuppressWarnings("unused")
	private static void compactSparseOutput(MatrixBlock ret) {
		if( !ret.sparse || ret.nonZeros > ret.rlen || ret.isEmpty() 
			|| ret.getSparseBlock() instanceof SparseBlockCSR )
			return; //early abort
		ret.sparseBlock = SparseBlockFactory
			.copySparseBlock(Type.CSR, ret.sparseBlock, false);
	}
	
	@SuppressWarnings("unused")
	private static void resetPosVect(int[] curk, SparseBlock sblock, int rl, int ru) {
		if( sblock instanceof SparseBlockMCSR ) {
			//all rows start at position 0 (individual arrays)
			Arrays.fill(curk, 0, ru-rl, 0);
		}
		else if( sblock instanceof SparseBlockCSR ) {
			//row start positions given in row ptr array
			SparseBlockCSR csr = (SparseBlockCSR) sblock;
			System.arraycopy(csr.rowPointers(), rl, curk, 0, ru-rl);
		}
		else { //general case
			for(int i=rl; i<ru; i++)
				curk[i-rl] = sblock.pos(i);
		}
	}

	private static void sumScalarResults(List<Future<Double>> tasks, MatrixBlock ret) 
		throws InterruptedException, ExecutionException
	{
		//aggregate partial results and check for errors
		double val = 0;
		for(Future<Double> task : tasks)
			val += task.get();
		ret.set(0, 0, val);
	}

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

	private static class MatrixMultTask implements Callable<Object> 
	{
		private final MatrixBlock _m1;
		private final MatrixBlock _m2;
		private MatrixBlock _ret = null;
		private final boolean _tm2; //transposed m2
		private final boolean _pm2r; //par over m2 rows
		private final boolean _pm2c; //par over m2 rows
		private final boolean _m1Perm; //sparse permutation
		// private final boolean _sparse; //sparse output
		private final int _rl;
		private final int _ru;
		private final ConcurrentHashMap<double[], double[]> _cache;

		protected MatrixMultTask( MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
			boolean tm2, boolean pm2r, boolean pm2c, boolean m1Perm, boolean sparse, int rl, int ru, ConcurrentHashMap<double[], double[]> cache )
		{
			_m1 = m1;
			_m2 = m2;
			_tm2 = tm2;
			_pm2r = pm2r;
			_pm2c = pm2c;
			_m1Perm = m1Perm;
			// _sparse = sparse;
			_rl = rl;
			_ru = ru;
			_cache = cache;

			if( pm2r ) { //vector-matrix / matrix-matrix
				//allocate local result for partial aggregation
				_ret = new MatrixBlock(ret.rlen, ret.clen, false);
			}
			else { //default case
				_ret = ret;
			}
		}
		
		@Override
		public Object call() {
			//setup target index ranges
			int rl = _pm2c ? 0 : _rl;
			int ru = _pm2c ? _m1.rlen : _ru;
			int cl = _pm2c ? _rl : 0;
			int cu = _pm2c ? _ru : _ret.clen;
			
			//thread-local allocation
			if( _pm2r )
				_ret.allocateDenseBlock();
			
			//compute block matrix multiplication
			if( _ret.sparse ) //ultra-sparse
				matrixMultUltraSparse(_m1, _m2, _ret, _m1Perm, rl, ru);
			else if(!_m1.sparse && !_m2.sparse && !_ret.sparse){
				if(_m1.denseBlock instanceof DenseBlockFP64DEDUP && _m2.denseBlock.isContiguous(0,_m1.clen) && cl == 0 && cu == _m2.clen)
					matrixMultDenseDenseMMDedup((DenseBlockFP64DEDUP) _m1.denseBlock, _m2.denseBlock, (DenseBlockFP64DEDUP) _ret.denseBlock, _m2.clen, _m1.clen, rl, ru, _cache);
				else
					matrixMultDenseDense(_m1, _m2, _ret, _tm2, _pm2r, rl, ru, cl, cu);
			}
			else if(_m1.sparse && _m2.sparse)
				matrixMultSparseSparse(_m1, _m2, _ret, _pm2r,  _ret.sparse, rl, ru);
			else if(_m1.sparse)
				matrixMultSparseDense(_m1, _m2, _ret, _pm2r, rl, ru);
			else
				matrixMultDenseSparse(_m1, _m2, _ret, _pm2r, rl, ru);
			
			//maintain block nnz (upper bounds inclusive)
			if( !_pm2r )
				return _ret.recomputeNonZeros(rl, ru-1, cl, cu-1);
			else
				return _ret.getDenseBlockValues();
		}
	}

	private static class MatrixMultChainTask implements Callable<double[]> 
	{
		private MatrixBlock _m1  = null;
		private MatrixBlock _m2  = null;
		private MatrixBlock _m3  = null;
		private ChainType _ct = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultChainTask( MatrixBlock mX, MatrixBlock mV, MatrixBlock mW, ChainType ct, int rl, int ru ) {
			_m1 = mX;
			_m2 = mV;
			_m3 = mW;
			_ct = ct;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public double[] call() {
			//thread-local allocation for partial aggregation
			MatrixBlock ret = new MatrixBlock(1, _m1.clen, false);
			ret.allocateDenseBlock();
			
			if( _m1.sparse )
				matrixMultChainSparse(_m1, _m2, _m3, ret, _ct, _rl, _ru);
			else
				matrixMultChainDense(_m1, _m2, _m3, ret, _ct, _rl, _ru);
			
			//NOTE: we dont do global aggregation from concurrent tasks in order
			//to prevent synchronization (sequential aggregation led to better 
			//performance after JIT)
			return ret.getDenseBlockValues();
		}
	}

	private static class MatrixMultTransposeTask implements Callable<Object> 
	{
		private final MatrixBlock _m1;
		private final MatrixBlock _m1t;
		private final MatrixBlock _ret;
		private final boolean _left;
		private final int _rl;
		private final int _ru;

		protected MatrixMultTransposeTask( MatrixBlock m1, MatrixBlock m1t, MatrixBlock ret, boolean left, int rl, int ru )
		{
			_m1 = m1;
			_m1t = m1t;
			_ret = ret;
			_left = left;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() {
			matrixMultTransposeSelf(_m1, _m1t, _ret, _left, _rl, _ru);
			return null;
		}
	}

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
		public Object call() {
			if( _m2.sparse )
				matrixMultPermuteSparse(_pm1, _m2, _ret1, _ret2, _rl, _ru);
			else if( _ret1.sparse )
				matrixMultPermuteDenseSparse(_pm1, _m2, _ret1, _ret2, _rl, _ru);
			else 
				matrixMultPermuteDense(_pm1, _m2, _ret1, _ret2, _rl, _ru);

			return null;
		}
	}

	private static class MatrixMultWSLossTask implements Callable<Double>
	{
		private MatrixBlock _mX = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _mW = null;
		private MatrixBlock _ret = null;
		private WeightsType _wt = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultWSLossTask(MatrixBlock mX, MatrixBlock mU, MatrixBlock mV, MatrixBlock mW, WeightsType wt, int rl, int ru) {
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
		public Double call() {
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

			return _ret.get(0, 0);
		}
	}

	private static class MatrixMultWSigmoidTask implements Callable<Long> 
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _ret = null;
		private WSigmoidType _wt = null;
		private int _rl = -1;
		private int _ru = -1;
		
		protected MatrixMultWSigmoidTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WSigmoidType wt, int rl, int ru) {
			_mW = mW;
			_mU = mU;
			_mV = mV;
			_ret = ret;
			_wt = wt;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() {
			//core weighted square sum mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWSigmoidDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWSigmoidSparseDense(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			else
				matrixMultWSigmoidGeneric(_mW, _mU, _mV, _ret, _wt, _rl, _ru);
			
			//maintain block nnz (upper bounds inclusive)
			return _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
		}
	}

	private static class MatrixMultWDivTask implements Callable<Long> 
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
		
		protected MatrixMultWDivTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock mX, MatrixBlock ret, WDivMMType wt, int rl, int ru, int cl, int cu) {
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
		public Long call() {
			//core weighted div mm computation
			boolean scalarX = _wt.hasScalar();
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && (_mX==null || !_mX.sparse || scalarX) && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWDivMMDense(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && (_mX==null || _mX.sparse || scalarX) && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWDivMMSparseDense(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
			else
				matrixMultWDivMMGeneric(_mW, _mU, _mV, _mX, _ret, _wt, _rl, _ru, _cl, _cu);
		
			//maintain partial nnz for right (upper bounds inclusive)
			int rl = _wt.isLeft() ? _cl : _rl;
			int ru = _wt.isLeft() ? _cu : _ru;
			return _ret.recomputeNonZeros(rl, ru-1, 0, _ret.getNumColumns()-1);
		}
	}
	
	private static class MatrixMultWCeTask implements Callable<Double>
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private double _eps = 0.0;
		private MatrixBlock _ret = null;
		private WCeMMType _wt = null;
		private int _rl = -1;
		private int _ru = -1;

		protected MatrixMultWCeTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, double eps, WCeMMType wt, int rl, int ru) {
			_mW = mW;
			_mU = mU;
			_mV = mV;
			_eps = eps;
			_wt = wt;
			_rl = rl;
			_ru = ru;
			
			//allocate local result for partial aggregation
			_ret = new MatrixBlock(1, 1, false);
			_ret.allocateDenseBlock();
		}
		
		@Override
		public Double call() {
			//core weighted cross entropy mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWCeMMDense(_mW, _mU, _mV, _eps, _ret, _wt, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWCeMMSparseDense(_mW, _mU, _mV, _eps, _ret, _wt, _rl, _ru);
			else
				matrixMultWCeMMGeneric(_mW, _mU, _mV, _eps, _ret, _wt, _rl, _ru);
			
			
			return _ret.get(0, 0);
		}
	}

	private static class MatrixMultWuTask implements Callable<Long> 
	{
		private MatrixBlock _mW = null;
		private MatrixBlock _mU = null;
		private MatrixBlock _mV = null;
		private MatrixBlock _ret = null;
		private WUMMType _wt = null;
		private ValueFunction _fn = null;
		private int _rl = -1;
		private int _ru = -1;
		
		protected MatrixMultWuTask(MatrixBlock mW, MatrixBlock mU, MatrixBlock mV, MatrixBlock ret, WUMMType wt, ValueFunction fn, int rl, int ru) {
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
		public Long call() {
			//core weighted square sum mm computation
			if( !_mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock() )
				matrixMultWuMMDense(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			else if( _mW.sparse && !_mU.sparse && !_mV.sparse && !_mU.isEmptyBlock() && !_mV.isEmptyBlock())
				matrixMultWuMMSparseDense(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			else
				matrixMultWuMMGeneric(_mW, _mU, _mV, _ret, _wt, _fn, _rl, _ru);
			
			//maintain block nnz (upper bounds inclusive)
			return _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
		}
	}
}
