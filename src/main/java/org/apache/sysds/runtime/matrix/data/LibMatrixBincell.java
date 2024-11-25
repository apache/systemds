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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.SortUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Library for binary cellwise operations (incl arithmetic, relational, etc). Currently,
 * we don't have dedicated support for the individual operations but for categories of
 * operations and combinations of dense/sparse and MM/MV. Safe/unsafe refer to sparse-safe
 * and sparse-unsafe operations.
 * 
 */
public class LibMatrixBincell {

	private static final Log LOG = LogFactory.getLog(LibMatrixBincell.class.getName());
	private static final long PAR_NUMCELL_THRESHOLD2 = 16*1024;   //Min 16K elements

	public enum BinaryAccessType {
		MATRIX_MATRIX,
		MATRIX_COL_VECTOR,
		MATRIX_ROW_VECTOR,
		COL_VECTOR_MATRIX,
		ROW_VECTOR_MATRIX,
		OUTER_VECTOR_VECTOR,
		INVALID;
		public boolean isMatrixVector() {
			return this == MATRIX_COL_VECTOR
				|| this == MATRIX_ROW_VECTOR
				|| this == COL_VECTOR_MATRIX
				|| this == ROW_VECTOR_MATRIX;
		}
	}
	
	private LibMatrixBincell() {
		//prevent instantiation via private constructor
	}
	
	///////////////////////////////////
	// public matrix bincell interface
	///////////////////////////////////
	
	public static MatrixBlock uncellOp(MatrixBlock m1, MatrixBlock ret, UnaryOperator op) {
		
		if(!m1.sparse && !m1.isEmptyBlock(false) 
			&& op.getNumThreads() > 1 && m1.getLength() > PAR_NUMCELL_THRESHOLD2  ) {
			//note: we apply multi-threading in a best-effort manner here
			//only for expensive operators such as exp, log, sigmoid, because
			//otherwise allocation, read and write anyway dominates
			if (!op.isInplace() || m1.isEmpty())
				ret.allocateDenseBlock(false);
			else
				ret = m1;
			int k = op.getNumThreads();
			DenseBlock a = m1.getDenseBlock();
			DenseBlock c = ret.getDenseBlock();
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<UncellTask> tasks = new ArrayList<>();
				ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(ret.rlen, k, false);
				for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
					tasks.add(new UncellTask(a, c, op, lb, lb+blklens.get(i)));
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				
				long nnz = 0;
				for( Future<Long> task : taskret )
					nnz += task.get();
				ret.setNonZeros(nnz);
			}
			catch(InterruptedException | ExecutionException ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}
		else {
			if (op.isInplace() && !m1.isInSparseFormat() )
				ret = m1;
			
			//default execute unary operations
			if(op.sparseSafe)
				sparseUnaryOperations(m1, ret, op);
			else
				denseUnaryOperations(m1, ret, op);

			ret.recomputeNonZeros();
		}
		return ret;
	}
	
	public static MatrixBlock bincellOpScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int k) {
		// estimate the sparsity structure of result matrix
		boolean sp = m1.sparse; // by default, we guess result.sparsity=input.sparsity
		if (!op.sparseSafe)
			sp = false; // if the operation is not sparse safe, then result will be in dense format
		
		//allocate the output matrix block
		if( ret==null )
			ret = new MatrixBlock(m1.getNumRows(), m1.getNumColumns(), sp, m1.nonZeros);
		else
			ret.reset(m1.getNumRows(), m1.getNumColumns(), sp, m1.nonZeros);

		if((op.fn instanceof Multiply && op.getConstant() == 0.0))
			return ret; // no op
		
		// fallback to singlet-threaded for special cases
		if( k <= 1 || m1.isEmpty() || !op.sparseSafe 
			|| ret.getLength() < PAR_NUMCELL_THRESHOLD2 ) {
			bincellOpScalarSingleThread(m1, ret, op);
		}
		else{
			bincellOpScalarParallel(m1, ret, op, k);
		}
		// ensure empty results sparse representation
		// (no additional memory requirements)
		if(ret.isEmptyBlock(false))
			ret.examSparsity(k);
		return ret;
	}

	public static MatrixBlock bincellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		try{

			//Timing time = new Timing(true);
			isValidDimensionsBinary(m1, m2);
			op = replaceOpWithSparseSafeIfApplicable(m1, m2, op);
			
			//compute output dimensions
			final BinaryAccessType atype = getBinaryAccessType(m1, m2);
			boolean outer = (atype == BinaryAccessType.OUTER_VECTOR_VECTOR); 
			int rows = m1.getNumRows();
			int cols = outer ? m2.getNumColumns(): m1.getNumColumns();

			//estimate output sparsity
			SparsityEstimate resultSparse = estimateSparsityOnBinary(m1, m2, op);
			if(ret == null)
				ret = new MatrixBlock(rows, cols, resultSparse.sparse, resultSparse.estimatedNonZeros);
			else
				ret.reset(rows, cols, resultSparse.sparse, resultSparse.estimatedNonZeros);
	
			final boolean skipEmpty = shouldSkipEmpty(m2, op);
			final boolean e1 = m1.isEmpty();
			final boolean e2 = m2.isEmpty();
			// early abort
			if(skipEmpty && (e1 || e2))
				return ret;
			
			ret.allocateBlock();
			int k = op.getNumThreads();

			// fallback to sequential computation for specialized operations
			// TODO fix all variants to be feasible for multi-threading
			if(k <= 1 || m1.isEmpty() || m2.isEmpty()
				|| ret.getLength() < PAR_NUMCELL_THRESHOLD2
				|| isSafeBinaryMcVDenseSparseMult(m1, m2, ret, op)
				|| !CommonThreadPool.useParallelismOnThread())
			{
				bincellOpMatrixSingle(m1, m2, ret, op, atype);
			}
			else {
				bincellOpMatrixParallel(m1, m2, ret, op, atype, k);
			}
			
			if(ret.isEmptyBlock(false))
				ret.examSparsity(k);
			//System.out.println("BinCell " + op + " " + m1.getNumRows() + ", " + m1.getNumColumns() + ", " + m1.getNonZeros()
			// 	+ " -- " + m2.getNumRows() + ", " + m2.getNumColumns() + " " + m2.getNonZeros() + "\t\t" + time.stop());
			
			return ret;
		}
		catch(Exception e){
			throw new RuntimeException("Failed to perform binary operation", e);
		}
	}

		
	/**
	 * NOTE: operations in place always require m1 and m2 to be of equal dimensions
	 * 
	 * defaults to right side operations, updating the m1 matrix with like:
	 * 
	 * m1ret op m2
	 * 
	 * @param m1ret result matrix updated in place
	 * @param m2 matrix block the other matrix to take values from
	 * @param op binary operator the operator that is placed in the middle of m1ret and m2
	 * @return The same pointer to m1ret argument, and the updated result.
	 */
	public static MatrixBlock bincellOpInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		return bincellOpInPlaceRight(m1ret, m2, op);
	}

	/**
	 * Right side operations, updating the m1 matrix like:
	 * 
	 * m1ret op m2
	 * 
	 * @param m1ret result matrix updated in place
	 * @param m2 matrix block the other matrix to take values from
	 * @param op binary operator the operator that is placed in the middle of m1ret and m2
	 * @return The result MatrixBlock (same object pointer to m1ret argument)
	 */
	public static MatrixBlock bincellOpInPlaceRight(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		isValidDimensionsBinary(m1ret, m2);
		op = replaceOpWithSparseSafeIfApplicable(m1ret, m2, op);

		//estimate output sparsity
		final boolean skipEmpty = shouldSkipEmpty(m2, op);
		final boolean e1 = m1ret.isEmpty();
		final boolean e2 = m2.isEmpty();
		// early abort
		if(skipEmpty && (e1 || e2)) {
			m1ret.reset(m1ret.rlen, m1ret.clen, 0.0); // fill with 0.0
			return m1ret;
		}

		final SparsityEstimate resultSparse = estimateSparsityOnBinary(m1ret, m2, op);
		if(e1 && e2){
			double r = op.fn.execute(0.0,0.0);
			m1ret.fill(r);
			return m1ret;
		}
		else if((resultSparse.sparse && e1))
			m1ret.allocateSparseRowsBlock();
		else if((!resultSparse.sparse && e1))
			m1ret.allocateDenseBlock();
		else if(resultSparse.sparse && !m1ret.sparse)
			m1ret.denseToSparse();
		else if(!resultSparse.sparse && m1ret.sparse)
			m1ret.sparseToDense();
		
		final long nnz;

		//execute binary cell operations
		if(op.sparseSafe || isSparseSafeDivideOrPow(op, m2))
			nnz = safeBinaryInPlace(m1ret, m2, op);
		else
			nnz = unsafeBinaryInPlace(m1ret, m2, op);
		
		m1ret.setNonZeros(nnz);

		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( m1ret.isEmptyBlock(false) )
			m1ret.examSparsity();

		return m1ret;
	}

	/**
	 * Left side operations, updating the m1 matrix like:
	 * 
	 * m2 op m1ret
	 * 
	 * @param m1ret result matrix updated in place
	 * @param m2 matrix block the other matrix to take values from
	 * @param op binary operator the operator that is placed in the middle of m1ret and m2
	 * @return The result MatrixBlock (same object pointer to m1ret argument)
	 */
	public static MatrixBlock bincellOpInPlaceLeft(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		final int nRows = m1ret.getNumRows();
		final int nCols = m1ret.getNumColumns();
		op = replaceOpWithSparseSafeIfApplicable(m1ret, m2, op);
		if(m1ret.isInSparseFormat()){
			// not doing in place, since the m1ret is in sparse format, and m2 might make it dense.
			// this is not ideal either, but makes it work
			LOG.warn("Inefficient bincell op in place left, because output is materialized in new matrix");
			MatrixBlock right = new MatrixBlock(nRows, nCols, true);
			right.copyShallow(m1ret);
			m1ret.cleanupBlock(true, true);
			bincellOp(m2, right, m1ret, op);
			return m1ret;
		}

		// m1ret is dense:
		final double[] retV = m1ret.getDenseBlockValues();
		final ValueFunction f = op.fn;
		
		if(m2.isInSparseFormat() && op.sparseSafe) {
			final SparseBlock sb = m2.getSparseBlock();
			for(int row = 0; row < nRows; row++){
				if(sb.isEmpty(row)){
					continue;
				}
				final int apos = sb.pos(row);
				final int alen = sb.size(row) + apos;
				final int[] aix = sb.indexes(row);
				final double[] aval = sb.values(row);
				final int offsetV = row * nCols;
				for(int j = apos; j < alen; j++){
					final int idx = offsetV + aix[j];
					retV[idx] =  f.execute(aval[j], retV[idx]);
				}
			}
		}
		else if(m2.isInSparseFormat()){
			throw new NotImplementedException("Not implemented left bincell in place unsafe operations");
		}
		else{
			final double[] m2V = m2.getDenseBlockValues();
			final int size = nRows * nCols;
			for(int i = 0; i < size; i++ ){
				retV[i] = f.execute(m2V[i], retV[i]);
			}
			
			if( m1ret.isEmptyBlock(false) )
				m1ret.examSparsity();
		}
		return m1ret;
	}

	public static BinaryAccessType getBinaryAccessType(MatrixBlock m1, MatrixBlock m2)
	{
		int rlen1 = m1.rlen;
		int rlen2 = m2.rlen;
		int clen1 = m1.clen;
		int clen2 = m2.clen;
		
		if( rlen1 == rlen2 && clen1 == clen2 )
			return BinaryAccessType.MATRIX_MATRIX;
		else if( rlen1 == rlen2 && clen2 == 1)
			return BinaryAccessType.MATRIX_COL_VECTOR;
		else if( clen1 == clen2 && rlen2 == 1 )
			return BinaryAccessType.MATRIX_ROW_VECTOR;
		else if( clen1 == 1 && rlen2 == 1 )
			return BinaryAccessType.OUTER_VECTOR_VECTOR;
		else
			return BinaryAccessType.INVALID;
	}

	public static BinaryAccessType getBinaryAccessTypeExtended(MatrixBlock m1, MatrixBlock m2) {
		final int rlen1 = m1.rlen;
		final int rlen2 = m2.rlen;
		final int clen1 = m1.clen;
		final int clen2 = m2.clen;

		if(rlen1 == rlen2) {
			if(clen1 == clen2)
				return BinaryAccessType.MATRIX_MATRIX;
			else if(clen1 < clen2 && clen1 == 1)
				return BinaryAccessType.COL_VECTOR_MATRIX;
			else if(clen2 == 1)
				return BinaryAccessType.MATRIX_COL_VECTOR;
			else 
				return BinaryAccessType.INVALID;
		}
		else if(clen1 == clen2) {
			if(rlen1 < rlen2 && rlen1 == 1)
				return BinaryAccessType.ROW_VECTOR_MATRIX;
			else if(rlen2 == 1)
				return BinaryAccessType.MATRIX_ROW_VECTOR;
			else 
				return BinaryAccessType.INVALID;
		}
		else if(clen1 == 1 && rlen2 == 1)
			return BinaryAccessType.OUTER_VECTOR_VECTOR;
		else
			return BinaryAccessType.INVALID;
	}

	public static void isValidDimensionsBinary(MatrixBlock m1, MatrixBlock m2)	{
		final int rlen1 = m1.rlen;
		final int clen1 = m1.clen;
		final int rlen2 = m2.rlen;
		final int clen2 = m2.clen;
		
		//currently we support three major binary cellwise operations:
		//1) MM (where both dimensions need to match)
		//2) MV operations w/ V either being a right-hand-side column or row vector 
		//  (where one dimension needs to match and the other dimension is 1)
		//3) VV outer vector operations w/ a common dimension of 1 
		boolean isValid = (   (rlen1 == rlen2 && clen1==clen2)        //MM 
							|| (rlen1 == rlen2 && clen1 > 1 && clen2 == 1) //MVc
							|| (clen1 == clen2 && rlen1 > 1 && rlen2 == 1) //MVr
							|| (clen1 == 1 && rlen2 == 1 ) );              //VV

		if( !isValid ) {
			throw new DMLRuntimeException("Block sizes are not matched for binary " +
					"cell operations: " + rlen1 + "x" + clen1 + " vs " + rlen2 + "x" + clen2);
		}
	}

	public static BinaryOperator replaceOpWithSparseSafeIfApplicable(MatrixBlock m1, MatrixBlock m2, BinaryOperator op) {
		if((m1.getSparsity() < 1 || m2.getSparsity() < 1) && op.fn instanceof Builtin &&
			((Builtin) op.fn).bFunc == BuiltinCode.LOG) {
			op = new BinaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.LOG_NZ), op.getNumThreads());
		}
		return op;
	}

	private static void bincellOpScalarSingleThread(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {	
		//execute binary cell operations
		long nnz = 0;
		if(op.sparseSafe)
			nnz = safeBinaryScalar(m1, ret, op, 0, m1.rlen);
		else
			nnz = unsafeBinaryScalar(m1, ret, op);
		
		ret.nonZeros = nnz;
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
		
	}
	
	private static void bincellOpScalarParallel(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int k) {

		// preallocate dense/sparse block for multi-threaded operations
		ret.allocateBlock();

		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			// execute binary cell operations
			final ArrayList<BincellScalarTask> tasks = new ArrayList<>();

			final int rMax = m1.getNumRows();
			final int blkLen = Math.max(Math.max(rMax / k, 1000 / ret.getNumColumns()), 1);
			for(int i = 0; i < rMax; i += blkLen)
				tasks.add(new BincellScalarTask(m1, ret, op, i, Math.min(rMax, i + blkLen)));

			// aggregate non-zeros
			long nnz = 0;
			for(Future<Long> task : pool.invokeAll(tasks))
				nnz += task.get();
			ret.nonZeros = nnz;
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
	}
	
	
	private static void bincellOpMatrixParallel(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, BinaryAccessType atype, int k) throws Exception {
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			//execute binary cell operations
			ArrayList<BincellTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(ret.rlen, k, false);
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
				tasks.add(new BincellTask(m1, m2, ret, op, atype, lb, lb+blklens.get(i)));
			List<Future<Long>> taskret = pool.invokeAll(tasks);
			
			//aggregate non-zeros
			long nnz =  0; //reset after execute
			for( Future<Long> task : taskret )
				nnz += task.get();
			
			ret.nonZeros = nnz;
			//ensure empty results sparse representation
			//(no additional memory requirements)
			if( ret.isEmptyBlock(false) )
				ret.examSparsity(k);
		}
		finally{
			pool.shutdown();
		}
		
	}

	private static void bincellOpMatrixSingle(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, BinaryAccessType atype) {
		// execute binary cell operations
		long nnz = 0;
		nnz = binCellOpExecute(m1, m2, ret, op, atype,0, m1.rlen);
		ret.setNonZeros(nnz);
	}

	private static long binCellOpExecute(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		BinaryAccessType atype, int rl, int ru) {
		long nnz;
		if(op.sparseSafe || isSparseSafeDivideOrPow(op, m2))
			nnz = safeBinary(m1, m2, ret, op, atype, rl, ru);
		else
			nnz = unsafeBinary(m1, m2, ret, op, rl, ru);
		return nnz;
	}
	

	private static boolean isSparseSafeDivideOrPow(BinaryOperator op, MatrixBlock rhs){
		// if rhs is fully dense, there cannot be a /0 and hence DIV becomes sparse safe
		// similarly if all non zero Power is sparse safe.
		return((op.fn instanceof Divide || op.fn instanceof Power) &&
			rhs.getNonZeros() == (long) rhs.getNumRows() * rhs.getNumColumns());
	}

	//////////////////////////////////////////////////////
	// private sparse-safe/sparse-unsafe implementations
	///////////////////////////////////

	private static void denseUnaryOperations(MatrixBlock m1, MatrixBlock ret, UnaryOperator op) {
		//prepare 0-value init (determine if unnecessarily sparse-unsafe)
		double val0 = op.fn.execute(0d);
		
		final int m = m1.rlen;
		final int n = m1.clen;
		
		//early abort possible if unnecessarily sparse unsafe
		//(otherwise full init with val0, no need for computation)
		if( m1.isEmptyBlock(false) ) {
			if( val0 != 0 )
				ret.reset(m, n, val0);
			return;
		}
		
		//redirection to sparse safe operation w/ init by val0
		if( m1.sparse && val0 != 0 ) {
			ret.reset(m, n, val0);
			ret.nonZeros = (long)m * n;
		}
		sparseUnaryOperations(m1, ret, op);
	}
	
	private static void sparseUnaryOperations(MatrixBlock m1, MatrixBlock ret, UnaryOperator op) {
		//early abort possible since sparse-safe
		if( m1.isEmptyBlock(false) )
			return;
		
		final int m = m1.rlen;
		final int n = m1.clen;
		
		if( m1.sparse && ret.sparse ) //SPARSE <- SPARSE
		{
			ret.allocateSparseRowsBlock();
			SparseBlock a = m1.sparseBlock;
			SparseBlock c = ret.sparseBlock;
		
			long nnz = 0;
			for(int i=0; i<m; i++) {
				if( a.isEmpty(i) ) continue;
				
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				c.allocate(i, alen); //avoid repeated alloc
				for( int j=apos; j<apos+alen; j++ ) {
					double val = op.fn.execute(avals[j]);
					c.append(i, aix[j], val);
					nnz += (val != 0) ? 1 : 0;
				}
			}
			ret.nonZeros = nnz;
		}
		else if( m1.sparse ) //DENSE <- SPARSE
		{
			ret.allocateDenseBlock(false);
			SparseBlock a = m1.sparseBlock;
			DenseBlock c = ret.denseBlock;
			long nnz = (ret.nonZeros > 0) ?
				(long) m*n-a.size() : 0;
			for(int i=0; i<m; i++) {
				if( a.isEmpty(i) ) continue;
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				for( int j=apos; j<apos+alen; j++ ) {
					double val = op.fn.execute(avals[j]);
					cvals[cix + aix[j]] = val; 
					nnz += (val != 0) ? 1 : 0;
				}
			}
			ret.nonZeros = nnz;
		}
		else //DENSE <- DENSE
		{
			if( m1 != ret ) //!in-place
				ret.allocateDenseBlock(false);
			DenseBlock da = m1.getDenseBlock();
			DenseBlock dc = ret.getDenseBlock();
			
			//unary op, incl nnz maintenance
			long nnz = 0;
			for( int bi=0; bi<da.numBlocks(); bi++ ) {
				double[] a = da.valuesAt(bi);
				double[] c = dc.valuesAt(bi);
				int len = da.size(bi);
				for( int i=0; i<len; i++ ) {
					c[i] = op.fn.execute(a[i]);
					nnz += (c[i] != 0) ? 1 : 0;
				}
			}
			ret.nonZeros = nnz;
		}
	}
	
	private static long safeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		BinaryAccessType atype, int rl, int ru){

		// skip empty blocks (since sparse-safe)
		if(m1.isEmptyBlock(false) && m2.isEmptyBlock(false))
			return 0;

		if(atype.isMatrixVector()) 
			return safeBinaryMV(m1, m2, ret, op, atype, rl, ru);
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) //VECTOR - VECTOR
			return safeBinaryVVGeneric(m1, m2, ret, op, rl, ru);
		else // MATRIX - MATRIX
			return safeBinaryMM(m1, m2, ret, op, rl, ru);
	}

	private static long safeBinaryMM(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int rl, int ru)
	{
		boolean copyLeftRightEmpty = (op.fn instanceof Plus || op.fn instanceof Minus 
		|| op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		boolean copyRightLeftEmpty = (op.fn instanceof Plus);

		if( copyLeftRightEmpty && m2.isEmpty() ) {
			//ret remains unchanged so a shallow copy is sufficient
			ret.copyShallow(m1);
			return ret.getNonZeros();
		}
		else if( copyRightLeftEmpty && m1.isEmpty() ) {
			//ret remains unchanged so a shallow copy is sufficient
			ret.copyShallow(m2);
			return ret.getNonZeros();
		}
		else if(m1.sparse && m2.sparse) {
			return safeBinaryMMSparseSparse(m1, m2, ret, op, rl, ru);
		}
		else if( !ret.sparse && (m1.sparse || m2.sparse) &&
			(op.fn instanceof Plus || op.fn instanceof Minus ||
			op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply ||
			(op.fn instanceof Multiply && !m2.sparse ))) {
			return safeBinaryMMSparseDenseDense(m1, m2, ret, op, rl, ru);
		}
		else if(!ret.sparse && !m1.isInSparseFormat() && !m2.isInSparseFormat() 
				&& !m1.isEmpty() && !m2.isEmpty()) {
			return safeBinaryMMDenseDenseDense(m1, m2, ret, op, rl, ru);
		}
		else if( shouldSkipEmpty(m2, op) && (m1.sparse || m2.sparse) ) {
			return safeBinaryMMSparseDenseSkip(m1, m2, ret, op, rl, ru);
		}
		else { //generic case
			return safeBinaryMMGeneric(m1, m2, ret, op, rl, ru);
		}
	}

	private static long safeBinaryMV(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		BinaryAccessType atype, int rl, int ru)
	{
		if( !m1.sparse && !m2.sparse && !ret.sparse ) //DENSE all
			return safeBinaryMVDense(m1, m2, ret, op, rl, ru);
		else if( m1.sparse && !m2.sparse && !m2.isEmpty() && !ret.sparse
			&& atype == BinaryAccessType.MATRIX_ROW_VECTOR)
			return safeBinaryMVSparseDenseRow(m1, m2, ret, op, rl, ru);
		else if( m1.sparse ) //SPARSE m1
			return safeBinaryMVSparseLeft(m1, m2, ret, op, rl, ru);
		else if( isSafeBinaryMcVDenseSparseMult(m1, m2, ret, op) )
			safeBinaryMcVDenseSparseMult(m1, m2, ret, op);
		else //generic combinations
			return safeBinaryMVGeneric(m1, m2, ret, op, rl, ru);

		//default catch all (set internally by single-threaded methods)
		return ret.getNonZeros();
	}

	
	
	private static boolean shouldSkipEmpty(MatrixBlock m2, BinaryOperator op) {
		return op.fn instanceof Multiply 
			|| (op.fn instanceof Builtin && ((Builtin)op.fn).bFunc == BuiltinCode.LOG_NZ)
			|| isSparseSafeDivideOrPow(op, m2);
	}

	private static long safeBinaryMVDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl,
		int ru) {

		// early abort on skip and empty
		if(op.fn instanceof Multiply && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)))
			return 0; // skip entire empty block

		final BinaryAccessType atype = getBinaryAccessType(m1, m2);

		if(atype == BinaryAccessType.MATRIX_COL_VECTOR)
			return safeBinaryMVDenseColVector(m1, m2, ret, op, rl, ru);
		else // if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
			return safeBinaryMVDenseRowVector(m1, m2, ret, op, rl, ru);
	}

	private static long safeBinaryMVDenseColVector(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		int rl, int ru) {
		final int clen = m1.clen;
		final DenseBlock da = m1.getDenseBlock();
		final DenseBlock dc = ret.getDenseBlock();
		final double[] b = m2.getDenseBlockValues();

		if(op.fn instanceof Multiply)
			return safeBinaryMVDenseColVectorMultiply(da, b, dc, clen, rl, ru);
		else if(op.fn instanceof Divide)
			return safeBinaryMVDenseColVectorDivide(da, b, dc, clen, rl, ru);
		else
			return safeBinaryMVDenseColVectorGeneric(da, b, dc, clen, op, rl, ru);
	}

	private static long safeBinaryMVDenseColVectorGeneric(DenseBlock da, double[] b, DenseBlock dc, int clen,
		BinaryOperator op, int rl, int ru) {
		if(b == null)
			return safeBinaryMVDenseColVectorGenericEmptyVector(da, dc, clen, op, rl, ru);
		else
			return safeBinaryMVDenseColVectorGenericDenseVector(da, b, dc, clen, op, rl, ru);
	}

	private static long safeBinaryMVDenseColVectorGenericEmptyVector(DenseBlock da, DenseBlock dc, int clen,
		BinaryOperator op, int rl, int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) {
			final double[] a = da.values(i);
			final double[] c = dc.values(i);
			final int ix = da.pos(i);
			for(int j = 0; j < clen; j++) {
				double val = op.fn.execute(a[ix + j], 0);
				nnz += ((c[ix + j] = val) != 0) ? 1 : 0;
			}
		}
		return nnz;
	}

	private static long safeBinaryMVDenseColVectorGenericDenseVector(DenseBlock da, double[] b, DenseBlock dc, int clen,
		BinaryOperator op, int rl, int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) {
			final double[] a = da.values(i);
			final double[] c = dc.values(i);
			final int ix = da.pos(i);
			final double v2 = b[i];

			for(int j = 0; j < clen; j++) {
				double val = op.fn.execute(a[ix + j], v2);
				nnz += ((c[ix + j] = val) != 0) ? 1 : 0;
			}
		}
		return nnz;
	}

	private static long safeBinaryMVDenseColVectorMultiply(DenseBlock da, double[] b, DenseBlock dc, int clen, int rl,
		int ru) {
		if(b == null)
			return 0;
		else {
			long nnz = 0;
			for(int i = rl; i < ru; i++) {
				final double[] a = da.values(i);
				final double[] c = dc.values(i);
				final int ix = da.pos(i);

				// replicate vector value
				final double v2 = b[i];
				if(v2 == 0) // skip empty rows
					continue;
				else if(v2 == 1) { // ROW COPY
					for(int j = ix; j < clen + ix; j++)
						nnz += ((c[j] = a[j]) != 0) ? 1 : 0;
				}
				else {// GENERAL CASE
					for(int j = ix; j < clen + ix; j++)
						nnz += ((c[j] = a[j] * v2) != 0) ? 1 : 0;
				}
			}
			return nnz;
		}
	}

	private static long safeBinaryMVDenseColVectorDivide(DenseBlock da, double[] b, DenseBlock dc, int clen, int rl,
		int ru) {

		if(b == null){
			dc.fill(Double.NaN);
			return (long)dc.getDim(0) * dc.getDim(1);
		}
		else {
			long nnz = 0;
			for(int i = rl; i < ru; i++) {
				final double[] a = da.values(i);
				final double[] c = dc.values(i);
				final int ix = da.pos(i);
				final double v2 = b[i];
				processRowMVDenseDivide(a,c, ix, clen, v2);
			}
			return nnz;
		}
	}

	private static long processRowMVDenseDivide(double[] a, double[] c, int ix, int clen, double v2) {
		long nnz = 0;
		if(v2 == 0) {// divide by zero.
			Arrays.fill(c, ix, clen, Double.NaN);
			nnz += clen;
		}
		else if(v2 == 1) { // ROW COPY
			for(int j = ix; j < clen + ix; j++)
				nnz += ((c[j] = a[j]) != 0) ? 1 : 0;
		}
		else { // GENERAL CASE
			for(int j = ix; j < clen + ix; j++)
				nnz += ((c[j] = a[j] / v2) != 0) ? 1 : 0;
		}
		return nnz;
	}

	private static long safeBinaryMVDenseRowVector(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		int rl, int ru) {
		final int clen = m1.clen;

		final DenseBlock da = m1.getDenseBlock();
		final DenseBlock dc = ret.getDenseBlock();
		long nnz = 0;
		final double[] b = m2.getDenseBlockValues(); // always single block

		if(da == null && b == null) { // both empty
			double val = op.fn.execute(0, 0);
			dc.set(rl, ru, 0, clen, val);
			nnz += (val != 0) ? (long) (ru - rl) * clen : 0;
		}
		else if(da == null) // left empty
		{
			// compute first row
			double[] c = dc.values(rl);
			for(int j = 0; j < clen; j++) {
				double val = op.fn.execute(0, b[j]);
				nnz += ((c[j] = val) != 0) ? (ru - rl) : 0;
			}
			// copy first to all other rows
			for(int i = rl + 1; i < ru; i++)
				dc.set(i, c);
		}
		else // default case (incl right empty)
		{
			for(int i = rl; i < ru; i++) {
				double[] a = da.values(i);
				double[] c = dc.values(i);
				int ix = da.pos(i);
				for(int j = 0; j < clen; j++) {
					double val = op.fn.execute(a[ix + j], ((b != null) ? b[j] : 0));
					nnz += ((c[ix + j] = val) != 0) ? 1 : 0;
				}
			}
		}
		return nnz;
	}
	

	private static long safeBinaryMVSparseDenseRow(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int rl, int ru) 
	{
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		int clen = m1.clen;
		SparseBlock a = m1.sparseBlock;
		double[] b = m2.getDenseBlockValues();
		DenseBlock c = ret.getDenseBlock();

		//early abort on skip and empty
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return 0; // skip entire empty block
		else if( !skipEmpty && m2.isEmptyBlock(false) && rl == 0 //only first task
			&& (op.fn instanceof Minus || op.fn instanceof Plus))
		{
			ret.copy(m1);
			return ret.nonZeros;
		}

		//prepare op(0, m2) vector once for all rows (inplace in first row)
		double[] tmp = c.values(rl);
		int tpos = c.pos(rl);
		if( !skipEmpty ) {
			for( int i=0; i<clen; i++ )
				tmp[tpos+i] = op.fn.execute(0, b[i]);
		}
		
		//set prepared empty row vector into output
		for( int i=rl+1; i<ru; i++ ) {
			if( skipEmpty && (a==null || a.isEmpty(i)) )
				continue; //skip empty rows
			System.arraycopy(tmp, tpos, c.values(i), c.pos(i), clen);
		}
		
		//execute remaining sparse-safe operation
		long nnz = 0;
		for( int i=rl; i<ru; i++ ) {
			if( skipEmpty && (a==null || a.isEmpty(i)) )
				continue; //skip empty rows
			double[] cvals = c.values(i);
			int cpos = c.pos(i);
			
			//overwrite row cells with existing sparse lhs values
			if( a!=null && !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				for( int j=apos; j<apos+alen; j++ )
					cvals[cpos+aix[j]] = op.fn.execute(avals[j], b[aix[j]]);
			}
			
			//compute row nnz with temporal locality
			nnz += UtilFunctions.computeNnz(cvals, cpos, clen);
		}
		return nnz;
	}
	
	private static long safeBinaryMVSparseLeft(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int rl, int ru)
	{
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply || isSparseSafeDivideOrPow(op, m2));
		BinaryAccessType atype = getBinaryAccessType(m1, m2);

		// early abort on skip and empty
		if(skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)))
			return 0; // skip entire empty block

		if(atype == BinaryAccessType.MATRIX_COL_VECTOR)
			safeBinaryMVSparseLeftColVector(m1, m2, ret, op, rl, ru);
		else if(atype == BinaryAccessType.MATRIX_ROW_VECTOR)
			safeBinaryMVSparseLeftRowVector(m1, m2, ret, op, rl, ru);

		return ret.recomputeNonZeros(rl, ru-1);
	}

	private static void safeBinaryMVSparseLeftColVector(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int rl, int ru)
	{
		final boolean isMultiply = (op.fn instanceof Multiply);
		final boolean skipEmpty = (isMultiply || isSparseSafeDivideOrPow(op, m2));

		final int clen = m1.clen;
		final SparseBlock a = m1.sparseBlock;
		final boolean aNull = a == null;
		if(skipEmpty && a == null)
			return;
		if(ret.isInSparseFormat()){
			final SparseBlockMCSR rb = (SparseBlockMCSR) ret.getSparseBlock();
			for(int i = rl; i < ru; i++) {
				final double v2 = m2.get(i, 0);
				final boolean emptyRow = !aNull ? a.isEmpty(i) : true;
				if((skipEmpty && (emptyRow || v2 == 0))  // skip empty one side zero
					|| (emptyRow && v2 == 0)){ // both sides zero 
					continue; // skip empty rows
				}
				final double vz = op.fn.execute(0, v2);
				final boolean fill = vz != 0;
	
				if(isMultiply && v2 == 1) // ROW COPY
					ret.appendRow(i, a.get(i));
				else if(!fill)
					safeBinaryMVSparseColVectorRowNoFill(a, i, rb, v2, emptyRow, op);
				else  // GENERAL CASE
					safeBinaryMVSparseColVectorRowWithFill(a, i, rb, vz, v2, clen, emptyRow, op);
			}
		}
		else{
			final DenseBlock db = ret.getDenseBlock();
			for(int i = rl; i < ru; i++) {
				final double v2 = m2.get(i, 0);

				final boolean emptyRow = !aNull ? a.isEmpty(i) : true;
				if((skipEmpty && (emptyRow || v2 == 0))  // skip empty one side zero
					|| (emptyRow && v2 == 0)){ // both sides zero 
					continue; // skip empty rows
				}
				final double vz = op.fn.execute(0, v2);
				final boolean fill = vz != 0;
				if(isMultiply && v2 == 1) // ROW COPY
					ret.appendRow(i, a.get(i));
				else if(!fill)
					safeBinaryMVSparseColVectorRowNoFill(a, i, db, v2, emptyRow, op);
				else  // GENERAL CASE
					safeBinaryMVSparseColVectorRowWithFill(a, i, db, vz, v2, clen, emptyRow, op);
			}
		}
	}

	private static final void safeBinaryMVSparseColVectorRowNoFill(SparseBlock a, int i, SparseBlockMCSR rb, double v2,
		boolean emptyRow, BinaryOperator op) {
		if(!emptyRow) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			rb.allocate(i, alen); // likely alen allocation
			for(int j = apos; j < apos + alen; j++) {
				double v = op.fn.execute(avals[j], v2);
				rb.append(i, aix[j], v);
			}
		}
	}

	private static final void safeBinaryMVSparseColVectorRowNoFill(SparseBlock a, int i, DenseBlock rb, double v2,
		boolean emptyRow, BinaryOperator op) {
		if(!emptyRow) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			for(int j = apos; j < apos + alen; j++) {
				double v = op.fn.execute(avals[j], v2);
				rb.set(i, aix[j], v);
			}
		}
	}

	private static final void safeBinaryMVSparseColVectorRowWithFill(SparseBlock a, int i, SparseBlockMCSR rb, double vz,
		double v2, int clen, boolean emptyRow, BinaryOperator op) {
		int lastIx = -1;
		if(!emptyRow) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			rb.allocate(i, clen); // likely clen allocation
			for(int j = apos; j < apos + alen; j++) {

				fillZeroValuesScalar(vz, rb, i, lastIx + 1, aix[j]);
				// actual value
				double v = op.fn.execute(avals[j], v2);
				rb.append(i, aix[j], v);
				lastIx = aix[j];
			}
			fillZeroValuesScalar(vz, rb, i, lastIx + 1, clen);
		}
		else{
			rb.allocate(i, clen);
			fillZeroValuesScalar(vz, rb, i, lastIx + 1, clen);
		}
	}

	private static final void safeBinaryMVSparseColVectorRowWithFill(SparseBlock a, int i, DenseBlock rb, double vz,
		double v2, int clen, boolean emptyRow, BinaryOperator op) {
		int lastIx = -1;
		if(!emptyRow) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			for(int j = apos; j < apos + alen; j++) {

				fillZeroValuesScalar(vz, rb, i, lastIx + 1, aix[j]);
				// actual value
				double v = op.fn.execute(avals[j], v2);
				rb.set(i, aix[j], v);
				lastIx = aix[j];
			}
			fillZeroValuesScalar(vz, rb, i, lastIx + 1, clen);
		}
		else{
			fillZeroValuesScalar(vz, rb, i, lastIx + 1, clen);
		}
	}

	private static final void fillZeroValuesScalar( double v, SparseBlock ret,
		int rpos, int cpos, int len) {

		for(int k = cpos; k < len; k++)
			ret.append(rpos, k, v);

	}


	private static final void fillZeroValuesScalar( double v, DenseBlock ret,
		int rpos, int cpos, int len) {
		ret.set(rpos, rpos + 1, cpos, len, v);
	}

	private static void safeBinaryMVSparseLeftRowVector(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int rl, int ru)
	{
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply || isSparseSafeDivideOrPow(op, m2));

		int clen = m1.clen;
		SparseBlock a = m1.sparseBlock;
		if(ret.isInSparseFormat()){
			SparseBlock sb = ret.getSparseBlock();
			for(int i = rl; i < ru; i++) {
				if(skipEmpty && (a == null || a.isEmpty(i)))
					continue; // skip empty rows
				if(skipEmpty && ret.sparse)
					ret.sparseBlock.allocate(i, a.size(i));
				int lastIx = -1;
				if(a != null && !a.isEmpty(i)) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for(int j = apos; j < apos + alen; j++) {
						// empty left
						fillZeroValues(op, m2, ret, skipEmpty, i, lastIx + 1, aix[j]);
						// actual value
						double v2 = m2.get(0, aix[j]);
						double v = op.fn.execute(avals[j], v2);
						sb.append(i, aix[j], v);
						lastIx = aix[j];
					}
				}
				// empty left
				fillZeroValues(op, m2, ret, skipEmpty, i, lastIx + 1, clen);
			}
		}
		else{
			DenseBlock db = ret.getDenseBlock();
			for(int i = rl; i < ru; i++){
				if(skipEmpty && (a == null || a.isEmpty(i)))
					continue; // skip empty rows
				if(skipEmpty && ret.sparse)
					ret.sparseBlock.allocate(i, a.size(i));
				int lastIx = -1;
				if(a != null && !a.isEmpty(i)) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for(int j = apos; j < apos + alen; j++) {
						// empty left
						fillZeroValues(op, m2, db, skipEmpty, i, lastIx + 1, aix[j]);
						// actual value
						double v2 = m2.get(0, aix[j]);
						double v = op.fn.execute(avals[j], v2);
						db.set(i, aix[j], v);
						lastIx = aix[j];
					}
				}
				// empty left
				fillZeroValues(op, m2, db, skipEmpty, i, lastIx + 1, clen);
			}
		}
	}
	
	
	private static void fillZeroValues(BinaryOperator op, MatrixBlock m2, MatrixBlock ret, boolean skipEmpty, int rpos,
		int cpos, int len) {
		if(skipEmpty)
			return;
		else if(m2.isEmpty()) 
			fillZeroValuesEmpty(op, m2, ret, skipEmpty, rpos, cpos, len);
		else if(m2.isInSparseFormat()) 
			fillZeroValuesSparse(op, m2, ret, skipEmpty, rpos, cpos, len);
		else 
			fillZeroValuesDense(op, m2, ret, skipEmpty, rpos, cpos, len);
	}

	private static void fillZeroValues(BinaryOperator op, MatrixBlock m2, DenseBlock ret, boolean skipEmpty, int rpos,
		int cpos, int len) {
		if(skipEmpty)
			return;
		else if(m2.isEmpty()) 
			fillZeroValuesEmpty(op, m2, ret, skipEmpty, rpos, cpos, len);
		else if(m2.isInSparseFormat()) 
			fillZeroValuesSparse(op, m2, ret, skipEmpty, rpos, cpos, len);
		else 
			fillZeroValuesDense(op, m2, ret, skipEmpty, rpos, cpos, len);
	}


	private static void fillZeroValuesEmpty(BinaryOperator op, MatrixBlock m2, MatrixBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {
		final double zero = op.fn.execute(0.0, 0.0);
		final boolean zeroIsZero = zero == 0.0;
		if(!zeroIsZero) {
			throw new RuntimeException("invalid safe fill");
			// while(cpos < len)
				// TODO change this to a fill operation.
				// ret.appendValue(rpos, cpos++, zero);
		}
	}

	private static void fillZeroValuesEmpty(BinaryOperator op, MatrixBlock m2, DenseBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {
		final double zero = op.fn.execute(0.0, 0.0);
		final boolean zeroIsZero = zero == 0.0;
		if(!zeroIsZero)
			ret.set(rpos, rpos+1, cpos, len, zero);
		
	}


	private static void fillZeroValuesDense(BinaryOperator op, MatrixBlock m2, MatrixBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {
		final DenseBlock db = m2.getDenseBlock();
		final double[] vals = db.values(0);
		final SparseBlock r = ret.getSparseBlock();
		if(ret.isInSparseFormat() && r instanceof SparseBlockMCSR) {
			SparseBlockMCSR mCSR = (SparseBlockMCSR) r;
			mCSR.allocate(rpos, cpos, len);
			SparseRow sr = mCSR.get(rpos);
			for(int k = cpos; k < len; k++) {
				sr.append(k, op.fn.execute(0, vals[k]));
			}
		}
		else {
			// def
			for(int k = cpos; k < len; k++) {
				ret.appendValue(rpos, k, op.fn.execute(0, vals[k]));
			}
		}
	}

	private static void fillZeroValuesDense(BinaryOperator op, MatrixBlock m2, DenseBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {
		final DenseBlock db = m2.getDenseBlock();
		final double[] vals = db.values(0);
		for(int k = cpos; k < len; k++) 
			ret.set(rpos, k, op.fn.execute(0, vals[k]));
		
	}

	private static void fillZeroValuesSparse(BinaryOperator op, MatrixBlock m2, MatrixBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {

		final double zero = op.fn.execute(0.0, 0.0);
		final boolean zeroIsZero = zero == 0.0;
		final SparseBlock sb = m2.getSparseBlock();
		if(sb.isEmpty(0)) {
			if(!zeroIsZero) {
				throw new RuntimeException("invalid fill zeros");
			// 	while(cpos < len)
			// 		ret.appendValue(rpos, cpos++, zero);
			}
		}
		else {
			int apos = sb.pos(0);
			final int alen = sb.size(0) + apos;
			final int[] aix = sb.indexes(0);
			final double[] vals = sb.values(0);
			// skip aix pos until inside range of cpos and len
			while(apos < alen && aix[apos] < len && cpos > aix[apos]) {
				apos++;
			}
			// for each point in the sparse range
			for(; apos < alen && aix[apos] < len; apos++) {
				if(!zeroIsZero) {
					throw new RuntimeException("invalid fill zeros");
					// while(cpos < len && cpos < aix[apos]) {
					// 	ret.appendValue(rpos, cpos++, zero);
					// }
				}
				cpos = aix[apos];
				final double v = op.fn.execute(0, vals[apos]);
				ret.appendValue(rpos, aix[apos], v);
				// cpos++;
			}
			// process tail.
			if(!zeroIsZero) {
				throw new RuntimeException("invalid fill zeros");
				// while(cpos < len) {
				// 	ret.appendValue(rpos, cpos++, zero);
				// }
			}
		}
	}


	private static void fillZeroValuesSparse(BinaryOperator op, MatrixBlock m2, DenseBlock ret, boolean skipEmpty,
		int rpos, int cpos, int len) {

		final double zero = op.fn.execute(0.0, 0.0);
		final boolean zeroIsZero = zero == 0.0;
		final SparseBlock sb = m2.getSparseBlock();
		if(sb.isEmpty(0)) {
			if(!zeroIsZero) {
				throw new RuntimeException("invalid fill zeros");
				// while(cpos < len)
				// 	ret.set(rpos, cpos++, zero);
			}
		}
		else {
			int apos = sb.pos(0);
			final int alen = sb.size(0) + apos;
			final int[] aix = sb.indexes(0);
			final double[] vals = sb.values(0);
			// skip aix pos until inside range of cpos and len
			while(apos < alen && aix[apos] < len && cpos > aix[apos]) {
				apos++;
			}
			// for each point in the sparse range
			for(; apos < alen && aix[apos] < len; apos++) {
				if(!zeroIsZero) {
					throw new RuntimeException("invalid fill zeros");
					// while(cpos < len && cpos < aix[apos]) {
					// 	ret.set(rpos, cpos++, zero);
					// }
				}
				cpos = aix[apos];
				final double v = op.fn.execute(0, vals[apos]);
				ret.set(rpos, aix[apos], v);
			}
			// process tail.
			if(!zeroIsZero) {
				throw new RuntimeException("invalid fill zeros");
				// while(cpos < len)
				// 	ret.set(rpos, cpos++, zero);
			}
		}
	}
	
	private static boolean isSafeBinaryMcVDenseSparseMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		return op.sparseSafe && !m1.sparse && !m2.sparse && ret.sparse
			&& op.fn instanceof Multiply
			&& atype == BinaryAccessType.MATRIX_COL_VECTOR
			&& (long)m1.rlen * m2.clen < Integer.MAX_VALUE;
	}

	private static void safeBinaryMcVDenseSparseMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		int rlen = m1.rlen;
		int clen = m1.clen;
		double[] a = m1.getDenseBlockValues();
		double[] b = m2.getDenseBlockValues();
		
		//note: invocation condition ensures max int nnz
		//count output nnz (for CSR preallocation)
		int nnz = 0;
		for(int i=0, aix=0; i<rlen; i++, aix+=clen)
			nnz += (b[i] != 0) ? UtilFunctions
				.countNonZeros(a, aix, clen) : 0;
		//allocate and compute output in CSR format
		int[] rptr = new int[rlen+1];
		int[] indexes = new int[nnz];
		double[] vals = new double[nnz];
		rptr[0] = 0;
		for( int i=0, aix=0, pos=0; i<rlen; i++, aix+=clen ) {
			double bval = b[i];
			if( bval != 0 ) {
				for( int j=0; j<clen; j++ ) {
					double aval = a[aix+j];
					if( aval == 0 ) continue;
					indexes[pos] = j;
					vals[pos] = aval * bval;
					pos++;
				}
			}
			rptr[i+1] = pos;
		}
		ret.sparseBlock = new SparseBlockCSR(
			rptr, indexes, vals, nnz);
		ret.setNonZeros(nnz);
	}
	
	private static long safeBinaryMVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru) 
	{
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);

		long nnz = 0;
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=rl; i<ru; i++ ){
				nnz += safeBinaryMcVGenericRow(m1, m2, ret, op, isMultiply, skipEmpty, clen, i);
			}
		}
		else // if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			//if the right hand side row vector is sparse we have to exploit that;
			//otherwise, both sparse access (binary search) and asymtotic behavior
			//in the number of cells become major bottlenecks
			if( m2.sparse && ret.sparse && isMultiply ) //SPARSE *
			{
				//note: sparse block guaranteed to be allocated (otherwise early abort)
				SparseBlock b = m2.sparseBlock;
				SparseBlock c = ret.sparseBlock;
				if( b.isEmpty(0) ) return 0; 
				int blen = b.size(0); //always pos 0
				int[] bix = b.indexes(0);
				double[] bvals = b.values(0);
				for( int i=rl; i<ru; i++ ) {
					c.allocate(i, blen);
					for( int j=0; j<blen; j++ )
						c.append(i, bix[j], m1.get(i, bix[j]) * bvals[j]);
				}
				ret.setNonZeros(c.size());
			}
			else //GENERAL CASE
			{
				nnz = safeBinaryMrVGeneric(m1, m2, ret, op, clen, rl, ru);
			}
		}
		
		return nnz;
	}

	private static long safeBinaryMrVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, 
		BinaryOperator op, int clen, int rl, int ru)
	{
		long nnz = 0;
		if(ret.isInSparseFormat()){
			SparseBlock sb = ret.getSparseBlock();
			for(int i = rl; i < ru; i++) {
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					double v2 = m2.get(0, j); // replicated vector value
					double v = op.fn.execute(v1, v2);
					sb.append(i, j, v);
				}
				nnz += sb.size(i);
			}
		}
		else{
			DenseBlock db = ret.getDenseBlock();
			for(int i = rl; i < ru; i++) {
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					double v2 = m2.get(0, j); // replicated vector value
					double v = op.fn.execute(v1, v2);
					db.set(i, j, v);
					nnz += v != 0 ? 1 : 0;
				}
			}
		}
		return nnz;
	}

	private static long safeBinaryMcVGenericRow(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		boolean isMultiply, boolean skipEmpty, int clen, int i) {
		// replicate vector value
		double v2 = m2.get(i, 0);
		long nnz = 0;
		if(skipEmpty && v2 == 0)
			return 0;
		if(ret.isInSparseFormat()){
			SparseBlock sb = ret.getSparseBlock();
			if(isMultiply && v2 == 1) // ROW COPY
			{
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					sb.append(i, j, v1);
				}
			}
			else // GENERAL CASE
			{
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					double v = op.fn.execute(v1, v2);
					sb.append(i, j, v);
				}
			}
			nnz += sb.size(i);
		}
		else{
			DenseBlock db = ret.getDenseBlock();
			if(isMultiply && v2 == 1) // ROW COPY
			{
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					db.set(i, j, v1);
				}
			}
			else // GENERAL CASE
			{
				for(int j = 0; j < clen; j++) {
					double v1 = m1.get(i, j);
					double v = op.fn.execute(v1, v2);
					db.set(i, j, v);
				}
			}
			nnz += ret.recomputeNonZeros(i, i);
		}
		return nnz;
	}
	
	private static long safeBinaryVVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru) {
		int clen = m2.clen;
		
		long nnz = 0;
		if(LibMatrixOuterAgg.isCompareOperator(op) 
			&& m2.getNumColumns()>16 && SortUtils.isSorted(m2) ) {
			return performBinOuterOperation(m1, m2, ret, op, rl, ru);
		}
		else if(ret.isInSparseFormat()) {
			SparseBlock sb = ret.getSparseBlock();
			for(int r=rl; r<ru; r++) {
				double v1 = m1.get(r, 0);
				for(int c=0; c<clen; c++) {
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					sb.append(r, c, v);
					nnz += v== 0.0 ? 0:1;
				}
			}
		}
		else {
			DenseBlock db = ret.getDenseBlock();
			for(int r=rl; r<ru; r++) {
				double v1 = m1.get(r, 0);
				for(int c=0; c<clen; c++) {
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					db.set(r, c, v);
					nnz += v== 0.0 ? 0:1;
				}
			}
		}
		return nnz;
		//no need to recomputeNonZeros since maintained in append value
	}
	
	private static long safeBinaryMMSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		int rl, int ru) {
		// both sparse blocks existing
		if(m1.sparseBlock != null && m2.sparseBlock != null) {
			SparseBlock lsblock = m1.sparseBlock;
			SparseBlock rsblock = m2.sparseBlock;
			if(ret.sparse && lsblock.isAligned(rsblock))
				return safeBinaryMMSparseSparseAligned(m1, m2, ret, op, rl, ru);
			else // general case
				return safeBinaryMMSparseSparseGeneric(m1, m2, ret, op, rl, ru);
		}
		else if(m2.sparseBlock != null)
			return safeBinaryMMSparseSparseNullRight(m1, m2, ret, op, rl, ru);
		else
			return safeBinaryMMSparseSparseNullLeft(m1, m2, ret, op, rl, ru);

	}

	private static long safeBinaryMMSparseSparseAligned(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru) {
		SparseBlock lsblock = m1.sparseBlock;
		SparseBlock rsblock = m2.sparseBlock;
		long lnnz = 0;
		SparseBlock c = ret.sparseBlock;
		for(int r = rl; r < ru; r++)
			if(!lsblock.isEmpty(r)) {
				int alen = lsblock.size(r);
				int apos = lsblock.pos(r);
				int[] aix = lsblock.indexes(r);
				double[] avals = lsblock.values(r);
				double[] bvals = rsblock.values(r);
				c.allocate(r, alen);
				for(int j = apos; j < apos + alen; j++) {
					double tmp = op.fn.execute(avals[j], bvals[j]);
					c.append(r, aix[j], tmp);
				}
				lnnz += c.size(r);
			}
		return lnnz;
	}

	private static long safeBinaryMMSparseSparseGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru) {
		SparseBlock lsblock = m1.sparseBlock;
		SparseBlock rsblock = m2.sparseBlock;
		long lnnz = 0;
		for(int r = rl; r < ru; r++) {
			final boolean emptyRowLeft = lsblock.isEmpty(r);
			final boolean emptyRowRight = rsblock.isEmpty(r);
			if(!emptyRowLeft) {
				final double[] lbV = lsblock.values(r);
				final int[] lbI = lsblock.indexes(r);
				final int lbP = lsblock.pos(r);
				final int lbS = lsblock.size(r);
				if(!emptyRowRight) {
					final double[] rbV = rsblock.values(r);
					final int[] rbI = rsblock.indexes(r);
					final int rbP = rsblock.pos(r);
					final int rbS = rsblock.size(r);
					lnnz += mergeForSparseBinary(op, lbV, lbI, lbP, lbS, rbV, rbI, rbP, rbS, r, ret);
				}
				else {
					lnnz += appendLeftForSparseBinary(op, lbV, lbI, lbP, lbS, r, ret);
				}
			}
			else if(!emptyRowRight) {
				final double[] rbV = rsblock.values(r);
				final int[] rbI = rsblock.indexes(r);
				final int rbP = rsblock.pos(r);
				final int rbS = rsblock.size(r);
				lnnz += appendRightForSparseBinary(op, rbV, rbI, rbP, rbS, r, ret);
			}
			
			// do nothing if both not existing
		}
		return lnnz;
	}


	private static long safeBinaryMMSparseSparseNullRight(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru) {
		long lnnz = 0;
		SparseBlock rsblock = m2.sparseBlock;
		for(int r = rl; r < ru; r++) {
			if(rsblock.isEmpty(r))
				continue;
			lnnz += appendRightForSparseBinary(op, rsblock.values(r), rsblock.indexes(r), rsblock.pos(r), rsblock.size(r), r,
				ret);
		}
		return lnnz;
	}

	private static long safeBinaryMMSparseSparseNullLeft(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru) {
		long lnnz = 0;
		SparseBlock lsblock = m1.sparseBlock;
		for(int r = rl; r < ru; r++) {
			if(lsblock.isEmpty(r))
				continue;
			lnnz += appendLeftForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), lsblock.pos(r), lsblock.size(r), r,
				ret);
		}
		return lnnz;
	}
	
	private static long safeBinaryMMSparseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru){
		//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
		final int n = ret.clen;
		
		DenseBlock dc = ret.getDenseBlock(); // dense output
		//1) process left input: assignment
		if(!m1.isEmpty()){
			if( m1.isInSparseFormat()) //SPARSE left 
				safeMMLSparsePreProcess(m1, rl, ru, dc);
			else   //DENSE left
				safeMMLDensePreProcess(m1, rl, ru, n, dc);
		}
		
		//2) process right input: op.fn (+,-,*), * only if dense
		long lnnz = 0;
		if(m2.isEmpty()){
			return ret.recomputeNonZeros(rl, ru-1);
		}
		else if( m2.isInSparseFormat() ) //SPARSE right
		{
			lnnz = safeMMRSparsePostProcess(m2, ret, op, rl, ru, dc, lnnz);
		}
		else  //DENSE right
		{
			lnnz = safeMMRDensePostProcess(m1, m2, ret, op, rl, ru, n, dc, lnnz);
		}
		
		//3) recompute nnz
		return lnnz;
	}

	private static long safeMMRDensePostProcess(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru,
		final int n, DenseBlock dc, long lnnz) {
		if( !m2.isEmptyBlock(false) ) {
			DenseBlock da = m2.getDenseBlock();
			for( int i=rl; i<ru; i++ ) {
				double[] a = da.values(i);
				double[] c = dc.values(i);
				int apos = da.pos(i);
				for( int j = apos; j<apos+n; j++ ) {
					c[j] = op.fn.execute(c[j], a[j]);
					// lnnz += (c[j]!=0) ? 1 : 0;
				}
				lnnz += ret.recomputeNonZeros(i, i);
			}
		}
		else if(op.fn instanceof Multiply)
			ret.denseBlock.set(0);
		else
			lnnz = m1.nonZeros;
		return lnnz;
	}

	private static long safeMMRSparsePostProcess(MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru, DenseBlock dc,
		long lnnz) {
		SparseBlock a = m2.sparseBlock;
		for(int i=rl; i<ru; i++) {
			double[] c = dc.values(i);
			int cpos = dc.pos(i);
			if( !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				for(int k = apos; k < apos+alen; k++) 
					c[cpos+aix[k]] = op.fn.execute(c[cpos+aix[k]], avals[k]);
			}
			//exploit temporal locality of rows
			lnnz += ret.recomputeNonZeros(i, i);
		}
		return lnnz;
	}

	private static void safeMMLDensePreProcess(MatrixBlock m1, int rl, int ru, final int n, DenseBlock dc) {
		if( !m1.isEmptyBlock(false) ) {
			int rlbix = dc.index(rl);
			int rubix = dc.index(ru-1);
			final DenseBlock da = m1.getDenseBlock();
			if( rlbix == rubix )
				System.arraycopy(da.valuesAt(rlbix), da.pos(rl), dc.valuesAt(rlbix), dc.pos(rl), (ru-rl)*n);
			else {
				for(int i=rl; i<ru; i++)
					System.arraycopy(da.values(i), da.pos(i), dc.values(i), dc.pos(i), n);
			}
		}
		else
			dc.set(0);
	}

	private static void safeMMLSparsePreProcess(MatrixBlock m1, int rl, int ru, DenseBlock dc) {
		final SparseBlock a = m1.getSparseBlock();
		for(int i=rl; i<ru; i++) {
			double[] c = dc.values(i);
			int cpos = dc.pos(i);
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			for(int k = apos; k < apos+alen; k++) 
				c[cpos+aix[k]] = avals[k];
		}
	}
	
	private static long safeBinaryMMDenseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru){
		final int clen = m1.clen;
		final boolean isPM = (op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		
		final DenseBlock da = m1.getDenseBlock();
		final DenseBlock db = m2.getDenseBlock();
		final DenseBlock dc = ret.getDenseBlock();
		
		if(isPM && clen >= 64)
			return safeBinaryMMDenseDenseDensePM_Vec(da, db, dc, op, rl, ru, clen);
		else if(da.isContiguous() && db.isContiguous() && dc.isContiguous()) {
			if(op.fn instanceof PlusMultiply)
				return safeBinaryMMDenseDenseDensePM(da, db, dc, op, rl, ru, clen);
			else
				return safeBinaryMMDenseDenseDenseContiguous(m1,m2,ret, op, rl, ru, clen);
		}
		else
			return safeBinaryMMDenseDenseDenseGeneric(da, db, dc, op, rl, ru, clen);
	}

	private static final long safeBinaryMMDenseDenseDensePM_Vec(DenseBlock da, DenseBlock db, DenseBlock dc, BinaryOperator op,
		int rl, int ru, int clen) {
		final double cntPM = (op.fn instanceof PlusMultiply ? ((PlusMultiply) op.fn).getConstant() : -1d *
			((MinusMultiply) op.fn).getConstant());
		long lnnz = 0;
		for(int i = rl; i < ru; i++) {
			final double[] a = da.values(i);
			final double[] b = db.values(i);
			final double[] c = dc.values(i);
			int pos = da.pos(i);
			System.arraycopy(a, pos, c, pos, clen);
			LibMatrixMult.vectMultiplyAdd(cntPM, b, c, pos, pos, clen);
			lnnz += UtilFunctions.computeNnz(c, pos, clen);
		}
		return lnnz;
	}

	private static final long safeBinaryMMDenseDenseDensePM(DenseBlock da, DenseBlock db, DenseBlock dc, BinaryOperator op,
		int rl, int ru, int clen) {
		long lnnz = 0;
		final double[] a = da.values(0);
		final double[] b = db.values(0);
		final double[] c = dc.values(0);
		final double d = ((PlusMultiply)op.fn).getConstant();
		for(int i = da.pos(rl); i < da.pos(ru); i++) {
			c[i] = a[i] + d * b[i];
			lnnz += (c[i] != 0) ? 1 : 0;
		}
		return lnnz;
	}

	private static final long safeBinaryMMDenseDenseDenseContiguous(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		int rl, int ru, int clen) {
	
		final DenseBlock da = m1.getDenseBlock();
		final DenseBlock db = m2.getDenseBlock();
		final DenseBlock dc = ret.getDenseBlock();
		final double[] a = da.values(0);
		final double[] b = db.values(0);
		final double[] c = dc.values(0);
		final int end = da.pos(ru);
		if(m1.getSparsity() == 1 && m2.getSparsity() == 1 && op.fn instanceof Multiply) {
			safeBinaryMMDDDCMult(a, b, c, da.pos(rl), end);
			return (long) m1.rlen * m1.clen;
		}
		else{
			return safeBinaryMMDDDCG(op, da, a, b, c, da.pos(rl), end);
		}
	}

	private static long safeBinaryMMDDDCG(BinaryOperator op,  final DenseBlock da, final double[] a,
		final double[] b, final double[] c, final int start, final int end) {
		long lnnz = 0;
		for(int i = start; i < end; i++) {
			c[i] = op.fn.execute(a[i], b[i]);
			lnnz += (c[i] != 0) ? 1 : 0;
		}
		return lnnz;
	}

	private static void safeBinaryMMDDDCMult(final double[] a, final double[] b, final double[] c, final int start,
		final int end) {
		final int h = (end - start) % 8;

		for(int i = start; i < start + h; i++)
			c[i] = a[i] * b[i];
		for(int i = start + h; i < end; i += 8) {
			by8(a, b, c, i);
		}
	}

	private static void by8(final double[] a, final double[] b, final double[] c, int i) {
		c[i] = a[i] * b[i];
		c[i + 1] = a[i + 1] * b[i + 1];
		c[i + 2] = a[i + 2] * b[i + 2];
		c[i + 3] = a[i + 3] * b[i + 3];
		c[i + 4] = a[i + 4] * b[i + 4];
		c[i + 5] = a[i + 5] * b[i + 5];
		c[i + 6] = a[i + 6] * b[i + 6];
		c[i + 7] = a[i + 7] * b[i + 7];
	}

	private static final long safeBinaryMMDenseDenseDenseGeneric(DenseBlock da, DenseBlock db, DenseBlock dc,
		BinaryOperator op, int rl, int ru, int clen) {
		final ValueFunction fn = op.fn;
		long lnnz = 0;
		for(int i = rl; i < ru; i++) {
			final double[] a = da.values(i);
			final double[] b = db.values(i);
			final double[] c = dc.values(i);
			int pos = da.pos(i);
			for(int j = pos; j < pos + clen; j++) {
				c[j] = fn.execute(a[j], b[j]);
				lnnz += (c[j] != 0) ? 1 : 0;
			}
		}
		return lnnz;
	}
	
	private static long safeBinaryMMSparseDenseSkip(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		final BinaryOperator op, int rl, int ru)
	{
		// this specialization does not care about which side is sparse.
		// any side will work.

		final SparseBlock a = m1.sparse ? m1.sparseBlock : m2.sparseBlock;
		// guaranteed to be not null otherwise we would have early aborted earlier
		
		//prepare second input and allocate output
		MatrixBlock b = m1.sparse ? m2 : m1;
		
		final boolean left = a == m1.sparseBlock;

		long lnnz = 0;
		for(int i = rl; i < Math.min(ru, a.numRows()); i++) {
			if(a.isEmpty(i))
				continue;
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			lnnz += safeMMSparseDenseSkipRow(ret, op, b, left, i, apos, alen, aix, avals);
		}
		return lnnz;
	}

	private static long safeMMSparseDenseSkipRow(MatrixBlock ret, final BinaryOperator op, MatrixBlock b,
		final boolean left, int i, int apos, int alen, int[] aix, double[] avals) {
		if(left) {
			if(ret.sparse)
				return safeMMSparseDenseSkipRowLeftSparseRet(op, b, i, apos, alen, aix, avals, ret.getSparseBlock());
			else
				return safeMMSparseDenseSkipRowLeftDenseRet(op, b, i, apos, alen, aix, avals, ret.getDenseBlock());
		}
		else {
			if(ret.sparse)
				return safeMMSparseDenseSkipRowRightSparseRet(op, b, i, apos, alen, aix, avals, ret.getSparseBlock());
			else
				return safeMMSparseDenseSkipRowRightDenseRet(op, b, i, apos, alen, aix, avals, ret.getDenseBlock());
		}
	}

	private static long safeMMSparseDenseSkipRowLeftDenseRet(final BinaryOperator op, MatrixBlock b, int i, int apos,
		int alen, int[] aix, double[] avals, final DenseBlock db) {
		long lnnz = 0;
		for(int k = apos; k < apos + alen; k++) {
			final double in2 = b.get(i, aix[k]);
			double val = op.fn.execute(avals[k], in2);
			lnnz += (val != 0) ? 1 : 0;
			db.set(i, aix[k], val);
		}
		return lnnz;
	}

	private static long safeMMSparseDenseSkipRowLeftSparseRet(final BinaryOperator op, final MatrixBlock b, final int i,
		final int apos, final int alen, final int[] aix, final double[] avals, final SparseBlock sb) {
		if(!b.sparse)
			sb.allocate(i, alen);
		for(int k = apos; k < apos + alen; k++) {
			final int idx = aix[k];
			sb.append(i, idx, op.fn.execute(avals[k], b.get(i, idx)));
		}
		return sb.size(i);
	}

	private static long safeMMSparseDenseSkipRowRightDenseRet(final BinaryOperator op, MatrixBlock b, int i, int apos,
		int alen, int[] aix, double[] avals, final DenseBlock db) {
		long lnnz = 0;
		for(int k = apos; k < apos + alen; k++) {
			final double in2 = b.get(i, aix[k]);
			double val = op.fn.execute(in2, avals[k]);
			lnnz += (val != 0) ? 1 : 0;
			db.set(i, aix[k], val);
		}
		return lnnz;
	}

	private static long safeMMSparseDenseSkipRowRightSparseRet(final BinaryOperator op, MatrixBlock b, int i, int apos,
		int alen, int[] aix, double[] avals, final SparseBlock sb) {
		if(!b.sparse)
			sb.allocate(i, alen);
		for(int k = apos; k < apos + alen; k++) {
			final double in2 = b.get(i, aix[k]);
			double val = op.fn.execute(in2, avals[k]);
			sb.append(i, aix[k], val);
		}
		return sb.size(i);
	}

	
	private static long safeBinaryMMGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl,
		int ru) {
		int clen = m2.clen;
		long lnnz = 0;
		if(ret.isInSparseFormat()){
			SparseBlock sb = ret.getSparseBlock();
			for(int r = rl; r < ru; r++){
				sb.allocate(r);
				for(int c = 0; c < clen; c++) {
					double in1 = m1.get(r, c);
					double in2 = m2.get(r, c);
					if(in1 == 0 && in2 == 0)
						continue;
					double val = op.fn.execute(in1, in2);
					lnnz += (val != 0) ? 1 : 0;
					sb.append(r, c, val);
				}
			}
		}
		else {
			DenseBlock db = ret.getDenseBlock();
			for(int r = rl; r < ru; r++){
				for(int c = 0; c < clen; c++) {
					double in1 = m1.get(r, c);
					double in2 = m2.get(r, c);
					if(in1 == 0 && in2 == 0)
						continue;
					double val = op.fn.execute(in1, in2);
					lnnz += (val != 0) ? 1 : 0;
					db.set(r, c, val);
				}
			}
		}
		return lnnz;
	}
	
	/**
	 * 
	 * This will do cell wise operation for &lt;, &lt;=, &gt;, &gt;=, == and != operators.
	 * 
	 * @param m1 left matrix
	 * @param m2 right matrix
	 * @param ret output matrix
	 * @param bOp binary operator
	 * 
	 */
	private static long performBinOuterOperation(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator bOp, int rl, int ru) {
		int clen = ret.clen;
		double b[] = DataConverter.convertToDoubleVector(m2);
		DenseBlock dc = ret.getDenseBlock();
		
		//pre-materialize various types used in inner loop
		boolean scanType1 = (bOp.fn instanceof LessThan || bOp.fn instanceof Equals 
			|| bOp.fn instanceof NotEquals || bOp.fn instanceof GreaterThanEquals);
		boolean scanType2 = (bOp.fn instanceof LessThanEquals || bOp.fn instanceof Equals 
			|| bOp.fn instanceof NotEquals || bOp.fn instanceof GreaterThan);
		boolean lt = (bOp.fn instanceof LessThan), lte = (bOp.fn instanceof LessThanEquals);
		boolean gt = (bOp.fn instanceof GreaterThan), gte = (bOp.fn instanceof GreaterThanEquals);
		boolean eqNeq = (bOp.fn instanceof Equals || bOp.fn instanceof NotEquals);
		
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			double[] cvals = dc.values(i);
			int pos = dc.pos(i);
			double value = m1.get(i, 0);
			int ixPos1 = Arrays.binarySearch(b, value);
			int ixPos2 = ixPos1;
			if( ixPos1 >= 0 ) { //match, scan to next val
				if(scanType1) while( ixPos1<b.length && value==b[ixPos1]  ) ixPos1++;
				if(scanType2) while( ixPos2 > 0 && value==b[ixPos2-1]) --ixPos2;
			} 
			else
				ixPos2 = ixPos1 = Math.abs(ixPos1) - 1;
			int start = lt ? ixPos1 : (lte||eqNeq) ? ixPos2 : 0;
			int end = gt ? ixPos2 : (gte||eqNeq) ? ixPos1 : clen;
			
			if (bOp.fn instanceof NotEquals) {
				Arrays.fill(cvals, pos, pos+start, 1.0);
				Arrays.fill(cvals, pos+end, pos+clen, 1.0);
				lnnz += (start+(clen-end));
			}
			else if( start < end ) {
				Arrays.fill(cvals, pos+start, pos+end, 1.0);
				lnnz += (end-start);
			}
		}
		return lnnz;
	}

	private static long unsafeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru) {
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
	
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )  //MATRIX - COL_VECTOR
			return unsafeBinaryMcV(m1, m2, ret, op, rl, ru, clen);
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )  //MATRIX - ROW_VECTOR
			return unsafeBinaryMrV(m1, m2, ret, op, rl, ru, clen);
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR )  //VECTOR - VECTOR
			return unsafeBinaryVoV(m1, m2, ret, op, rl, ru);
		else // MATRIX - MATRIX
			return unsafeBinaryMM(m1, m2, ret, op, rl, ru, clen);
	}

	private static long unsafeBinaryMM(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru,
		int clen) {
		long lnnz = 0;
		//dense non-empty vectors (always single block)
		if( m1.clen==1 && !m1.sparse && !m1.isEmptyBlock(false)
			&& !m2.sparse && !m2.isEmptyBlock(false)  )
		{
			ret.allocateDenseBlock();
			double[] a = m1.getDenseBlockValues();
			double[] b = m2.getDenseBlockValues();
			double[] c = ret.getDenseBlockValues();
			for( int i=rl; i<ru; i++ ) {
				c[i] = op.fn.execute( a[i], b[i] );
				lnnz += (c[i] != 0) ? 1 : 0;
			}
		}
		else if(!ret.isInSparseFormat()){ // dense
			ret.allocateDenseBlock();
			DenseBlock db = ret.getDenseBlock();
			for(int r=rl; r<ru; r++){
				for(int c=0; c<clen; c++) {
					double v1 = m1.get(r, c);
					double v2 = m2.get(r, c);
					double v = op.fn.execute( v1, v2 );
					db.set(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
			}
		}
		else { // sparse
			ret.allocateSparseRowsBlock();
			SparseBlock sb = ret.getSparseBlock();
			for(int r=rl; r<ru; r++){
				for(int c=0; c<clen; c++) {
					double v1 = m1.get(r, c);
					double v2 = m2.get(r, c);
					double v = op.fn.execute( v1, v2 );
					sb.append(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
			}
		}
		return lnnz;
	}

	private static long unsafeBinaryVoV(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl,
		int ru) {
		long lnnz = 0;
		int clen2 = m2.clen; 
		if(LibMatrixOuterAgg.isCompareOperator(op) 
			&& m2.getNumColumns()>16 && SortUtils.isSorted(m2)) {
			lnnz = performBinOuterOperation(m1, m2, ret, op, rl, ru);
		} 
		else if(ret.isInSparseFormat()){
			ret.allocateSparseRowsBlock();
			SparseBlock sb = ret.getSparseBlock();
			for(int r=rl; r<ru; r++) {
				double v1 = m1.get(r, 0);
				for(int c=0; c<clen2; c++) {
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					lnnz += (v != 0) ? 1 : 0;
					sb.append(r, c, v);
				}
			}
		}
		else {
			ret.allocateDenseBlock();
			DenseBlock db = ret.getDenseBlock();
			for(int r=rl; r<ru; r++) {
				double v1 = m1.get(r, 0);
				for(int c=0; c<clen2; c++) {
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					lnnz += (v != 0) ? 1 : 0;
					db.set(r, c, v);
				}
			}
		}
		return lnnz;
	}

	private static long unsafeBinaryMrV(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl,
		int ru, int clen) {
			long lnnz = 0;
		if(ret.isInSparseFormat()){
			ret.allocateSparseRowsBlock();
			SparseBlock sb = ret.getSparseBlock();
			for(int r=rl; r<ru; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1.get(r, c);
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					sb.append(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
		}
		else{
			ret.allocateDenseBlock();
			DenseBlock db = ret.getDenseBlock();
			for(int r=rl; r<ru; r++){
				for(int c=0; c<clen; c++) {
					double v1 = m1.get(r, c);
					double v2 = m2.get(0, c);
					double v = op.fn.execute( v1, v2 );
					db.set(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
			}
		}
		return lnnz;
	}

	private static long unsafeBinaryMcV(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl,
		int ru, int clen) {
		long lnnz = 0;
		if(ret.isInSparseFormat()) {
			ret.allocateSparseRowsBlock();
			SparseBlock sb = ret.getSparseBlock();
			for(int r = rl; r < ru; r++) {
				double v2 = m2.get(r, 0);
				for(int c = 0; c < clen; c++) {
					double v1 = m1.get(r, c);
					double v = op.fn.execute(v1, v2);
					sb.append(r, c, v);
					lnnz += (v != 0) ? 1 : 0;
				}
			}
		}
		else {
			ret.allocateDenseBlock();
			DenseBlock db = ret.getDenseBlock();
			for(int r = rl; r < ru; r++) {
				double v2 = m2.get(r, 0);
				for(int c = 0; c < clen; c++) {
					double v1 = m1.get(r, c);
					double v = op.fn.execute(v1, v2);
					db.set(r, c, v);
					lnnz += (v != 0) ? 1 : 0;
				}
			}
		}
		return lnnz;
	}

	private static long safeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl, int ru) {
		// early abort possible since sparse safe
		if(m1.isEmptyBlock(false))
			return 0;
		else if(m1.sparse != ret.sparse)
			throw new DMLRuntimeException(
				"Unsupported safe binary scalar operations over different input/output representation: " //
					+ m1.sparse + " " + ret.sparse + "  " + op);
		else if(m1.sparse)
			return safeBinaryScalarSparseSparse(m1, ret, op, rl, ru);
		else
			return safeBinaryScalarDenseDense(m1, ret, op, rl, ru);
	}

	private static long safeBinaryScalarSparseSparse(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl,
		int ru) {
		ret.allocateSparseRowsBlock();
		final boolean copyOnes = (op.fn instanceof NotEquals && op.getConstant() == 0);
		if(copyOnes)
			return safeBinaryScalarSparseSparseCopyOnes(m1, ret, op, rl, ru);
		else{

			final boolean allocExact = op.fn instanceof Multiply //
				|| op.fn instanceof Multiply2 //
				|| op.fn instanceof Power2 //
				|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MAX) //
				|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MIN);
			if(allocExact && ret.sparseBlock instanceof SparseBlockMCSR)
				return safeBinaryScalarSparseSparseExact(m1, ret, op, rl, ru);
			else
				return safeBinaryScalarSparseSparseGeneric(m1, ret, op, rl, ru);
		}
	}

	private static long safeBinaryScalarSparseSparseExact(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl,
		int ru) {
		// allocate sparse row structure
		final SparseBlock a = m1.sparseBlock;
		final SparseBlockMCSR c = (SparseBlockMCSR) ret.sparseBlock;

		final boolean neverReturnZeroOnNonZero = op.fn instanceof Power2 //
			|| (op.fn instanceof Multiply && op.getConstant() != 0);
		long nnz = 0;
		for(int r = rl; r < ru; r++) {
			if(a.isEmpty(r))
				continue;
			final int apos = a.pos(r);
			final int alen = a.size(r);
			final int[] aix = a.indexes(r);
			final double[] avals = a.values(r);
			if(neverReturnZeroOnNonZero)
				nnz += safeBinaryScalarSparseSparseExactRowNoZero(apos, alen, aix, avals, r, c, op);
			else 
				nnz += safeBinaryScalarSparseSparseExactRow(apos, alen, aix, avals, r, c, op);
		}
		return nnz;
	}

	private static long safeBinaryScalarSparseSparseExactRow(int apos, int alen, int[] aix, double[] avals, int r,
		SparseBlockMCSR c, ScalarOperator op) {
		// create sparse row without repeated resizing for specific ops
		final int[] cix = new int[alen];
		final double[] cvals = new double[alen];

		int k = 0;
		for(int j = apos; j < apos + alen; j++) {
			double v = op.executeScalar(avals[j]);
			if(v != 0) {
				cix[k] = aix[j];
				cvals[k++] = v;
			}
		}

		SparseRowVector sr = new SparseRowVector(cvals, cix, k);
		c.set(r, sr, false);
		return k;
	}

	private static long safeBinaryScalarSparseSparseExactRowNoZero(int apos, int alen, int[] aix, double[] avals, int r,
		SparseBlockMCSR c, ScalarOperator op) {
		// create sparse row without repeated resizing for specific ops
		final int[] cix = new int[alen];
		System.arraycopy(aix, apos, cix, 0, alen);
		final double[] cvals = new double[alen];

		int k = 0;
		for(int j = apos; j < apos + alen; j++)
			cvals[k++] = op.executeScalar(avals[j]);

		SparseRowVector sr = new SparseRowVector(cvals, cix, k);
		c.set(r, sr, false);
		return k;
	}

	private static long safeBinaryScalarSparseSparseGeneric(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl,
		int ru) {
		final boolean allocExact = op.fn instanceof Multiply //
			|| op.fn instanceof Multiply2 //
			|| op.fn instanceof Power2 //
			|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MAX) //
			|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MIN);

		// allocate sparse row structure
		final SparseBlock a = m1.sparseBlock;
		final SparseBlock c = ret.sparseBlock;

		long nnz = 0;
		for(int r = rl; r < ru; r++) {
			if(a.isEmpty(r))
				continue;

			final int apos = a.pos(r);
			final int alen = a.size(r);
			final int[] aix = a.indexes(r);
			final double[] avals = a.values(r);

			// create sparse row without repeated resizing for specific ops
			if(allocExact)
				c.allocate(r, alen);

			for(int j = apos; j < apos + alen; j++) {
				double val = op.executeScalar(avals[j]);
				c.append(r, aix[j], val);
				nnz += (val != 0) ? 1 : 0;
			}
			
		}
		return nnz;
	}

	private static long safeBinaryScalarSparseSparseCopyOnes(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl,
		int ru) {

		long lnnz = 0;
		// allocate sparse row structure
		final SparseBlock a = m1.sparseBlock;
		final SparseBlock c = ret.sparseBlock;

		long nnz = 0;
		for(int r = rl; r < ru; r++) {
			if(a.isEmpty(r))
				continue;

			final int apos = a.pos(r);
			final int alen = a.size(r);
			final int[] aix = a.indexes(r);

			// create sparse row without repeated resizing
			SparseRowVector crow = new SparseRowVector(alen);
			crow.setSize(alen);

			// memcopy/memset of indexes/values (sparseblock guarantees absence of 0s)
			System.arraycopy(aix, apos, crow.indexes(), 0, alen);
			Arrays.fill(crow.values(), 0, alen, 1);
			c.set(r, crow, false);
			nnz += alen;

		}
		lnnz = (ret.nonZeros = nnz);
		return lnnz;
	}

	private static long safeBinaryScalarDenseDense(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl, int ru) {
		return denseBinaryScalar(m1, ret, op, rl, ru);
	}

	/**
	 * Since this operation is sparse-unsafe, output ret is dependent on ret.sparse, if set use sparse output, else dense.
	 * 
	 * @param m1  Input matrix
	 * @param ret Result matrix
	 * @param op  Scalar operator
	 */
	private static long unsafeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false)) {
			//compute 0 op constant once and set into dense output
			final double val = op.executeScalar(0);
			if( val != 0 )
				ret.reset(ret.rlen, ret.clen, val);
			return (val != 0) ? ret.getLength() : 0;
		}
		
		//sanity check input/output sparsity
		if( ret.sparse )
			return unsafeBinaryScalarSparseOut(m1, ret, op);
		else 
			return unsafeBinaryScalarDenseOut(m1, ret, op);
	}


	private static long unsafeBinaryScalarDenseOut(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
			
		int m = m1.rlen;
		int n = m1.clen;
		long lnnz = 0;
		
		if( m1.sparse ) //SPARSE MATRIX
		{
			ret.allocateDenseBlock();
			
			SparseBlock a = m1.sparseBlock;
			DenseBlock dc = ret.getDenseBlock();
			
			//init dense result with unsafe 0-value
			double val0 = op.executeScalar(0);
			boolean lsparseSafe = (val0 == 0);
			if( !lsparseSafe )
				dc.set(val0);
			
			//compute non-zero input values
			long nnz = lsparseSafe ? 0 : m * n;
			for(int bi=0; bi<dc.numBlocks(); bi++) {
				int blen = dc.blockSize(bi);
				double[] c = dc.valuesAt(bi);
				for(int i=bi*dc.blockSize(), cix=i*n; i<blen && i<m; i++, cix+=n) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for(int j=apos; j<apos+alen; j++) {
						double val = op.executeScalar(avals[j]);
						c[ cix+aix[j] ] = val;
						nnz += lsparseSafe ? (val!=0 ? 1 : 0) :
							(val==0 ? -1 : 0);
					}
				}
			}
			lnnz = (ret.nonZeros = nnz);
		}
		else { //DENSE MATRIX
			lnnz = denseBinaryScalar(m1, ret, op, 0, m);
			if(op.fn instanceof Multiply)
				lnnz = m1.getNonZeros(); // guaranteed to be same as input nnz
		}
		
		return lnnz;
	}

	private static long unsafeBinaryScalarSparseOut(MatrixBlock m1, MatrixBlock ret, ScalarOperator op){
		final int m = m1.rlen; 
		final int n = m1.clen;
		long lnnz = 0;
		ret.allocateSparseRowsBlock();
		SparseBlock c = ret.getSparseBlock();
		if(m1.isInSparseFormat()){
			SparseBlock a = m1.getSparseBlock();
			final double val0 = op.executeScalar(0);
			if(val0 == 0)
				throw new NotImplementedException("Not implemented unsafe binary where op returns 0 on zero");
			
			for(int r = 0; r < m; r++){
				if(a.isEmpty(r)){ // fill that row with zero result.
					for(int j = 0; j < n; j++)
						c.append(r, j, val0);
				}
				else{
					int[] aix = a.indexes(r);
					double[] aval = a.values(r);
					int apos = a.pos(r);
					int alen = apos + a.size(r);
					int j = 0;
					for(; j < n && apos < alen ; j++){
						if(aix[apos] == j){
							c.append(r, j , op.executeScalar(aval[apos]));
							apos++;
						}
						else 
							c.append(r,j, val0);
					}
					for(; j < n ; j++)
						c.append(r,j, val0);
				}

				lnnz += c.size(r);
			}
		}
		else{
			DenseBlock da = m1.getDenseBlock();
			for(int r = 0; r < m; r++){
				double[] a = da.values(r);
				int apos = da.pos(r);
				for(int j = 0; j < n; j++) {
					double v  = op.executeScalar(a[apos + j]);
					c.append(r, j, v);
				}
				lnnz += c.size(r);
			}
		}
		return lnnz;
	}

	private static long denseBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl, int ru) {
		// allocate dense block (if necessary), incl clear nnz
		ret.allocateDenseBlock(true);

		final DenseBlock da = m1.getDenseBlock();
		final DenseBlock dc = ret.getDenseBlock();
		final int clen = m1.clen;
		// compute scalar operation, incl nnz maintenance
		long nnz = 0;
		if(op.fn instanceof Multiply){
			// in case of multiply we do not need to count nnz. since they are known by the m1 input.
			if(clen == 1)
				nnz = denseBinaryScalarMultiplySingleCol(da.valuesAt(0), dc.valuesAt(0), op.getConstant(), rl, ru);
			else if(da.isContiguous())
				nnz = denseBinaryScalarMultiplyMultiColContiguous(da, dc, op.getConstant(), clen, rl, ru);
			else
				nnz = denseBinaryScalarMultiplyMultiCol(da, dc, op.getConstant(), clen, rl, ru);
		}
		else{
			if(clen == 1)
				nnz = denseBinaryScalarSingleCol(da.valuesAt(0), dc.valuesAt(0), op, rl, ru);
			else
				nnz = denseBinaryScalarMultiCol(da, dc, op, clen, rl, ru);
		}
		return nnz;
	}

	private static long denseBinaryScalarSingleCol(double[] a, double[] c, ScalarOperator op, int rl, int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) { // VECTOR
			c[i] = op.executeScalar(a[i]);
			nnz += (c[i] != 0) ? 1 : 0;
		}
		return nnz;
	}

	private static long denseBinaryScalarMultiCol(DenseBlock da, DenseBlock dc, ScalarOperator op, int clen, int rl,
		int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) {
			double[] a = da.values(i);
			double[] c = dc.values(i);
			int apos = da.pos(i), cpos = dc.pos(i);
			for(int j = 0; j < clen; j++) {
				c[cpos + j] = op.executeScalar(a[apos + j]);
				nnz += (c[cpos + j] != 0) ? 1 : 0;
			}
		}
		return nnz;
	}


	private static long denseBinaryScalarMultiplySingleCol(double[] a, double[] c,  double b, int rl, int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) // VECTOR
			if(0 != (c[i] = b * a[i]))
				nnz++;
		return nnz;
	}


	private static long denseBinaryScalarMultiplyMultiColContiguous(DenseBlock da, DenseBlock dc, double b, int clen,
		int rl, int ru) {
		final double[] a = da.values(0);
		final double[] c = dc.values(0);
		long nnz = 0;
		final int start = rl * clen;
		final int end = ru * clen;
		final int cells = end - start;

		for(int i = start; i < end - (cells % 8); i += 8) 
			nnz += unroll8Multiply(a, b, c, i);
		for(int i = end - (cells % 8); i < end; i ++) 
			if(0 != (c[i] = b * a[i]))
				nnz ++;
		return nnz;
	}

	private static long unroll8Multiply(double[] a, double b, double[] c, int i){
		long nnz = 0;
		nnz += (0 != (c[i] = b * a[i])) ? 1 : 0;
		nnz += (0 != (c[i+1] = b * a[i+1])) ? 1 : 0;
		nnz += (0 != (c[i+2] = b * a[i+2])) ? 1 : 0;
		nnz += (0 != (c[i+3] = b * a[i+3])) ? 1 : 0;
		nnz += (0 != (c[i+4] = b * a[i+4])) ? 1 : 0;
		nnz += (0 != (c[i+5] = b * a[i+5])) ? 1 : 0;
		nnz += (0 != (c[i+6] = b * a[i+6])) ? 1 : 0;
		nnz += (0 != (c[i+7] = b * a[i+7])) ? 1 : 0;
		return nnz;
	}

	private static long denseBinaryScalarMultiplyMultiCol(DenseBlock da, DenseBlock dc, double b, int clen, int rl,
		int ru) {
		long nnz = 0;
		for(int i = rl; i < ru; i++) {
			final double[] a = da.values(i);
			final double[] c = dc.values(i);
			int apos = da.pos(i), cpos = dc.pos(i);
			for(int j = 0; j < clen; j++) {
				if(0 != (c[cpos + j] = b * a[apos + j]))
					nnz ++;
			}
		}
		return nnz;
	}

	private static long safeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		// early abort on skip and empty
		final boolean PoM = op.fn instanceof Plus || op.fn instanceof Minus;
		if((m1ret.isEmpty() && m2.isEmpty()) || (PoM && m2.isEmpty())) {
			final boolean isEquals = op.fn instanceof Equals || op.fn instanceof LessThanEquals ||
				op.fn instanceof GreaterThanEquals;

			if(isEquals){
				m1ret.reset(m1ret.rlen, m1ret.clen, 1.0); // fill with 1.0
				return (long)m1ret.rlen * m1ret.clen;
			}
			return 0; // skip entire empty block
		}
		else if(m2.isEmpty() && // empty other side
			(op.fn instanceof Multiply || (op.fn instanceof And))) {
			m1ret.reset(m1ret.rlen, m1ret.clen, 0.0);
			return 0;
		}

		if(m1ret.getNumRows() > 1 && m2.getNumRows() == 1)
			return safeBinaryInPlaceMatrixRowVector(m1ret, m2, op);
		else
			return safeBinaryInPlaceMatrixMatrix(m1ret, m2, op);
	}

	private static long safeBinaryInPlaceMatrixRowVector(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		if(m1ret.sparse) {
			if(m2.isInSparseFormat() && !op.isRowSafeLeft(m2))
				throw new DMLRuntimeException("Invalid row safety of in place row operation: " + op);
			else if(m2.isEmpty())
				safeBinaryInPlaceSparseConst(m1ret, 0.0, op);
			else if(m2.sparse)
				throw new NotImplementedException("Not made sparse vector in place to sparse " + op);
			else
				safeBinaryInPlaceSparseVector(m1ret, m2, op);
		}
		else {
			if(!m1ret.isAllocated()) {
				LOG.warn("Allocating in place output dense block");
				m1ret.allocateBlock();
			}

			if(m2.isEmpty())
				safeBinaryInPlaceDenseConst(m1ret, 0.0, op);
			else if(m2.sparse)
				throw new NotImplementedException("Not made sparse vector in place to dense " + op);
			else
				safeBinaryInPlaceDenseVector(m1ret, m2, op);
		}

		return m1ret.recomputeNonZeros();
	}

	private static long safeBinaryInPlaceMatrixMatrix(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		if(op.fn instanceof Plus && m1ret.isEmpty() && !m1ret.isAllocated())
			m1ret.copy(m2);
		else if(m1ret.sparse && m2.sparse)
			safeBinaryInPlaceSparse(m1ret, m2, op);
		else if(!m1ret.sparse && !m2.sparse)
			safeBinaryInPlaceDense(m1ret, m2, op);
		else if(m2.sparse && (op.fn instanceof Plus || op.fn instanceof Minus))
			safeBinaryInPlaceDenseSparseAdd(m1ret, m2, op);
		else
			safeBinaryInPlaceGeneric(m1ret, m2, op);

		return m1ret.recomputeNonZeros();
	}

	private static void safeBinaryInPlaceSparse(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		//allocation and preparation (note: for correctness and performance, this 
		//implementation requires the lhs in MCSR and hence we explicitly convert)
		if( m1ret.sparseBlock!=null )
			m1ret.allocateSparseRowsBlock(false);
		if( !(m1ret.sparseBlock instanceof SparseBlockMCSR) )
			m1ret.sparseBlock = SparseBlockFactory.copySparseBlock(
				SparseBlock.Type.MCSR, m1ret.sparseBlock, false);
		if( m2.sparseBlock!=null )
			m2.allocateSparseRowsBlock(false);
		final SparseBlock c = m1ret.sparseBlock;
		final SparseBlock b = m2.sparseBlock;
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		
		final boolean compact = shouldCompact(null, op); // do not analyze right
		final boolean mcsr = c instanceof SparseBlockMCSR;
		long nnz = 0;
		if( c!=null && b!=null ) {
			for(int r=0; r<rlen; r++) {
				if(c.isEmpty(r) && b.isEmpty(r))
					continue;
				if( b.isEmpty(r) ) {
					zeroRightForSparseBinary(op, r, m1ret);
				}
				else if( c.isEmpty(r) ) {
					nnz += appendRightForSparseBinary(op, b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
				}
				else {
					//this approach w/ single copy only works with the MCSR format
					int estimateSize = Math.min(clen, (!c.isEmpty(r) ?
						c.size(r) : 0) + (!b.isEmpty(r) ? b.size(r) : 0));
					SparseRow old = c.get(r);
					c.set(r, new SparseRowVector(estimateSize), false);
					
					nnz += mergeForSparseBinary(op, old.values(), old.indexes(), 0, 
						old.size(), b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
				}
				if(compact && mcsr && !c.isEmpty(r)){
					c.get(r).compact();
				}
			}

			if(compact && !mcsr){
				SparseBlockCSR sbcsr = (SparseBlockCSR)c;
				sbcsr.compact();
			}
			m1ret.setNonZeros(nnz);
		}
		else if( c == null && b != null ) { //lhs empty
			m1ret.sparseBlock = SparseBlockFactory.createSparseBlock(rlen);
			nnz = 0; 
			for(int r = 0; r < rlen; r++) {
				if(b.isEmpty(r))
					continue;
				nnz += appendRightForSparseBinary(op, b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
			}
			m1ret.setNonZeros(nnz);
		}
		else if (c != null) { //rhs empty
			for(int r=0; r<rlen; r++) {
				if( c.isEmpty(r) ) continue;
				zeroRightForSparseBinary(op, r, m1ret);
			}
			m1ret.recomputeNonZeros(op.getNumThreads());
		}
		
	}

	private static void safeBinaryInPlaceSparseConst(MatrixBlock m1ret, double m2, BinaryOperator op) {
		
		final SparseBlock sb = m1ret.getSparseBlock();
		final int rlen = m1ret.rlen;
		for(int r = 0; r < rlen; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final double[] avals = sb.values(r);
			for(int k = apos; k < alen; k++)
				avals[k] = op.fn.execute(avals[k], m2);
		}
	}

	private static void safeBinaryInPlaceSparseVector(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {

		if(m1ret.isEmpty()) // early termination... it is empty and safe... just stop.
			return;
		final SparseBlock sb = m1ret.getSparseBlock();
		final double[] b = m2.getDenseBlockValues();
		final int rlen = m1ret.rlen;

		final boolean compact = shouldCompact(m2, op);
		final boolean mcsr = sb instanceof SparseBlockMCSR;
		long nnz = 0;
		for(int r = 0; r < rlen; r++) {
			if(sb.isEmpty(r))
				continue;
			final int apos = sb.pos(r);
			final int alen = sb.size(r) + apos;
			final double[] avals = sb.values(r);
			final int[] aix = sb.indexes(r);
			for(int k = apos; k < alen; k++)
				avals[k] = op.fn.execute(avals[k], b[aix[k]]);

			if(compact && mcsr) {
				SparseRow sr = sb.get(r);
				if(sr instanceof SparseRowVector)
					((SparseRowVector) sr).setSize(avals.length);
				sr.compact();
				nnz += sr.size();
			}
		}
		if(compact && !mcsr) {
			final SparseBlockCSR sbcsr = (SparseBlockCSR) sb;
			sbcsr.compact();
			nnz = sbcsr.size();
		}
		m1ret.setNonZeros(nnz);
	}

	private static boolean shouldCompact(MatrixBlock m2, BinaryOperator op) {
		return (op.fn instanceof Multiply || op.fn instanceof And || 
			op.fn instanceof Builtin && ((Builtin)op.fn).bFunc == BuiltinCode.LOG_NZ) //
			&& (m2 == null || op.isIntroducingZerosRight(m2));
	}

	private static void safeBinaryInPlaceDense(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		// prepare outputs
		if(!m1ret.isAllocated()) // allocate
			m1ret.allocateDenseBlock();
		// guaranteed not to be empty inputs because of earlier aborts.
		if(op.fn instanceof Plus)
			safeBinaryInPlaceDensePlus(m1ret, m2, op);
		else
			safeBinaryInPlaceDenseGeneric(m1ret, m2, op);
	}

	private static void safeBinaryInPlaceDensePlus(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		DenseBlock a = m1ret.getDenseBlock();
		DenseBlock b = m2.getDenseBlock();
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		long lnnz = 0;
		if(a.isContiguous() && b.isContiguous()){
			final double[] avals = a.values(0);
			final double[] bvals = b.values(0);
			for(int i = 0; i < avals.length; i++)
				lnnz += (avals[i] += bvals[i]) == 0 ? 0 : 1;
		}
		else{
			for(int r = 0; r < rlen; r++) {
				final int aix = a.pos(r), bix = b.pos(r);
				final double[] avals = a.values(r), bvals = b.values(r);
				LibMatrixMult.vectAdd(bvals, avals, bix, aix, clen);
				lnnz += UtilFunctions.computeNnz(avals, aix, clen);
			}
		}
		m1ret.setNonZeros(lnnz);
	}

	private static void safeBinaryInPlaceDenseGeneric(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		DenseBlock a = m1ret.getDenseBlock();
		DenseBlock b = m2.getDenseBlock();
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		long lnnz = 0;
		for(int r = 0; r < rlen; r++) {
			double[] avals = a.values(r), bvals = b.values(r);
			for(int c = 0, ix = a.pos(r); c < clen; c++, ix++) {
				double tmp = op.fn.execute(avals[ix], bvals[ix]);
				lnnz += (avals[ix] = tmp) != 0 ? 1 : 0;
			}
		}
		m1ret.setNonZeros(lnnz);
	}


	private static void safeBinaryInPlaceDenseConst(MatrixBlock m1ret, double m2, BinaryOperator op) {
		// prepare outputs
		m1ret.allocateDenseBlock();
		DenseBlock a = m1ret.getDenseBlock();
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;

		long lnnz = 0;
		for(int r = 0; r < rlen; r++) {
			double[] avals = a.values(r);
			for(int c = 0, ix = a.pos(r); c < clen; c++, ix++) {
				double tmp = op.fn.execute(avals[ix], m2);
				lnnz += (avals[ix] = tmp) != 0 ? 1 : 0;
			}
		}

		m1ret.setNonZeros(lnnz);
	}

	private static void safeBinaryInPlaceDenseVector(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		// prepare outputs
		m1ret.allocateDenseBlock();
		DenseBlock a = m1ret.getDenseBlock();
		double[] b = m2.getDenseBlockValues();
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;

		long lnnz = 0;
		for(int r = 0; r < rlen; r++) {
			double[] avals = a.values(r);
			for(int c = 0, ix = a.pos(r); c < clen; c++, ix++) {
				double tmp = op.fn.execute(avals[ix], b[ix % clen]);
				lnnz += (avals[ix] = tmp) != 0 ? 1 : 0;
			}
		}
		m1ret.setNonZeros(lnnz);
	}
	
	private static void safeBinaryInPlaceDenseSparseAdd(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		final int rlen = m1ret.rlen;
		DenseBlock a = m1ret.denseBlock;
		SparseBlock b = m2.sparseBlock;
		long nnz = m1ret.getNonZeros();
		for(int r=0; r<rlen; r++) {
			if( b.isEmpty(r) ) continue;
			int apos = a.pos(r), bpos = b.pos(r);
			int blen = b.size(r);
			int[] bix = b.indexes(r);
			double[] avals = a.values(r), bvals = b.values(r);
			for(int k = bpos; k<bpos+blen; k++) {
				double vold = avals[apos+bix[k]];
				double vnew = op.fn.execute(vold, bvals[k]);
				nnz += (vold == 0 && vnew != 0) ? 1 :
					(vold != 0 && vnew ==0) ? -1  : 0;
				avals[apos+bix[k]] = vnew;
			}
		}
		m1ret.setNonZeros(nnz);
	}
	
	private static void safeBinaryInPlaceGeneric(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++) {
				double thisvalue = m1ret.get(r, c);
				double thatvalue = m2.get(r, c);
				double resultvalue = op.fn.execute(thisvalue, thatvalue);
				m1ret.set(r, c, resultvalue);
			}
	}
	
	private static long unsafeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op){
		int rlen = m1ret.rlen;
		int clen = m1ret.clen;
		BinaryAccessType atype = getBinaryAccessType(m1ret, m2);
		long nnz = 0;
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) //MATRIX - COL_VECTOR
		{
			for(int r=0; r<rlen; r++) {
				//replicated value
				double v2 = m2.get(r, 0);
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.get(r, c);
					double v = op.fn.execute( v1, v2 );
					m1ret.set(r, c, v);
					nnz += v != 0 ? 1 : 0;
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) //MATRIX - ROW_VECTOR
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.get(r, c);
					double v2 = m2.get(0, c); //replicated value
					double v = op.fn.execute( v1, v2 );
					m1ret.set(r, c, v);
					nnz += v != 0 ? 1 : 0;
				}
		}
		else // MATRIX - MATRIX
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.get(r, c);
					double v2 = m2.get(r, c);
					double v = op.fn.execute( v1, v2 );
					m1ret.set(r, c, v);
					nnz += v != 0 ? 1 : 0;
				}
		}

		return nnz;
	}
	
	private static long mergeForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1, 
			double[] values2, int[] cols2, int pos2, int size2, int resultRow, MatrixBlock result) {
		size1 = pos1 + size1;
		size2 = pos2 + size2;
		if(op.fn instanceof Multiply && result.isInSparseFormat())  //skip empty
			return mergeForSparseBinaryMultiply(op, values1, cols1, pos1, size1, values2, cols2, pos2, size2, resultRow, result);
		else if(result.isInSparseFormat())
			return mergeForSparseBinaryGeneric(op, values1, cols1, pos1, size1, values2, cols2, pos2, size2, resultRow, result);
		else 
			return mergeForSparseBinaryDenseOut(op, values1, cols1, pos1, size1, values2, cols2, pos2, size2, resultRow, result);
	}

	private static long mergeForSparseBinaryMultiply(final BinaryOperator op, final double[] values1, final int[] cols1,
		int pos1, final int size1, final double[] values2, final int[] cols2, int pos2, final int size2,
		final int resultRow, final MatrixBlock result) {

		final SparseBlockMCSR sblock = (SparseBlockMCSR) result.getSparseBlock();
		final SparseRowVector r = new SparseRowVector(Math.min(size1 - pos1, size2 - pos2),
			Math.max(size1 - pos1, size2 - pos2));

		while(pos1 < size1 && pos2 < size2) {
			final int colPos1 = cols1[pos1];
			final int colPos2 = cols2[pos2];
			if(colPos1 == colPos2)
				r.append(colPos1, op.fn.execute(values1[pos1++], values2[pos2++]));
			else if(colPos1 < colPos2)
				pos1++;
			else // colPos2 < colPos1
				pos2++;
		}
		sblock.set(resultRow, r, false);
		return sblock.size(resultRow);
	}

	private static long mergeForSparseBinaryGeneric(BinaryOperator op, double[] values1, int[] cols1, int pos1,
		int size1, double[] values2, int[] cols2, int pos2, int size2, int resultRow, MatrixBlock result) {

		final SparseBlockMCSR c = (SparseBlockMCSR) result.getSparseBlock();
		// preallocate at least biggest side size
		final int s1 = size1 - pos1;
		final int s2 = size2 - pos2;
		final int biggestSize = Math.max(s1, s2);
		final SparseRowVector r = new SparseRowVector(biggestSize, s1 + s2);
		// general case: merge-join (with outer join semantics)
		while(pos1 < size1 && pos2 < size2) {
			final int colPos1 = cols1[pos1];
			final int colPos2 = cols2[pos2];
			if(colPos1 < colPos2)
				r.append(colPos1, op.fn.execute(values1[pos1++], 0));
			else if(colPos1 == colPos2)
				r.append(colPos1, op.fn.execute(values1[pos1++], values2[pos2++]));
			else
				r.append(colPos2, op.fn.execute(0, values2[pos2++]));
		}
		// tails
		while(pos1 < size1) {
			r.append(cols1[pos1], op.fn.execute(values1[pos1], 0));
			pos1++;
		}
		while(pos2 < size2) {
			r.append(cols2[pos2], op.fn.execute(0, values2[pos2]));
			pos2++;
		}

		c.set(resultRow, r, false);
		return c.size(resultRow);
	}

	private static long mergeForSparseBinaryDenseOut(BinaryOperator op, double[] values1, int[] cols1, int pos1,
		int size1, double[] values2, int[] cols2, int pos2, int size2, int resultRow, MatrixBlock result) {

		// general case: merge-join (with outer join semantics)
		while(pos1 < size1 && pos2 < size2) {
			final int colPos1 = cols1[pos1];
			final int colPos2 = cols2[pos2];
			if(colPos1 < colPos2) 
				result.set(resultRow, colPos1, op.fn.execute(values1[pos1++], 0));
			else if(colPos1 == colPos2) 
				result.set(resultRow, colPos1, op.fn.execute(values1[pos1++], values2[pos2++]));
			else 
				result.set(resultRow, colPos2, op.fn.execute(0, values2[pos2++]));
		}
		// tails
		while (pos1 < size1){
			result.set(resultRow, cols1[pos1], op.fn.execute(values1[pos1], 0));
			pos1++;
		}
		while(pos2 < size2){
			result.set(resultRow, cols2[pos2], op.fn.execute(0, values2[pos2]));
			pos2++;
		}
		return result.recomputeNonZeros(resultRow, resultRow);
	}


	private static long appendLeftForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1,
		 int resultRow, MatrixBlock result) {
		if(result.isInSparseFormat()){
			final SparseBlock sb = result.getSparseBlock();
			for(int j = pos1 ; j < pos1 + size1; j++) {
				double v = op.fn.execute(values1[j], 0);
				sb.append(resultRow, cols1[j], v);
			}
			return sb.size(resultRow);
		}
		else{
			final DenseBlock db = result.getDenseBlock();
			long nnz = 0;
			for(int j = pos1; j < pos1 + size1; j++) {
				double v = op.fn.execute(values1[j], 0);
				db.set(resultRow, cols1[j], v);
				nnz += v != 0 ? 1 : 0;
			}
			return nnz;
		}
	}

	private static long appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int pos2, int size2,
		int r, MatrixBlock result) {
		if(result.isInSparseFormat()) {
			final SparseBlock sb = result.getSparseBlock();
			for(int j = pos2; j < pos2 + size2; j++) {
				double v = op.fn.execute(0, values2[j]);
				sb.append(r, cols2[j], v);
			}
			return sb.size(r);
		}
		else {
			final DenseBlock db = result.getDenseBlock();
			long nnz = 0;
			for(int j = pos2; j < pos2 + size2; j++) {
				double v = op.fn.execute(0, values2[j]);
				db.set(r, cols2[j], v);
				nnz += v != 0 ? 1 : 0;
			}
			return nnz;
		}
	}
	
	private static void zeroRightForSparseBinary(BinaryOperator op, int r, MatrixBlock ret) {
		if( op.fn instanceof Plus || op.fn instanceof Minus )
			return;
		SparseBlock c = ret.sparseBlock;
		int apos = c.pos(r);
		int alen = c.size(r);
		double[] values = c.values(r);
		boolean zero = false;
		for(int i=apos; i<apos+alen; i++)
			zero |= ((values[i] = op.fn.execute(values[i], 0)) == 0);
		if( zero )
			c.compact(r);
	}


	private static SparsityEstimate estimateSparsityOnBinary(MatrixBlock m1, MatrixBlock m2, BinaryOperator op){
		long nz1 = m1.getNonZeros();
		long nz2 = m2.getNonZeros();
		// If either side of matrix did not know the non zeros.
		if(nz1 <= 0)
			nz1 = m1.recomputeNonZeros(op.getNumThreads());
		if(nz2 <= 0)
			nz2 = m2.recomputeNonZeros(op.getNumThreads());

		
		final BinaryAccessType atype = LibMatrixBincell.getBinaryAccessType(m1, m2);
		final boolean outer = (atype == BinaryAccessType.OUTER_VECTOR_VECTOR);

		final long m = m1.getNumRows();
		final long n = outer ? m2.getNumColumns() : m1.getNumColumns();

		//estimate dense output for all sparse-unsafe operations, except DIV (because it commonly behaves like
		//sparse-safe but is not due to 0/0->NaN, this is consistent with the current hop sparsity estimate)
		//see also, special sparse-safe case for DIV in LibMatrixBincell 
		if( !op.sparseSafe && !(op.fn instanceof Divide && m2.getSparsity()==1.0) ) {
			// if not sparse safe and not div.
			return new SparsityEstimate(false, m*n);
		}
		else if(!outer && op.fn instanceof Divide && m2.getSparsity() == 1.0) {
			return new SparsityEstimate(m1.sparse, nz1);
		}
		
		
		//account for matrix vector and vector/vector
		long estnnz = 0;
		if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR )
		{
			estnnz = OptimizerUtils.getOuterNonZeros(
				m, n, nz1, nz2, op.getBinaryOperatorOpOp2());
		}
		else //DEFAULT CASE
		{
			if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
				nz2 = nz2 * n;
			else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
				nz2 = nz2 * m;
			
			//compute output sparsity consistent w/ the hop compiler
			double sp1 = OptimizerUtils.getSparsity(m, n, nz1);
			double sp2 = OptimizerUtils.getSparsity(m, n, nz2);
			double spout = OptimizerUtils.getBinaryOpSparsity(
				sp1, sp2, op.getBinaryOperatorOpOp2(), true);
			estnnz = UtilFunctions.toLong(spout * m * n);
		}
	
		return new SparsityEstimate(MatrixBlock.evalSparseFormatInMemory(m, n, estnnz), estnnz);
	}
	
	private static class BincellTask implements Callable<Long> {
		private final MatrixBlock _m1;
		private final MatrixBlock _m2;
		private final MatrixBlock _ret;
		private final BinaryOperator _bop;
		BinaryAccessType _atype;
		private final int _rl;
		private final int _ru;

		protected BincellTask( MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator bop, BinaryAccessType atype, int rl, int ru ) {
			_m1 = m1;
			_m2 = m2;
			_ret = ret;
			_bop = bop;
			_atype = atype;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() throws Exception {
			return binCellOpExecute(_m1,_m2, _ret,_bop, _atype, _rl, _ru);
		}
	}
	
	private static class BincellScalarTask implements Callable<Long> {
		private final MatrixBlock _m1;
		private final MatrixBlock _ret;
		private final ScalarOperator _sop;
		private final int _rl;
		private final int _ru;

		protected BincellScalarTask( MatrixBlock m1, MatrixBlock ret, ScalarOperator sop, int rl, int ru ) {
			_m1 = m1;
			_ret = ret;
			_sop = sop;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() throws Exception {
			return safeBinaryScalar(_m1, _ret, _sop, _rl, _ru);
		}
	}
	
	private static class UncellTask implements Callable<Long> {
		private final DenseBlock _a;
		private final DenseBlock _c;
		private final UnaryOperator _op;
		private final int _rl;
		private final int _ru;

		protected UncellTask(DenseBlock a, DenseBlock c, UnaryOperator op, int rl, int ru ) {
			_a = a;
			_c = c;
			_op = op;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() throws Exception {
			long nnz = 0;
			//fast dense-dense operations
			if(_a.isContiguous(_rl, _ru)) {
				double[] avals = _a.values(_rl);
				double[] cvals = _c.values(_rl);
				int start = _a.pos(_rl), end = _a.pos(_ru);
				for( int i=start; i<end; i++ ) {
					cvals[i] = _op.fn.execute(avals[i]);
					nnz += (cvals[i] != 0) ? 1 : 0;
				}
			}
			//generic dense-dense, including large blocks
			else {
				int clen = _a.getDim(1);
				for(int i=_rl; i<_ru; i++) {
					double[] avals = _a.values(i);
					double[] cvals = _c.values(i);
					int pos = _a.pos(i);
					for( int j=0; j<clen; j++ ) {
						cvals[pos+j] = _op.fn.execute(avals[pos+j]);
						nnz += (cvals[pos+j] != 0) ? 1 : 0;
					}
				}
			}
			return nnz;
		}
	}


	private static class SparsityEstimate {
		protected final  long estimatedNonZeros;
		protected final boolean sparse ;

		protected SparsityEstimate(boolean sp, long nnz) {
			this.estimatedNonZeros = nnz;
			this.sparse = sp;
		}
	}
}
