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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
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
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
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
	
	/**
	 * matrix-scalar, scalar-matrix binary operations.
	 * 
	 * @param m1 input matrix
	 * @param ret result matrix
	 * @param op scalar operator
	 */
	public static void bincellOp(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//check internal assumptions 
		if(   (op.sparseSafe && m1.isInSparseFormat()!=ret.isInSparseFormat())
			||(!op.sparseSafe && ret.isInSparseFormat()) ) {
			throw new DMLRuntimeException("Wrong output representation for safe="+op.sparseSafe+": "+m1.isInSparseFormat()+", "+ret.isInSparseFormat());
		}
		
		//execute binary cell operations
		if(op.sparseSafe)
			safeBinaryScalar(m1, ret, op, 0, m1.rlen);
		else
			unsafeBinaryScalar(m1, ret, op);
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
	}
	
	public static void bincellOp(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int k) {
		//check internal assumptions 
		if(   (op.sparseSafe && m1.isInSparseFormat()!=ret.isInSparseFormat())
			||(!op.sparseSafe && ret.isInSparseFormat()) ) {
			throw new DMLRuntimeException("Wrong output representation for safe="+op.sparseSafe+": "+m1.isInSparseFormat()+", "+ret.isInSparseFormat());
		}
		
		//fallback to singlet-threaded for special cases
		if( m1.isEmpty() || !op.sparseSafe 
			|| ret.getLength() < PAR_NUMCELL_THRESHOLD2 ) {
			bincellOp(m1, ret, op);
			return;
		}
		
		//preallocate dense/sparse block for multi-threaded operations
		ret.allocateBlock();
		
		try {
			//execute binary cell operations
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<BincellScalarTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(ret.rlen, k, false);
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
				tasks.add(new BincellScalarTask(m1, ret, op, lb, lb+blklens.get(i)));
			List<Future<Long>> taskret = pool.invokeAll(tasks);
			
			//aggregate non-zeros
			ret.nonZeros = 0; //reset after execute
			for( Future<Long> task : taskret )
				ret.nonZeros += task.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
	}
	
	/**
	 * matrix-matrix binary operations, MM, MV
	 * 
	 * @param m1 input matrix 1
	 * @param m2 input matrix 2
	 * @param ret result matrix
	 * @param op binary operator
	 */
	public static void bincellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		BinaryAccessType atype = getBinaryAccessType(m1, m2);

		// preallocate for consistency (but be careful 
		// not to allocate if empty inputs might allow early abort)
		if( atype == BinaryAccessType.MATRIX_MATRIX
			&& !(m1.isEmpty() || m2.isEmpty()) )
		{
			ret.allocateBlock(); //chosen outside
		}

		//execute binary cell operations
		long nnz = 0;
		if(op.sparseSafe || isSparseSafeDivide(op, m2))
			nnz = safeBinary(m1, m2, ret, op, atype, 0, m1.rlen);
		else
			nnz = unsafeBinary(m1, m2, ret, op, 0, m1.rlen);
		ret.setNonZeros(nnz);
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
	}
	
	public static void bincellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int k) {
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		//fallback to sequential computation for specialized operations
		if( m1.isEmpty() || m2.isEmpty()
 			|| ret.getLength() < PAR_NUMCELL_THRESHOLD2
			|| ((op.sparseSafe || isSparseSafeDivide(op, m2))
				&& !(atype == BinaryAccessType.MATRIX_MATRIX
					|| atype.isMatrixVector() && isAllDense(m1, m2, ret))))
		{
			bincellOp(m1, m2, ret, op);
			return;
		}
		
		//preallocate dense/sparse block for multi-threaded operations
		ret.allocateBlock(); //chosen outside
		
		try {
			//execute binary cell operations
			ExecutorService pool = CommonThreadPool.get(k);
			ArrayList<BincellTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(ret.rlen, k, false);
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
				tasks.add(new BincellTask(m1, m2, ret, op, atype, lb, lb+blklens.get(i)));
			List<Future<Long>> taskret = pool.invokeAll(tasks);
			
			//aggregate non-zeros
			ret.nonZeros = 0; //reset after execute
			for( Future<Long> task : taskret )
				ret.nonZeros += task.get();
			pool.shutdown();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//ensure empty results sparse representation
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
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
		//execute binary cell operations
		if(op.sparseSafe || isSparseSafeDivide(op, m2))
			safeBinaryInPlace(m1ret, m2, op);
		else
			unsafeBinaryInPlace(m1ret, m2, op);
		
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
		else if( clen1 > 1 && clen2 == 1 )
			return BinaryAccessType.MATRIX_COL_VECTOR;
		else if( rlen1 > 1 && clen1 > 1 && rlen2 == 1 )
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
			else if(clen1 < clen2)
				return BinaryAccessType.COL_VECTOR_MATRIX;
			else
				return BinaryAccessType.MATRIX_COL_VECTOR;
		}
		else if(clen1 == clen2) {
			if(rlen1 < rlen2)
				return BinaryAccessType.ROW_VECTOR_MATRIX;
			else
				return BinaryAccessType.MATRIX_ROW_VECTOR;
		}
		else if(clen1 == 1 && rlen2 == 1)
			return BinaryAccessType.OUTER_VECTOR_VECTOR;
		else
			return BinaryAccessType.INVALID;
	}

	public static void isValidDimensionsBinary(MatrixBlock m1, MatrixBlock m2)
	{
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
			throw new RuntimeException("Block sizes are not matched for binary " +
					"cell operations: " + rlen1 + "x" + clen1 + " vs " + rlen2 + "x" + clen2);
		}
	}

	public static void isValidDimensionsBinaryExtended(MatrixBlock m1, MatrixBlock m2) {
		final int rlen1 = m1.rlen;
		final int clen1 = m1.clen;
		final int rlen2 = m2.rlen;
		final int clen2 = m2.clen;

		// Added extra 2 options
		// 2a) VM operations with V either being a left-hand-side column or row vector.
		boolean isValid = ((rlen1 == rlen2 && clen1 == clen2) // MM
			|| (rlen1 == rlen2 && clen1 > 1 && clen2 == 1) // MVc
			|| (rlen1 == rlen2 && clen1 == 1 && clen2 > 1) // VMc
			|| (clen1 == clen2 && rlen1 > 1 && rlen2 == 1) // MVr
			|| (clen1 == clen2 && rlen1 == 1 && rlen2 > 1) // VMr
			|| (clen1 == 1 && rlen2 == 1)); // VV

		if(!isValid) {
			throw new RuntimeException("Block sizes are not matched for binary " + "cell operations: " + rlen1 + "x"
				+ clen1 + " vs " + rlen2 + "x" + clen2);
		}
	}

	public static boolean isSparseSafeDivide(BinaryOperator op, MatrixBlock rhs)
	{
		//if rhs is fully dense, there cannot be a /0 and hence DIV becomes sparse safe
		return (op.fn instanceof Divide && rhs.getNonZeros()==(long)rhs.getNumRows()*rhs.getNumColumns());
	}
	
	public static boolean isAllDense(MatrixBlock... mb) {
		return Arrays.stream(mb).allMatch(m -> !m.sparse);
	}
	
	//////////////////////////////////////////////////////
	// private sparse-safe/sparse-unsafe implementations
	///////////////////////////////////

	private static long safeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op,
		BinaryAccessType atype, int rl, int ru)
	{
		//NOTE: multi-threaded over rl-ru only applied for matrix-matrix, non-empty
		
		boolean skipEmpty = (op.fn instanceof Multiply 
			|| isSparseSafeDivide(op, m2) );
		boolean copyLeftRightEmpty = (op.fn instanceof Plus || op.fn instanceof Minus 
			|| op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		boolean copyRightLeftEmpty = (op.fn instanceof Plus);
		
		//skip empty blocks (since sparse-safe)
		if( m1.isEmptyBlock(false) && m2.isEmptyBlock(false) 
			|| skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)) ) 
		{
			return 0;
		}
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR //MATRIX - VECTOR
			|| atype == BinaryAccessType.MATRIX_ROW_VECTOR)
		{
			//note: m2 vector and hence always dense
			if( !m1.sparse && !m2.sparse && !ret.sparse ) //DENSE all
				return safeBinaryMVDense(m1, m2, ret, op, rl, ru);
			else if( m1.sparse && !m2.sparse && !ret.sparse
				&& atype == BinaryAccessType.MATRIX_ROW_VECTOR)
				safeBinaryMVSparseDenseRow(m1, m2, ret, op);
			else if( m1.sparse ) //SPARSE m1
				safeBinaryMVSparse(m1, m2, ret, op);
			else if( !m1.sparse && !m2.sparse && ret.sparse && op.fn instanceof Multiply
				&& atype == BinaryAccessType.MATRIX_COL_VECTOR
				&& (long)m1.rlen * m2.clen < Integer.MAX_VALUE )
				safeBinaryMVDenseSparseMult(m1, m2, ret, op);
			else //generic combinations
				safeBinaryMVGeneric(m1, m2, ret, op);
		}
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) //VECTOR - VECTOR
		{
			safeBinaryVVGeneric(m1, m2, ret, op);
		}
		else //MATRIX - MATRIX
		{
			if( copyLeftRightEmpty && m2.isEmpty() ) {
				//ret remains unchanged so a shallow copy is sufficient
				ret.copyShallow(m1);
			}
			else if( copyRightLeftEmpty && m1.isEmpty() ) {
				//ret remains unchanged so a shallow copy is sufficient
				ret.copyShallow(m2);
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
			else if( !ret.sparse && !m1.sparse && !m2.sparse 
				&& m1.denseBlock!=null && m2.denseBlock!=null ) {
				return safeBinaryMMDenseDenseDense(m1, m2, ret, op, rl, ru);
			}
			else if( skipEmpty && (m1.sparse || m2.sparse) ) {
				return safeBinaryMMSparseDenseSkip(m1, m2, ret, op, rl, ru);
			}
			else { //generic case
				return safeBinaryMMGeneric(m1, m2, ret, op, rl, ru);
			}
		}
		//default catch all
		return ret.getNonZeros();
	}

	private static long safeBinaryMVDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru) {
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		int clen = m1.clen;
		
		//early abort on skip and empy
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return 0; // skip entire empty block
		
		//guard for postponed allocation in single-threaded exec
		if( !ret.isAllocated() )
			ret.allocateDenseBlock();
		
		DenseBlock da = m1.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		long nnz = 0;
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			double[] b = m2.getDenseBlockValues(); // always single block
			
			for( int i=rl; i<ru; i++ ) {
				double[] a = da.values(i);
				double[] c = dc.values(i);
				int ix = da.pos(i);
				
				//replicate vector value
				double v2 = (b==null) ? 0 : b[i];
				if( skipEmpty && v2 == 0 ) //skip empty rows
					continue;
					
				if( isMultiply && v2 == 1 ) { //ROW COPY
					//a guaranteed to be non-null (see early abort)
					System.arraycopy(a, ix, c, ix, clen);
					nnz += m1.recomputeNonZeros(i, i, 0, clen-1);
				}
				else { //GENERAL CASE
					if( a != null )
						for( int j=0; j<clen; j++ ) {
							double val = op.fn.execute( a[ix+j], v2 );
							nnz += ((c[ix+j] = val) != 0) ? 1 : 0;
						}
					else {
						double val = op.fn.execute( 0, v2 );
						Arrays.fill(c, ix, ix+clen, val);
						nnz += (val != 0) ? clen : 0;
					}
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			double[] b = m2.getDenseBlockValues(); // always single block
			
			if( da==null && b==null ) { //both empty
				double val = op.fn.execute( 0, 0 );
				dc.set(rl, ru, 0, clen, val);
				nnz += (val != 0) ? (long)(ru-rl)*clen : 0;
			}
			else if( da==null ) //left empty
			{
				//compute first row
				double[] c = dc.values(rl);
				for( int j=0; j<clen; j++ ) {
					double val = op.fn.execute( 0, b[j] );
					nnz += ((c[j]=val) != 0) ? (ru-rl) : 0;
				}
				//copy first to all other rows
				for( int i=rl+1; i<ru; i++ )
					dc.set(i, c);
			}
			else //default case (incl right empty) 
			{
				for( int i=rl; i<ru; i++ ) {
					double[] a = da.values(i);
					double[] c = dc.values(i);
					int ix = da.pos(i);
					for( int j=0; j<clen; j++ ) {
						double val = op.fn.execute( a[ix+j], ((b!=null) ? b[j] : 0) );
						nnz += ((c[ix+j]=val) != 0) ? 1 : 0;
					}
				}
			}
		}
		
		return nnz;
	}

	private static void safeBinaryMVSparseDenseRow(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		int rlen = m1.rlen;
		int clen = m1.clen;
		SparseBlock a = m1.sparseBlock;
		double[] b = m2.getDenseBlockValues();
		DenseBlock c = ret.allocateDenseBlock().getDenseBlock();
		
		//early abort on skip and empty
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block

		//prepare op(0, m2) vector once for all rows
		double[] tmp = new double[clen];
		if( !skipEmpty ) {
			for( int i=0; i<clen; i++ )
				tmp[i] = op.fn.execute(0, b[i]);
		}
		
		long nnz = 0;
		for( int i=0; i<rlen; i++ ) {
			if( skipEmpty && (a==null || a.isEmpty(i)) )
				continue; //skip empty rows
			
			//set prepared empty row vector into output
			double[] cvals = c.values(i);
			int cpos = c.pos(i);
			System.arraycopy(tmp, 0, cvals, cpos, clen);
			
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
		ret.nonZeros = nnz;
	}
	
	private static void safeBinaryMVSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply || isSparseSafeDivide(op, m2));
		
		int rlen = m1.rlen;
		int clen = m1.clen;
		SparseBlock a = m1.sparseBlock;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		//early abort on skip and empty
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		//allocate once in order to prevent repeated reallocation
		if( ret.sparse )
			ret.allocateSparseRowsBlock();
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=0; i<rlen; i++ ) {
				double v2 = m2.quickGetValue(i, 0);
				
				if( (skipEmpty && (a==null || a.isEmpty(i) || v2 == 0 ))
					|| ((a==null || a.isEmpty(i)) && v2 == 0) )
				{
					continue; //skip empty rows
				}
					
				if( isMultiply && v2==1 ) { //ROW COPY
					if( a != null && !a.isEmpty(i)  )
						ret.appendRow(i, a.get(i));
				}
				else { //GENERAL CASE
					int lastIx = -1;
					if( a != null && !a.isEmpty(i) ) {
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for( int j=apos; j<apos+alen; j++ ) {
							//empty left
							fillZeroValues(op, v2, ret, skipEmpty, i, lastIx+1, aix[j]);
							//actual value
							double v = op.fn.execute( avals[j], v2 );
							ret.appendValue(i, aix[j], v);	
							lastIx = aix[j];
						}
					}
					//empty left
					fillZeroValues(op, v2, ret, skipEmpty, i, lastIx+1, clen);
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			for( int i=0; i<rlen; i++ ) {
				if( skipEmpty && (a==null || a.isEmpty(i)) )
					continue; //skip empty rows
				if( skipEmpty && ret.sparse )
					ret.sparseBlock.allocate(i, a.size(i));
				int lastIx = -1;
				if( a!=null && !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						//empty left
						fillZeroValues(op, m2, ret, skipEmpty, i, lastIx+1, aix[j]);
						//actual value
						double v2 = m2.quickGetValue(0, aix[j]);
						double v = op.fn.execute( avals[j], v2 );
						ret.appendValue(i, aix[j], v);
						lastIx = aix[j];
					}
				}
				//empty left
				fillZeroValues(op, m2, ret, skipEmpty, i, lastIx+1, clen);
			}
		}
		
		//no need to recomputeNonZeros since maintained in append value
	}
	
	private static void fillZeroValues(BinaryOperator op, double v2, MatrixBlock ret, boolean skipEmpty, int rpos, int cpos, int len) {
		if(skipEmpty)
			return;
		for( int k=cpos; k<len; k++ ){
			double v = op.fn.execute(0, v2);
			ret.appendValue(rpos, k, v);
		}
	}
	
	private static void fillZeroValues(BinaryOperator op, MatrixBlock m2, MatrixBlock ret, boolean skipEmpty, int rpos, int cpos, int len) {
		if(skipEmpty)
			return;
		for( int k=cpos; k<len; k++ ){
			double v2 = m2.quickGetValue(0, k);
			double v = op.fn.execute(0, v2);
			ret.appendValue(rpos, k, v);
		}
	}

	private static void safeBinaryMVDenseSparseMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
			return;
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		double[] a = m1.getDenseBlockValues();
		double[] b = m2.getDenseBlockValues();
		
		//note: invocation condition ensures max int nnz
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) {
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
	}
	
	private static void safeBinaryMVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		//early abort on skip and empty
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		//allocate once in order to prevent repeated reallocation 
		if( ret.sparse )
			ret.allocateSparseRowsBlock();
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=0; i<rlen; i++ )
			{
				//replicate vector value
				double v2 = m2.quickGetValue(i, 0);
				if( skipEmpty && v2 == 0 ) //skip zero rows
					continue;
				
				if(isMultiply && v2 == 1) //ROW COPY
				{
					for( int j=0; j<clen; j++ ) {
						double v1 = m1.quickGetValue(i, j);
						ret.appendValue(i, j, v1);
					}
				}
				else //GENERAL CASE
				{
					for( int j=0; j<clen; j++ ) {
						double v1 = m1.quickGetValue(i, j);
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(i, j, v);
					}
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			//if the right hand side row vector is sparse we have to exploit that;
			//otherwise, both sparse access (binary search) and asymtotic behavior
			//in the number of cells become major bottlenecks
			if( m2.sparse && ret.sparse && isMultiply ) //SPARSE *
			{
				//note: sparse block guaranteed to be allocated (otherwise early about)
				SparseBlock b = m2.sparseBlock;
				SparseBlock c = ret.sparseBlock;
				if( b.isEmpty(0) ) return; 
				int blen = b.size(0); //always pos 0
				int[] bix = b.indexes(0);
				double[] bvals = b.values(0);
				for( int i=0; i<rlen; i++ ) {
					c.allocate(i, blen);
					for( int j=0; j<blen; j++ )
						c.append(i, bix[j], m1.quickGetValue(i, bix[j]) * bvals[j]);
				}
				ret.setNonZeros(c.size());
			}
			else //GENERAL CASE
			{
				for( int i=0; i<rlen; i++ )
					for( int j=0; j<clen; j++ ) {
						double v1 = m1.quickGetValue(i, j);
						double v2 = m2.quickGetValue(0, j); //replicated vector value
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(i, j, v);
					}
			}
		}
		
		//no need to recomputeNonZeros since maintained in append value
	}
	
	private static void safeBinaryVVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		int rlen = m1.rlen;
		int clen = m2.clen;
		
		//allocate once in order to prevent repeated reallocation 
		if( ret.sparse )
			ret.allocateSparseRowsBlock();
		
		if(LibMatrixOuterAgg.isCompareOperator(op) 
			&& m2.getNumColumns()>16 && SortUtils.isSorted(m2) ) {
			performBinOuterOperation(m1, m2, ret, op);
		}
		else {
			for(int r=0; r<rlen; r++) {
				double v1 = m1.quickGetValue(r, 0);
				for(int c=0; c<clen; c++) {
					double v2 = m2.quickGetValue(0, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);
				}
			}
		}
		
		//no need to recomputeNonZeros since maintained in append value
	}
	
	private static long safeBinaryMMSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru)
	{
		//guard for postponed allocation in single-threaded exec
		if( ret.sparse && !ret.isAllocated() )
			ret.allocateSparseRowsBlock();
		
		//both sparse blocks existing
		long lnnz = 0;
		if(m1.sparseBlock!=null && m2.sparseBlock!=null)
		{
			SparseBlock lsblock = m1.sparseBlock;
			SparseBlock rsblock = m2.sparseBlock;
			
			if( ret.sparse && lsblock.isAligned(rsblock) )
			{
				SparseBlock c = ret.sparseBlock;
				for(int r=rl; r<ru; r++) 
					if( !lsblock.isEmpty(r) ) {
						int alen = lsblock.size(r);
						int apos = lsblock.pos(r);
						int[] aix = lsblock.indexes(r);
						double[] avals = lsblock.values(r);
						double[] bvals = rsblock.values(r);
						c.allocate(r, alen);
						for( int j=apos; j<apos+alen; j++ ) {
							double tmp = op.fn.execute(avals[j], bvals[j]);
							c.append(r, aix[j], tmp);
						}
						lnnz += c.size(r);
					}
			}
			else //general case
			{
				for(int r=rl; r<ru; r++) {
					if( !lsblock.isEmpty(r) && !rsblock.isEmpty(r) ) {
						mergeForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), lsblock.pos(r), lsblock.size(r),
							rsblock.values(r), rsblock.indexes(r), rsblock.pos(r), rsblock.size(r), r, ret);
					}
					else if( !rsblock.isEmpty(r) ) {
						appendRightForSparseBinary(op, rsblock.values(r), rsblock.indexes(r), 
							rsblock.pos(r), rsblock.size(r), 0, r, ret);
					}
					else if( !lsblock.isEmpty(r) ){
						appendLeftForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), 
							lsblock.pos(r), lsblock.size(r), 0, r, ret);
					}
					// do nothing if both not existing
					lnnz += ret.recomputeNonZeros(r, r);
				}
			}
		}
		//right sparse block existing
		else if( m2.sparseBlock!=null )
		{
			SparseBlock rsblock = m2.sparseBlock;
			for(int r=rl; r<Math.min(ru, rsblock.numRows()); r++) {
				if( rsblock.isEmpty(r) ) continue;
				appendRightForSparseBinary(op, rsblock.values(r), rsblock.indexes(r), 
					rsblock.pos(r), rsblock.size(r), 0, r, ret);
				lnnz += ret.recomputeNonZeros(r, r);
			}
		}
		//left sparse block existing
		else
		{
			SparseBlock lsblock = m1.sparseBlock;
			for(int r=rl; r<ru; r++) {
				if( lsblock.isEmpty(r) ) continue;
				appendLeftForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), 
					lsblock.pos(r), lsblock.size(r), 0, r, ret);
				lnnz += ret.recomputeNonZeros(r, r);
			}
		}
		return lnnz;
	}
	
	private static long safeBinaryMMSparseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru)
	{
		//guard for postponed allocation in single-threaded exec
		if( !ret.isAllocated() )
			ret.allocateDenseBlock();
		
		//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
		final int n = ret.clen;
		DenseBlock dc = ret.getDenseBlock();
		
		//1) process left input: assignment
		if( m1.sparse && m1.sparseBlock != null ) //SPARSE left
		{
			SparseBlock a = m1.sparseBlock;
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
		else if( !m1.sparse ) //DENSE left
		{
			if( !m1.isEmptyBlock(false) ) {
				int rlbix = dc.index(rl);
				int rubix = dc.index(ru-1);
				DenseBlock da = m1.getDenseBlock();
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
		
		//2) process right input: op.fn (+,-,*), * only if dense
		long lnnz = 0;
		if( m2.sparse && m2.sparseBlock!=null ) //SPARSE right
		{
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
		}
		else if( !m2.sparse ) //DENSE right
		{
			if( !m2.isEmptyBlock(false) ) {
				DenseBlock da = m2.getDenseBlock();
				for( int i=rl; i<ru; i++ ) {
					double[] a = da.values(i);
					double[] c = dc.values(i);
					int apos = da.pos(i);
					for( int j = apos; j<apos+n; j++ ) {
						c[j] = op.fn.execute(c[j], a[j]);
						lnnz += (c[j]!=0) ? 1 : 0;
					}
				}
			}
			else if(op.fn instanceof Multiply)
				ret.denseBlock.set(0);
			else
				lnnz = m1.nonZeros;
		}
		
		//3) recompute nnz
		return lnnz;
	}
	
	private static long safeBinaryMMDenseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru)
	{
		boolean isPM = m1.clen >= 512 & (op.fn instanceof PlusMultiply | op.fn instanceof MinusMultiply);
		double cntPM = !isPM ? Double.NaN : (op.fn instanceof PlusMultiply ?
			((PlusMultiply)op.fn).getConstant() : -1d * ((MinusMultiply)op.fn).getConstant());
		
		//guard for postponed allocation in single-threaded exec
		if( !ret.isAllocated() )
			ret.allocateDenseBlock();
		
		DenseBlock da = m1.getDenseBlock();
		DenseBlock db = m2.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		ValueFunction fn = op.fn;
		int clen = m1.clen;
		
		//compute dense-dense binary, maintain nnz on-the-fly
		long lnnz = 0;
		for(int i=rl; i<ru; i++) {
			double[] a = da.values(i);
			double[] b = db.values(i);
			double[] c = dc.values(i);
			int pos = da.pos(i);
			
			if( isPM ) {
				System.arraycopy(a, pos, c, pos, clen);
				LibMatrixMult.vectMultiplyAdd(cntPM, b, c, pos, pos, clen);
				lnnz += UtilFunctions.computeNnz(c, pos, clen);
			}
			else {
				for(int j=pos; j<pos+clen; j++) {
					c[j] = fn.execute(a[j], b[j]);
					lnnz += (c[j]!=0)? 1 : 0;
				}
			}
		}
		return lnnz;
	}
	
	private static long safeBinaryMMSparseDenseSkip(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret,
		BinaryOperator op, int rl, int ru)
	{
		SparseBlock a = m1.sparse ? m1.sparseBlock : m2.sparseBlock;
		if( a == null )
			return 0;
		
		//prepare second input and allocate output
		MatrixBlock b = m1.sparse ? m2 : m1;
		
		//guard for postponed allocation in single-threaded exec
		if( !ret.isAllocated() )
			ret.allocateBlock();
		
		long lnnz = 0;
		for( int i=rl; i<Math.min(ru, a.numRows()); i++ ) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			if( ret.sparse && !b.sparse )
				ret.sparseBlock.allocate(i, alen);
			for(int k = apos; k < apos+alen; k++) {
				double in2 = b.quickGetValue(i, aix[k]);
				if( in2==0 ) continue;
				double val = op.fn.execute(avals[k], in2);
				lnnz += (val != 0) ? 1 : 0;
				ret.appendValuePlain(i, aix[k], val);
			}
		}
		return lnnz;
	}
	
	private static long safeBinaryMMGeneric(MatrixBlock m1, MatrixBlock m2,
		MatrixBlock ret, BinaryOperator op, int rl, int ru)
	{
		int clen = m2.clen;
		long lnnz = 0;
		for(int r=rl; r<ru; r++)
			for(int c=0; c<clen; c++) {
				double in1 = m1.quickGetValue(r, c);
				double in2 = m2.quickGetValue(r, c);
				if( in1==0 && in2==0) continue;
				double val = op.fn.execute(in1, in2);
				lnnz += (val != 0) ? 1 : 0;
				ret.appendValuePlain(r, c, val);
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
	private static long performBinOuterOperation(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator bOp) {
		int rlen = m1.rlen;
		int clen = ret.clen;
		double b[] = DataConverter.convertToDoubleVector(m2);
		if(!ret.isAllocated())
			ret.allocateDenseBlock();
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
		for( int bi=0; bi<dc.numBlocks(); bi++ ) {
			double[] c = dc.valuesAt(bi);
			for( int r=bi*dc.blockSize(), off=0; r<rlen; r++, off+=clen ) {
				double value = m1.quickGetValue(r, 0);
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
					Arrays.fill(c, off, off+start, 1.0);
					Arrays.fill(c, off+end, off+clen, 1.0);
					lnnz += (start+(clen-end));
				}
				else if( start < end ) {
					Arrays.fill(c, off+start, off+end, 1.0);
					lnnz += (end-start);
				}
			}
		}
		ret.setNonZeros(lnnz);
		ret.examSparsity();
		return lnnz;
	}

	private static long unsafeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op, int rl, int ru) {
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		long lnnz = 0;
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) { //MATRIX - COL_VECTOR
			for(int r=rl; r<ru; r++) {
				double v2 = m2.quickGetValue(r, 0);
				for(int c=0; c<clen; c++) {
					double v1 = m1.quickGetValue(r, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValuePlain(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) { //MATRIX - ROW_VECTOR
			for(int r=rl; r<ru; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1.quickGetValue(r, c);
					double v2 = m2.quickGetValue(0, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValuePlain(r, c, v);
					lnnz += (v!=0) ? 1 : 0;
				}
		}
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) { //VECTOR - VECTOR
			int clen2 = m2.clen; 
			if(LibMatrixOuterAgg.isCompareOperator(op) 
				&& m2.getNumColumns()>16 && SortUtils.isSorted(m2)) {
				lnnz = performBinOuterOperation(m1, m2, ret, op);
			} 
			else {
				for(int r=rl; r<ru; r++) {
					double v1 = m1.quickGetValue(r, 0);
					for(int c=0; c<clen2; c++) {
						double v2 = m2.quickGetValue(0, c);
						double v = op.fn.execute( v1, v2 );
						lnnz += (v != 0) ? 1 : 0;
						ret.appendValuePlain(r, c, v);
					}
				}
			}
		}
		else // MATRIX - MATRIX
		{
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
			//general case
			else {
				for(int r=rl; r<ru; r++)
					for(int c=0; c<clen; c++) {
						double v1 = m1.quickGetValue(r, c);
						double v2 = m2.quickGetValue(r, c);
						double v = op.fn.execute( v1, v2 );
						ret.appendValuePlain(r, c, v);
						lnnz += (v!=0) ? 1 : 0;
					}
			}
		}
		
		return lnnz;
	}

	private static long safeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl, int ru) {
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false) ) {
			return 0;
		}
		
		//sanity check input/output sparsity
		if( m1.sparse != ret.sparse )
			throw new DMLRuntimeException("Unsupported safe binary scalar operations over different input/output representation: "+m1.sparse+" "+ret.sparse);
		
		boolean copyOnes = (op.fn instanceof NotEquals && op.getConstant()==0);
		boolean allocExact = (op.fn instanceof Multiply || op.fn instanceof Multiply2 
			|| op.fn instanceof Power2 || Builtin.isBuiltinCode(op.fn, BuiltinCode.MAX)
			|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MIN));
		long lnnz = 0;
		
		if( m1.sparse ) //SPARSE <- SPARSE
		{
			//allocate sparse row structure
			ret.allocateSparseRowsBlock();
			SparseBlock a = m1.sparseBlock;
			SparseBlock c = ret.sparseBlock;
			
			long nnz = 0;
			for(int r=rl; r<ru; r++) {
				if( a.isEmpty(r) ) continue;
				
				int apos = a.pos(r);
				int alen = a.size(r);
				int[] aix = a.indexes(r);
				double[] avals = a.values(r);
				
				if( copyOnes ) { //SPECIAL CASE: e.g., (X != 0) 
					//create sparse row without repeated resizing
					SparseRowVector crow = new SparseRowVector(alen);
					crow.setSize(alen);
					
					//memcopy/memset of indexes/values (sparseblock guarantees absence of 0s) 
					System.arraycopy(aix, apos, crow.indexes(), 0, alen);
					Arrays.fill(crow.values(), 0, alen, 1);
					c.set(r, crow, false);
					nnz += alen;
				}
				else { //GENERAL CASE
					//create sparse row without repeated resizing for specific ops
					if( allocExact )
						c.allocate(r, alen);
					
					for(int j=apos; j<apos+alen; j++) {
						double val = op.executeScalar(avals[j]);
						c.append(r, aix[j], val);
						nnz += (val != 0) ? 1 : 0; 
					}
				}
			}
			lnnz = (ret.nonZeros = nnz);
		}
		else { //DENSE <- DENSE
			lnnz = denseBinaryScalar(m1, ret, op, rl, ru);
		}
		
		return lnnz;
	}
	
	/**
	 * Since this operation is sparse-unsafe, ret should always be passed in dense representation.
	 * 
	 * @param m1 input matrix
	 * @param ret result matrix
	 * @param op scalar operator
	 */
	private static long unsafeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false) ) {
			//compute 0 op constant once and set into dense output
			double val = op.executeScalar(0);
			if( val != 0 )
				ret.reset(ret.rlen, ret.clen, val);
			return (val != 0) ? ret.getLength() : 0;
		}
		
		//sanity check input/output sparsity
		if( ret.sparse )
			throw new DMLRuntimeException("Unsupported unsafe binary scalar operations over sparse output representation.");
		
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
		}
		
		return lnnz;
	}

	private static long denseBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op, int rl, int ru) {
		//allocate dense block (if necessary), incl clear nnz
		ret.allocateDenseBlock(true);
		
		DenseBlock da = m1.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		int clen = m1.clen;
		
		//compute scalar operation, incl nnz maintenance
		long nnz = 0;
		for(int i=rl; i<ru; i++) {
			double[] a = da.values(i);
			double[] c = dc.values(i);
			int apos = da.pos(i), cpos = dc.pos(i);
			for(int j=0; j<clen; j++) {
				c[cpos+j] = op.executeScalar( a[apos+j] );
				nnz += (c[cpos+j] != 0) ? 1 : 0;
			}
		}
		return ret.nonZeros = nnz;
	}

	private static void safeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		//early abort on skip and empty 
		if( (m1ret.isEmpty() && m2.isEmpty() )
			|| (op.fn instanceof Plus && m2.isEmpty())
			|| (op.fn instanceof Minus && m2.isEmpty()))
			return; // skip entire empty block
		//special case: start aggregation
		else if( op.fn instanceof Plus && m1ret.isEmpty() ){
			m1ret.copy(m2);
			return; 
		}
		
		if(m1ret.sparse && m2.sparse)
			safeBinaryInPlaceSparse(m1ret, m2, op);
		else if(!m1ret.sparse && !m2.sparse)
			safeBinaryInPlaceDense(m1ret, m2, op);
		else if(m2.sparse && (op.fn instanceof Plus || op.fn instanceof Minus))
			safeBinaryInPlaceDenseSparseAdd(m1ret, m2, op);
		else //GENERIC
			safeBinaryInPlaceGeneric(m1ret, m2, op);
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
		SparseBlock c = m1ret.sparseBlock;
		SparseBlock b = m2.sparseBlock;
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		
		if( c!=null && b!=null ) {
			for(int r=0; r<rlen; r++) {
				if(c.isEmpty(r) && b.isEmpty(r))
					continue;
				if( b.isEmpty(r) ) {
					zeroRightForSparseBinary(op, r, m1ret);
				}
				else if( c.isEmpty(r) ) {
					appendRightForSparseBinary(op, b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
				}
				else {
					//this approach w/ single copy only works with the MCSR format
					int estimateSize = Math.min(clen, (!c.isEmpty(r) ?
						c.size(r) : 0) + (!b.isEmpty(r) ? b.size(r) : 0));
					SparseRow old = c.get(r);
					c.set(r, new SparseRowVector(estimateSize), false);
					m1ret.nonZeros -= old.size();
					mergeForSparseBinary(op, old.values(), old.indexes(), 0, 
						old.size(), b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
				}
			}
		}
		else if( c == null ) { //lhs empty
			m1ret.sparseBlock = SparseBlockFactory.createSparseBlock(rlen);
			for( int r=0; r<rlen; r++) {
				if( b.isEmpty(r) ) continue;
				appendRightForSparseBinary(op, b.values(r),
					b.indexes(r), b.pos(r), b.size(r), r, m1ret);
			}
		}
		else { //rhs empty
			for(int r=0; r<rlen; r++) {
				if( c.isEmpty(r) ) continue;
				zeroRightForSparseBinary(op, r, m1ret);
			}
		}
		
		m1ret.recomputeNonZeros();
	}

	private static void safeBinaryInPlaceDense(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		//prepare outputs
		m1ret.allocateDenseBlock();
		DenseBlock a = m1ret.getDenseBlock();
		DenseBlock b = m2.getDenseBlock();
		final int rlen = m1ret.rlen;
		final int clen = m1ret.clen;
		
		long lnnz = 0;
		if( m2.isEmptyBlock(false) ) {
			for(int r=0; r<rlen; r++) {
				double[] avals = a.values(r);
				for(int c=0, ix=a.pos(r); c<clen; c++, ix++) {
					double tmp = op.fn.execute(avals[ix], 0);
					lnnz += (avals[ix] = tmp) != 0 ? 1: 0;
				}
			}
		}
		else if( op.fn instanceof Plus ) {
			for(int r=0; r<rlen; r++) {
				int aix = a.pos(r), bix = b.pos(r);
				double[] avals = a.values(r), bvals = b.values(r);
				LibMatrixMult.vectAdd(bvals, avals, bix, aix, clen);
				lnnz += UtilFunctions.computeNnz(avals, aix, clen);
			}
		}
		else {
			for(int r=0; r<rlen; r++) {
				double[] avals = a.values(r), bvals = b.values(r);
				for(int c=0, ix=a.pos(r); c<clen; c++, ix++) {
					double tmp = op.fn.execute(avals[ix], bvals[ix]);
					lnnz += (avals[ix] = tmp) != 0 ? 1 : 0;
				}
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
				double thisvalue = m1ret.quickGetValue(r, c);
				double thatvalue = m2.quickGetValue(r, c);
				double resultvalue = op.fn.execute(thisvalue, thatvalue);
				m1ret.quickSetValue(r, c, resultvalue);
			}
	}
	
	private static void unsafeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op)
	{
		int rlen = m1ret.rlen;
		int clen = m1ret.clen;
		BinaryAccessType atype = getBinaryAccessType(m1ret, m2);
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) //MATRIX - COL_VECTOR
		{
			for(int r=0; r<rlen; r++) {
				//replicated value
				double v2 = m2.quickGetValue(r, 0);
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.quickGetValue(r, c);
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) //MATRIX - ROW_VECTOR
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.quickGetValue(r, c);
					double v2 = m2.quickGetValue(0, c); //replicated value
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
		}
		else // MATRIX - MATRIX
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1ret.quickGetValue(r, c);
					double v2 = m2.quickGetValue(r, c);
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
		}
	}
	
	private static void mergeForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1, 
			double[] values2, int[] cols2, int pos2, int size2, int resultRow, MatrixBlock result) {
		int p1 = 0, p2 = 0;
		if( op.fn instanceof Multiply ) { //skip empty
			//skip empty: merge-join (with inner join semantics)
			//similar to sorted list intersection
			if(result.getSparseBlock() == null)
				result.allocateSparseRowsBlock();
			SparseBlock sblock = result.getSparseBlock();
			sblock.allocate(resultRow, Math.min(size1, size2), result.clen);
			while( p1 < size1 && p2 < size2 ) {
				int colPos1 = cols1[pos1+p1];
				int colPos2 = cols2[pos2+p2];
				if( colPos1 == colPos2 )
					sblock.append(resultRow, colPos1,
						op.fn.execute(values1[pos1+p1], values2[pos2+p2]));
				p1 += (colPos1 <= colPos2) ? 1 : 0;
				p2 += (colPos1 >= colPos2) ? 1 : 0;
			}
			result.nonZeros += sblock.size(resultRow);
		}
		else {
			//general case: merge-join (with outer join semantics) 
			while( p1 < size1 && p2 < size2 ) {
				if(cols1[pos1+p1]<cols2[pos2+p2]) {
					result.appendValue(resultRow, cols1[pos1+p1], 
						op.fn.execute(values1[pos1+p1], 0));
					p1++;
				}
				else if(cols1[pos1+p1]==cols2[pos2+p2]) {
					result.appendValue(resultRow, cols1[pos1+p1], 
						op.fn.execute(values1[pos1+p1], values2[pos2+p2]));
					p1++;
					p2++;
				}
				else {
					result.appendValue(resultRow, cols2[pos2+p2], 
						op.fn.execute(0, values2[pos2+p2]));
					p2++;
				}
			}
			//add left over
			appendLeftForSparseBinary(op, values1, cols1, pos1, size1, p1, resultRow, result);
			appendRightForSparseBinary(op, values2, cols2, pos2, size2, p2, resultRow, result);
		}
	}

	private static void appendLeftForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1, 
				int pos, int resultRow, MatrixBlock result) {
		for(int j=pos1+pos; j<pos1+size1; j++) {
			double v = op.fn.execute(values1[j], 0);
			result.appendValue(resultRow, cols1[j], v);
		}
	}

	private static void appendRightForSparseBinary(BinaryOperator op, double[] vals, int[] ix, int pos, int size, int r, MatrixBlock ret) {
		appendRightForSparseBinary(op, vals, ix, pos, size, 0, r, ret);
	}
	
	private static void appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int pos2, int size2, 
			int pos, int r, MatrixBlock result) {
		for( int j=pos2+pos; j<pos2+size2; j++ ) {
			double v = op.fn.execute(0, values2[j]);
			result.appendValue(r, cols2[j], v);
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
		public Long call() {
			// execute binary operation on row partition
			// (including nnz maintenance)
			if(_bop.sparseSafe || isSparseSafeDivide(_bop, _m2))
				return safeBinary(_m1, _m2, _ret, _bop, _atype, _rl, _ru);
			else
				return unsafeBinary(_m1, _m2, _ret, _bop, _rl, _ru);
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
		public Long call() {
			//execute binary operation on row partition
			return safeBinaryScalar(_m1, _ret, _sop, _rl, _ru);
		}
	}
}
