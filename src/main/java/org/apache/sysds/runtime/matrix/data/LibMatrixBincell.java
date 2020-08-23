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

import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.Builtin;
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
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
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
public class LibMatrixBincell 
{

	public enum BinaryAccessType {
		MATRIX_MATRIX,
		MATRIX_COL_VECTOR,
		MATRIX_ROW_VECTOR,
		OUTER_VECTOR_VECTOR,
		INVALID,
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
			safeBinaryScalar(m1, ret, op);
		else
			unsafeBinaryScalar(m1, ret, op);
		
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
		//execute binary cell operations
		if(op.sparseSafe || isSparseSafeDivide(op, m2))
			safeBinary(m1, m2, ret, op);
		else
			unsafeBinary(m1, m2, ret, op);
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( ret.isEmptyBlock(false) )
			ret.examSparsity();
	}
	
	/**
	 * NOTE: operations in place always require m1 and m2 to be of equal dimensions
	 * 
	 * @param m1ret result matrix
	 * @param m2 matrix block
	 * @param op binary operator
	 */
	public static void bincellOpInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) {
		//execute binary cell operations
		if(op.sparseSafe || isSparseSafeDivide(op, m2))
			safeBinaryInPlace(m1ret, m2, op);
		else
			unsafeBinaryInPlace(m1ret, m2, op);
		
		//ensure empty results sparse representation 
		//(no additional memory requirements)
		if( m1ret.isEmptyBlock(false) )
			m1ret.examSparsity();
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

	public static boolean isValidDimensionsBinary(MatrixBlock m1, MatrixBlock m2)
	{
		int rlen1 = m1.rlen;
		int clen1 = m1.clen;
		int rlen2 = m2.rlen;
		int clen2 = m2.clen;
		
		//currently we support three major binary cellwise operations:
		//1) MM (where both dimensions need to match)
		//2) MV operations w/ V either being a right-hand-side column or row vector 
		//  (where one dimension needs to match and the other dimension is 1)
		//3) VV outer vector operations w/ a common dimension of 1 
		
		return (   (rlen1 == rlen2 && clen1==clen2)            //MM 
				|| (rlen1 == rlen2 && clen1 > 1 && clen2 == 1) //MVc
				|| (clen1 == clen2 && rlen1 > 1 && rlen2 == 1) //MVr
				|| (clen1 == 1 && rlen2 == 1 ) );              //VV
	}

	public static boolean isSparseSafeDivide(BinaryOperator op, MatrixBlock rhs)
	{
		//if rhs is fully dense, there cannot be a /0 and hence DIV becomes sparse safe
		return (op.fn instanceof Divide && rhs.getNonZeros()==(long)rhs.getNumRows()*rhs.getNumColumns());
	}
	
	//////////////////////////////////////////////////////
	// private sparse-safe/sparse-unsafe implementations
	///////////////////////////////////

	private static void safeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		boolean skipEmpty = (op.fn instanceof Multiply 
			|| isSparseSafeDivide(op, m2) );
		boolean copyLeftRightEmpty = (op.fn instanceof Plus || op.fn instanceof Minus 
			|| op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply);
		boolean copyRightLeftEmpty = (op.fn instanceof Plus);
		
		//skip empty blocks (since sparse-safe)
		if( m1.isEmptyBlock(false) && m2.isEmptyBlock(false) 
			|| skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)) ) 
		{
			return;
		}
		
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR //MATRIX - VECTOR
			|| atype == BinaryAccessType.MATRIX_ROW_VECTOR)  
		{
			//note: m2 vector and hence always dense
			if( !m1.sparse && !m2.sparse && !ret.sparse ) //DENSE all
				safeBinaryMVDense(m1, m2, ret, op);
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
				safeBinaryMMSparseSparse(m1, m2, ret, op);
			}
			else if( !ret.sparse && (m1.sparse || m2.sparse) &&
				(op.fn instanceof Plus || op.fn instanceof Minus ||
				op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply ||
				(op.fn instanceof Multiply && !m2.sparse ))) {
				safeBinaryMMSparseDenseDense(m1, m2, ret, op);
			}
			else if( !ret.sparse && !m1.sparse && !m2.sparse 
				&& m1.denseBlock!=null && m2.denseBlock!=null ) {
				safeBinaryMMDenseDenseDense(m1, m2, ret, op);
			}
			else if( skipEmpty && (m1.sparse || m2.sparse) ) {
				safeBinaryMMSparseDenseSkip(m1, m2, ret, op);
			}
			else { //generic case
				safeBinaryMMGeneric(m1, m2, ret, op);
			}
		}
	}

	private static void safeBinaryMVDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		int rlen = m1.rlen;
		int clen = m1.clen;
		
		//early abort on skip and empy
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		ret.allocateDenseBlock();
		DenseBlock da = m1.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		long nnz = 0;
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			double[] b = m2.getDenseBlockValues(); // always single block
			
			for( int bi=0; bi<dc.numBlocks(); bi++ ) {
				double[] a = da.valuesAt(bi);
				double[] c = dc.valuesAt(bi);
				int len = dc.blockSize(bi);
				int off = bi * dc.blockSize();
				for( int i=0, ix=0; i<len; i++, ix+=clen )
				{
					//replicate vector value
					double v2 = (b==null) ? 0 : b[off+i];
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
								c[ix+j] = op.fn.execute( a[ix+j], v2 );	
								nnz += (c[ix+j] != 0) ? 1 : 0;
							}
						else {
							double val = op.fn.execute( 0, v2 );
							Arrays.fill(c, ix, ix+clen, val);
							nnz += (val != 0) ? clen : 0;
						}
					}
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			double[] b = m2.getDenseBlockValues(); // always single block
			
			if( da==null && b==null ) { //both empty
				double v = op.fn.execute( 0, 0 );
				dc.set(v);
				nnz += (v != 0) ? (long)rlen*clen : 0;
			}
			else if( da==null ) //left empty
			{
				//compute first row
				double[] c = dc.valuesAt(0);
				for( int j=0; j<clen; j++ ) {
					c[j] = op.fn.execute( 0, b[j] );
					nnz += (c[j] != 0) ? rlen : 0;
				}
				//copy first to all other rows
				for( int i=1; i<rlen; i++ )
					dc.set(i, c);
			}
			else //default case (incl right empty) 
			{
				for( int bi=0; bi<dc.numBlocks(); bi++ ) {
					double[] a = da.valuesAt(bi);
					double[] c = dc.valuesAt(bi);
					int len = dc.blockSize(bi);
					for( int i=0, ix=0; i<len; i++, ix+=clen )
						for( int j=0; j<clen; j++ ) {
							c[ix+j] = op.fn.execute( a[ix+j], ((b!=null) ? b[j] : 0) );
							nnz += (c[ix+j] != 0) ? 1 : 0;
						}
				}
			}
		}
		
		ret.nonZeros = nnz;
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
		boolean skipEmpty = (isMultiply);
		
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
	
	private static void safeBinaryMMSparseSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		final int rlen = m1.rlen;
		if(ret.sparse)
			ret.allocateSparseRowsBlock();
		
		//both sparse blocks existing
		if(m1.sparseBlock!=null && m2.sparseBlock!=null)
		{
			SparseBlock lsblock = m1.sparseBlock;
			SparseBlock rsblock = m2.sparseBlock;
			
			if( ret.sparse && lsblock.isAligned(rsblock) )
			{
				SparseBlock c = ret.sparseBlock;
				for(int r=0; r<rlen; r++) 
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
						ret.nonZeros += c.size(r);
					}
			}
			else //general case
			{
				for(int r=0; r<rlen; r++) {
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
				}
			}
		}
		//right sparse block existing
		else if( m2.sparseBlock!=null )
		{
			SparseBlock rsblock = m2.sparseBlock;
			for(int r=0; r<Math.min(rlen, rsblock.numRows()); r++) {
				if( rsblock.isEmpty(r) ) continue;
				appendRightForSparseBinary(op, rsblock.values(r), rsblock.indexes(r), 
					rsblock.pos(r), rsblock.size(r), 0, r, ret);
			}
		}
		//left sparse block existing
		else
		{
			SparseBlock lsblock = m1.sparseBlock;
			for(int r=0; r<rlen; r++) {
				if( lsblock.isEmpty(r) ) continue;
				appendLeftForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), 
					lsblock.pos(r), lsblock.size(r), 0, r, ret);
			}
		}
	}
	
	private static void safeBinaryMMSparseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
		ret.allocateDenseBlock();
		final int n = ret.clen;
		DenseBlock dc = ret.getDenseBlock();
		
		//1) process left input: assignment
		
		if( m1.sparse && m1.sparseBlock != null ) //SPARSE left
		{
			SparseBlock a = m1.sparseBlock;
			for( int bi=0; bi<dc.numBlocks(); bi++ ) {
				double[] c = dc.valuesAt(bi);
				int blen = dc.blockSize(bi);
				int off = bi * dc.blockSize();
				for( int i=0, ix=0; i<blen; i++, ix+=n ) {
					int ai = off + i;
					if( a.isEmpty(ai) ) continue;
					int apos = a.pos(ai);
					int alen = a.size(ai);
					int[] aix = a.indexes(ai);
					double[] avals = a.values(ai);
					for(int k = apos; k < apos+alen; k++) 
						c[ix+aix[k]] = avals[k];
				}
			}
		}
		else if( !m1.sparse ) //DENSE left
		{
			if( !m1.isEmptyBlock(false) ) 
				dc.set(m1.getDenseBlock());
			else
				dc.set(0);
		}
		
		//2) process right input: op.fn (+,-,*), * only if dense
		long lnnz = 0;
		if( m2.sparse && m2.sparseBlock!=null ) //SPARSE right
		{
			SparseBlock a = m2.sparseBlock;
			for( int bi=0; bi<dc.numBlocks(); bi++ ) {
				double[] c = dc.valuesAt(bi);
				int blen = dc.blockSize(bi);
				int off = bi * dc.blockSize();
				for( int i=0, ix=0; i<blen; i++, ix+=n ) {
					int ai = off + i;
					if( !a.isEmpty(ai) ) {
						int apos = a.pos(ai);
						int alen = a.size(ai);
						int[] aix = a.indexes(ai);
						double[] avals = a.values(ai);
						for(int k = apos; k < apos+alen; k++) 
							c[ix+aix[k]] = op.fn.execute(c[ix+aix[k]], avals[k]);
					}
					//exploit temporal locality of rows
					lnnz += ret.recomputeNonZeros(ai, ai, 0, n-1);
				}
			}
		}
		else if( !m2.sparse ) //DENSE right
		{
			if( !m2.isEmptyBlock(false) ) {
				for( int bi=0; bi<dc.numBlocks(); bi++ ) {
					double[] a = m2.getDenseBlock().valuesAt(bi);
					double[] c = dc.valuesAt(bi);
					int len = dc.size(bi);
					for( int i=0; i<len; i++ ) {
						c[i] = op.fn.execute(c[i], a[i]);
						lnnz += (c[i]!=0) ? 1 : 0;
					}
				}
			}
			else if(op.fn instanceof Multiply)
				ret.denseBlock.set(0);
			else
				lnnz = m1.nonZeros;
		}
		
		//3) recompute nnz
		ret.setNonZeros(lnnz);
	}
	
	private static void safeBinaryMMDenseDenseDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		ret.allocateDenseBlock();
		DenseBlock da = m1.getDenseBlock();
		DenseBlock db = m2.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		ValueFunction fn = op.fn;
		
		//compute dense-dense binary, maintain nnz on-the-fly
		long lnnz = 0;
		for( int bi=0; bi<da.numBlocks(); bi++ ) {
			double[] a = da.valuesAt(bi);
			double[] b = db.valuesAt(bi);
			double[] c = dc.valuesAt(bi);
			int len = da.size(bi);
			for( int i=0; i<len; i++ ) {
				c[i] = fn.execute(a[i], b[i]);
				lnnz += (c[i]!=0)? 1 : 0;
			}
		}
		ret.setNonZeros(lnnz);
	}
	
	private static void safeBinaryMMSparseDenseSkip(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		SparseBlock a = m1.sparse ? m1.sparseBlock : m2.sparseBlock;
		if( a == null )
			return;
		
		//prepare second input and allocate output
		MatrixBlock b = m1.sparse ? m2 : m1;
		ret.allocateBlock();
		
		for( int i=0; i<a.numRows(); i++ ) {
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
				ret.appendValue(i, aix[k], val);
			}
		}
	}
	
	private static void safeBinaryMMGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		int rlen = m1.rlen;
		int clen = m2.clen;
		for(int r=0; r<rlen; r++)
			for(int c=0; c<clen; c++) {
				double in1 = m1.quickGetValue(r, c);
				double in2 = m2.quickGetValue(r, c);
				if( in1==0 && in2==0) continue;
				double val = op.fn.execute(in1, in2);
				ret.appendValue(r, c, val);
			}
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
	private static void performBinOuterOperation(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator bOp) {
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
	}

	private static void unsafeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) {
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) //MATRIX - COL_VECTOR
		{
			for(int r=0; r<rlen; r++) {
				double v2 = m2.quickGetValue(r, 0);
				for(int c=0; c<clen; c++) {
					double v1 = m1.quickGetValue(r, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) //MATRIX - ROW_VECTOR
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++) {
					double v1 = m1.quickGetValue(r, c);
					double v2 = m2.quickGetValue(0, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);
				}
		}
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) //VECTOR - VECTOR
		{
			int clen2 = m2.clen; 
			
			if(LibMatrixOuterAgg.isCompareOperator(op) 
				&& m2.getNumColumns()>16 && SortUtils.isSorted(m2)) {
				performBinOuterOperation(m1, m2, ret, op);
			} 
			else {
				for(int r=0; r<rlen; r++) {
					double v1 = m1.quickGetValue(r, 0);
					for(int c=0; c<clen2; c++) {
						double v2 = m2.quickGetValue(0, c);
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(r, c, v);
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
				int lnnz = 0;
				for( int i=0; i<rlen; i++ ) {
					c[i] = op.fn.execute( a[i], b[i] );
					lnnz += (c[i] != 0) ? 1 : 0;
				}
				ret.nonZeros = lnnz;
			}
			//general case
			else 
			{
				for(int r=0; r<rlen; r++)
					for(int c=0; c<clen; c++) {
						double v1 = m1.quickGetValue(r, c);
						double v2 = m2.quickGetValue(r, c);
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(r, c, v);
					}
			}
		}
	}

	private static void safeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false) ) {
			return;
		}
		
		//sanity check input/output sparsity
		if( m1.sparse != ret.sparse )
			throw new DMLRuntimeException("Unsupported safe binary scalar operations over different input/output representation: "+m1.sparse+" "+ret.sparse);
		
		boolean copyOnes = (op.fn instanceof NotEquals && op.getConstant()==0);
		boolean allocExact = (op.fn instanceof Multiply || op.fn instanceof Multiply2 
			|| op.fn instanceof Power2 || Builtin.isBuiltinCode(op.fn, BuiltinCode.MAX)
			|| Builtin.isBuiltinCode(op.fn, BuiltinCode.MIN));
		
		if( m1.sparse ) //SPARSE <- SPARSE
		{
			//allocate sparse row structure
			ret.allocateSparseRowsBlock();
			SparseBlock a = m1.sparseBlock;
			SparseBlock c = ret.sparseBlock;
			int rlen = Math.min(m1.rlen, a.numRows());
			
			long nnz = 0;
			for(int r=0; r<rlen; r++) {
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
			ret.nonZeros = nnz;
		}
		else { //DENSE <- DENSE
			denseBinaryScalar(m1, ret, op);
		}
	}
	
	/**
	 * Since this operation is sparse-unsafe, ret should always be passed in dense representation.
	 * 
	 * @param m1 input matrix
	 * @param ret result matrix
	 * @param op scalar operator
	 */
	private static void unsafeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false) ) {
			//compute 0 op constant once and set into dense output
			double val = op.executeScalar(0);
			if( val != 0 )
				ret.reset(ret.rlen, ret.clen, val);
			return;
		}
		
		//sanity check input/output sparsity
		if( ret.sparse )
			throw new DMLRuntimeException("Unsupported unsafe binary scalar operations over sparse output representation.");
		
		if( m1.sparse ) //SPARSE MATRIX
		{
			ret.allocateDenseBlock();
			
			SparseBlock a = m1.sparseBlock;
			DenseBlock dc = ret.getDenseBlock();
			int m = m1.rlen;
			int n = m1.clen;
			
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
			ret.nonZeros = nnz;
		}
		else { //DENSE MATRIX
			denseBinaryScalar(m1, ret, op);
		}
	}

	private static void denseBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) {
		//allocate dense block (if necessary), incl clear nnz
		ret.allocateDenseBlock(true);
		
		DenseBlock da = m1.getDenseBlock();
		DenseBlock dc = ret.getDenseBlock();
		
		//compute scalar operation, incl nnz maintenance
		long nnz = 0;
		for( int bi=0; bi<da.numBlocks(); bi++) {
			double[] a = da.valuesAt(bi);
			double[] c = dc.valuesAt(bi);
			int limit = da.size(bi);
			for( int i=0; i<limit; i++ ) {
				c[i] = op.executeScalar( a[i] );
				nnz += (c[i] != 0) ? 1 : 0;
			}
		}
		ret.nonZeros = nnz;
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
}
