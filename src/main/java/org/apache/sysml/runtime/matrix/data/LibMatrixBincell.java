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

import java.util.Arrays;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Divide;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.LessThanEquals;
import org.apache.sysml.runtime.functionobjects.Minus;
import org.apache.sysml.runtime.functionobjects.MinusMultiply;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Multiply2;
import org.apache.sysml.runtime.functionobjects.NotEquals;
import org.apache.sysml.runtime.functionobjects.Or;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.PlusMultiply;
import org.apache.sysml.runtime.functionobjects.Power2;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.SortUtils;

/**
 * MB:
 * Library for binary cellwise operations (incl arithmetic, relational, etc). Currently,
 * we don't have dedicated support for the individual operations but for categories of
 * operations and combinations of dense/sparse and MM/MV. Safe/unsafe refer to sparse-safe
 * and sparse-unsafe operations.
 *  
 * 
 * 
 * TODO: custom operator implementations in order to turn unnecessarily sparse-unsafe
 * operations into sparse safe (e.g., relational operations)
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
	 * @param m1
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException
	 */
	public static void bincellOp(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) 
		throws DMLRuntimeException
	{
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
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException
	 */
	public static void bincellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException
	{
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
	 * @param m1ret
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	public static void bincellOpInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) 
		throws DMLRuntimeException
	{
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
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
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
	
	
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
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
	
	/**
	 * 
	 * @param op
	 * @param rhs
	 * @return
	 */
	public static boolean isSparseSafeDivide(BinaryOperator op, MatrixBlock rhs)
	{
		//if rhs is fully dense, there cannot be a /0 and hence DIV becomes sparse safe
		return (op.fn instanceof Divide && rhs.getNonZeros()==(long)rhs.getNumRows()*rhs.getNumColumns());
	}
	
	//////////////////////////////////////////////////////
	// private sparse-safe/sparse-unsafe implementations
	///////////////////////////////////

	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void safeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
		boolean skipEmpty = (op.fn instanceof Multiply 
				|| isSparseSafeDivide(op, m2) );
		
		//skip empty blocks (since sparse-safe)
		if(    m1.isEmptyBlock(false) && m2.isEmptyBlock(false) 
			|| skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)) ) 
		{
			return;
		}
	
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		if(    atype == BinaryAccessType.MATRIX_COL_VECTOR //MATRIX - VECTOR
			|| atype == BinaryAccessType.MATRIX_ROW_VECTOR)  
		{
			//note: m2 vector and hence always dense
			if( !m1.sparse && !m2.sparse && !ret.sparse ) //DENSE all
				safeBinaryMVDense(m1, m2, ret, op);
			else if( m1.sparse ) //SPARSE m1
				safeBinaryMVSparse(m1, m2, ret, op);
			else //generic combinations
				safeBinaryMVGeneric(m1, m2, ret, op);
		}	
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) //VECTOR - VECTOR
		{
			safeBinaryVVGeneric(m1, m2, ret, op);
		}
		else //MATRIX - MATRIX
		{
			if(m1.sparse && m2.sparse)
			{
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
						for(int r=0; r<rlen; r++)
						{
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
					
					for(int r=0; r<Math.min(rlen, rsblock.numRows()); r++)
						if( !rsblock.isEmpty(r) )
						{
							appendRightForSparseBinary(op, rsblock.values(r), rsblock.indexes(r), 
									rsblock.pos(r), rsblock.size(r), 0, r, ret);
						}
				}
				//left sparse block existing
				else
				{
					SparseBlock lsblock = m1.sparseBlock;
					
					for(int r=0; r<rlen; r++)
						if( !lsblock.isEmpty(r) )
						{
							appendLeftForSparseBinary(op, lsblock.values(r), lsblock.indexes(r), 
									lsblock.pos(r), lsblock.size(r), 0, r, ret);
						}
				}
			}
			else if( !ret.sparse && (m1.sparse || m2.sparse) &&
					(op.fn instanceof Plus || op.fn instanceof Minus ||
					op.fn instanceof PlusMultiply || op.fn instanceof MinusMultiply ||
					(op.fn instanceof Multiply && !m2.sparse )))
			{
				//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
				ret.allocateDenseBlock();
				final int m = ret.rlen;
				final int n = ret.clen;
				double[] c = ret.denseBlock;
				
				//1) process left input: assignment
				
				if( m1.sparse ) //SPARSE left
				{
					Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0); 
					
					if( m1.sparseBlock != null )
					{
						SparseBlock a = m1.sparseBlock;
						
						for( int i=0, ix=0; i<m; i++, ix+=n ) {
							if( !a.isEmpty(i) )
							{
								int apos = a.pos(i);
								int alen = a.size(i);
								int[] aix = a.indexes(i);
								double[] avals = a.values(i);
								for(int k = apos; k < apos+alen; k++) 
									c[ix+aix[k]] = avals[k];
							}
						}
					}
				}
				else //DENSE left
				{
					if( !m1.isEmptyBlock(false) ) 
						System.arraycopy(m1.denseBlock, 0, c, 0, m*n);
					else
						Arrays.fill(ret.denseBlock, 0, m*n, 0); 
				}
				
				//2) process right input: op.fn (+,-,*), * only if dense
				if( m2.sparse ) //SPARSE right
				{				
					if(m2.sparseBlock!=null)
					{
						SparseBlock a = m2.sparseBlock;
						
						for( int i=0, ix=0; i<m; i++, ix+=n ) {
							if( !a.isEmpty(i) ) {
								int apos = a.pos(i);
								int alen = a.size(i);
								int[] aix = a.indexes(i);
								double[] avals = a.values(i);
								for(int k = apos; k < apos+alen; k++) 
									c[ix+aix[k]] = op.fn.execute(c[ix+aix[k]], avals[k]);
							}
						}	
					}
				}
				else //DENSE right
				{
					if( !m2.isEmptyBlock(false) )
						for( int i=0; i<m*n; i++ )
							c[i] = op.fn.execute(c[i], m2.denseBlock[i]);
					else if(op.fn instanceof Multiply)
						Arrays.fill(ret.denseBlock, 0, m*n, 0); 
				}
	
				//3) recompute nnz
				ret.recomputeNonZeros();
			}
			else if( !ret.sparse && !m1.sparse && !m2.sparse 
					&& m1.denseBlock!=null && m2.denseBlock!=null )
			{
				ret.allocateDenseBlock();
				final int m = ret.rlen;
				final int n = ret.clen;
				double[] a = m1.denseBlock;
				double[] b = m2.denseBlock;
				double[] c = ret.denseBlock;
				ValueFunction fn = op.fn;
				
				//compute dense-dense binary, maintain nnz on-the-fly
				int nnz = 0;
				for( int i=0; i<m*n; i++ ) {
					c[i] = fn.execute(a[i], b[i]);
					nnz += (c[i]!=0)? 1 : 0;
				}
				ret.nonZeros = nnz;
			}
			else if( skipEmpty && (m1.sparse || m2.sparse) ) 
			{
				SparseBlock a = m1.sparse ? m1.sparseBlock : m2.sparseBlock;
				if( a != null ) {
					MatrixBlock b = m1.sparse ? m2 : m1;
					for( int i=0; i<a.numRows(); i++ ) {
						if( a.isEmpty(i) ) continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for(int k = apos; k < apos+alen; k++) {
							double in2 = b.quickGetValue(i, aix[k]);
							if( in2==0 ) continue;
							double val = op.fn.execute(avals[k], in2);
							ret.appendValue(i, aix[k], val);
						}
					}
				}
			}
			else //generic case
			{
				for(int r=0; r<rlen; r++)
					for(int c=0; c<clen; c++) {
						double in1 = m1.quickGetValue(r, c);
						double in2 = m2.quickGetValue(r, c);
						if( in1==0 && in2==0) continue;
						double val = op.fn.execute(in1, in2);
						ret.appendValue(r, c, val);
					}
			}
		}
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException 
	 */
	private static void safeBinaryMVDense(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		int rlen = m1.rlen;
		int clen = m1.clen;
		
		//early abort on skip and empy
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		ret.allocateDenseBlock();
		double[] a = m1.denseBlock;
		double[] b = m2.denseBlock;
		double[] c = ret.denseBlock;
		int nnz = 0;
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=0, ix=0; i<rlen; i++, ix+=clen )
			{
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
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			if( a==null && b==null ) { //both empty
				double v = op.fn.execute( 0, 0 );
				Arrays.fill(c, 0, rlen*clen, v);
				nnz += (v != 0) ? rlen*clen : 0;
			}
			else if( a==null ) //left empty
			{
				//compute first row
				for( int j=0; j<clen; j++ ) {
					c[j] = op.fn.execute( 0, b[j] );
					nnz += (c[j] != 0) ? rlen : 0;
				}
				//copy first to all other rows
				for( int i=1, ix=clen; i<rlen; i++, ix+=clen )
					System.arraycopy(c, 0, c, ix, clen);
			}
			else //default case (incl right empty) 
			{
				for( int i=0, ix=0; i<rlen; i++, ix+=clen )
					for( int j=0; j<clen; j++ ) {
						c[ix+j] = op.fn.execute( a[ix+j], ((b!=null) ? b[j] : 0) );	
						nnz += (c[ix+j] != 0) ? 1 : 0;
					}
			}
		}
		
		ret.nonZeros = nnz;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException 
	 */
	private static void safeBinaryMVSparse(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
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
			for( int i=0; i<rlen; i++ )
			{
				double v2 = m2.quickGetValue(i, 0);
				
				if( (skipEmpty && (a==null || a.isEmpty(i) || v2 == 0 ))
					|| ((a==null || a.isEmpty(i)) && v2 == 0) )
				{
					continue; //skip empty rows
				}
					
				if( isMultiply && v2==1 ) //ROW COPY
				{
					if( a != null && !a.isEmpty(i)  )
						ret.appendRow(i, a.get(i));
				}
				else //GENERAL CASE
				{
					int lastIx = -1;
					if( a != null && !a.isEmpty(i) ) 
					{
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for( int j=apos; j<apos+alen; j++ )
						{
							//empty left
							for( int k = lastIx+1; k<aix[j]; k++ ){
								double v = op.fn.execute( 0, v2 );
								ret.appendValue(i, k, v);
							}
							//actual value
							double v = op.fn.execute( avals[j], v2 );
							ret.appendValue(i, aix[j], v);	
							lastIx = aix[j];
						}
					}
					
					//empty left
					for( int k = lastIx+1; k<clen; k++ ){
						double v = op.fn.execute( 0, v2 );
						ret.appendValue(i, k, v);
					}
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			for( int i=0; i<rlen; i++ )
			{
				if( skipEmpty && (a==null || a.isEmpty(i)) )
					continue; //skip empty rows
					
				int lastIx = -1;
				if( a!=null && !a.isEmpty(i) ) 
				{
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ )
					{
						//empty left
						for( int k=lastIx+1; !skipEmpty&&k<aix[j]; k++ ){
							double v2 = m2.quickGetValue(0, k);
							double v = op.fn.execute( 0, v2 );
							ret.appendValue(i, k, v);
						}
						//actual value
						double v2 = m2.quickGetValue(0, aix[j]);
						double v = op.fn.execute( avals[j], v2 );
						ret.appendValue(i, aix[j], v);	
						lastIx = aix[j];
					}
				}
				
				//empty left
				for( int k=lastIx+1; !skipEmpty&&k<clen; k++ ){
					double v2 = m2.quickGetValue(0, k);
					double v = op.fn.execute( 0, v2 );
					ret.appendValue(i, k, v);
				}
			}
		}
		
		//no need to recomputeNonZeros since maintained in append value
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException 
	 */
	private static void safeBinaryMVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
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
					for( int j=0; j<clen; j++ )
					{
						double v1 = m1.quickGetValue(i, j);
						ret.appendValue(i, j, v1);		
					}
				}
				else //GENERAL CASE
				{
					for( int j=0; j<clen; j++ )
					{
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
			if( m2.sparse && isMultiply ) //SPARSE *
			{
				//note: sparse block guaranteed to be allocated (otherwise early about)
				SparseBlock b = m2.sparseBlock;
				if( !b.isEmpty(0) ) 
				{
					int blen = b.size(0); //always pos 0
					int[] bix = b.indexes(0);
					double[] bvals = b.values(0);
					for( int i=0; i<rlen; i++ ) {
						//for each row iterate only over non-zeros elements in rhs
						for( int j=0; j<blen; j++ ) {
							double v1 = m1.quickGetValue(i, bix[j]);
							double v = op.fn.execute( v1, bvals[j] );
							ret.appendValue(i, bix[j], v);					
						}
					}
				}
			}
			else //GENERAL CASE
			{
				for( int i=0; i<rlen; i++ )
					for( int j=0; j<clen; j++ )
					{
						double v1 = m1.quickGetValue(i, j);
						double v2 = m2.quickGetValue(0, j); //replicated vector value
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(i, j, v);		
					}
			}
		}
			
		//no need to recomputeNonZeros since maintained in append value
	}
	
	private static void safeBinaryVVGeneric(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
		int rlen = m1.rlen;
		int clen = m2.clen;
		
		//allocate once in order to prevent repeated reallocation 
		if( ret.sparse )
			ret.allocateSparseRowsBlock();
		
		if(LibMatrixOuterAgg.isCompareOperator(op) && SortUtils.isSorted(0, m2.getNumColumns(), DataConverter.convertToDoubleVector(m2))) {
			performBinOuterOperation(m1, m2, ret, op);
		} else {
			for(int r=0; r<rlen; r++) {
				double v1 = m1.quickGetValue(r, 0);		
				for(int c=0; c<clen; c++)
				{
					double v2 = m2.quickGetValue(0, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);	
				}
			}	
		}	
			
		//no need to recomputeNonZeros since maintained in append value
	}
	
	/**
	 * 
	 * This will do cell wise operation for <,<=, >, >=, == and != operators. 
	 * 
	 * @param mbLeft
	 * @param mbRight
	 * @param mbOut
	 * @param bOp
	 * 
	 */
	private static void performBinOuterOperation(MatrixBlock mbLeft, MatrixBlock mbRight, MatrixBlock mbOut, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int rlen = mbLeft.rlen;
		double bv[] = DataConverter.convertToDoubleVector(mbRight); 
		
		if(!mbOut.isAllocated())
			mbOut.allocateDenseBlock();
		
		long lNNZ = 0;
		for(int r=0; r<rlen; r++) {
			double value = mbLeft.quickGetValue(r, 0);		
			int ixPos1 = Arrays.binarySearch(bv, value);
			int ixPos2 = ixPos1;

			if( ixPos1 >= 0 ){ //match, scan to next val
				if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals 
						|| bOp.fn instanceof Equals || bOp.fn instanceof NotEquals)
					while( ixPos1<bv.length && value==bv[ixPos1]  ) ixPos1++;
				if(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals 
						|| bOp.fn instanceof Equals || bOp.fn instanceof NotEquals)
					while(  ixPos2 > 0 && value==bv[ixPos2-1]) --ixPos2;
			} else {
				ixPos2 = ixPos1 = Math.abs(ixPos1) - 1;
			}

			int iStartPos = 0, iEndPos = bv.length;

			if(bOp.fn instanceof LessThan)
				iStartPos = ixPos1;
			else  if(bOp.fn instanceof LessThanEquals)
				iStartPos = ixPos2;  
			else if(bOp.fn instanceof GreaterThan)
				iEndPos = ixPos2;
			else if(bOp.fn instanceof GreaterThanEquals)
				iEndPos = ixPos1;
			else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
				iStartPos = ixPos2;
				iEndPos = ixPos1;
			}
			if(iStartPos < iEndPos || bOp.fn instanceof NotEquals) {
				int iOffSet = r*mbRight.getNumColumns();
				if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals 
						|| bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals 
						|| bOp.fn instanceof Equals)	{
					Arrays.fill(mbOut.getDenseBlock(), iOffSet+iStartPos, iOffSet+iEndPos, 1.0);
					lNNZ += (iEndPos-iStartPos);
				}
				else if (bOp.fn instanceof NotEquals) {
					Arrays.fill(mbOut.getDenseBlock(), iOffSet, iOffSet+iStartPos, 1.0);
					Arrays.fill(mbOut.getDenseBlock(), iOffSet+iEndPos, iOffSet+bv.length, 1.0);
					lNNZ += (iStartPos+(bv.length-iEndPos));
				}
			}
		}
		mbOut.setNonZeros(lNNZ);		
		mbOut.examSparsity();
	}
			

	
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void unsafeBinary(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException 
	{
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) //MATRIX - COL_VECTOR
		{
			for(int r=0; r<rlen; r++)
			{
				//replicated value
				double v2 = m2.quickGetValue(r, 0);
				
				for(int c=0; c<clen; c++)
				{
					double v1 = m1.quickGetValue(r, c);	
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) //MATRIX - ROW_VECTOR
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					double v1 = m1.quickGetValue(r, c);	
					double v2 = m2.quickGetValue(0, c);
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(r, c, v);
				}
		}
		else if( atype == BinaryAccessType.OUTER_VECTOR_VECTOR ) //VECTOR - VECTOR
		{
			int clen2 = m2.clen; 
			
			//TODO performance improvement for relational operations like ">"
			//sort rhs by val, compute cutoff and memset 1/0 for halfs
	
			if(LibMatrixOuterAgg.isCompareOperator(op) && SortUtils.isSorted(0, m2.getNumColumns(), DataConverter.convertToDoubleVector(m2))) {
				performBinOuterOperation(m1, m2, ret, op);
			} else {
				for(int r=0; r<rlen; r++) {
					double v1 = m1.quickGetValue(r, 0);		
					for(int c=0; c<clen2; c++)
					{
						double v2 = m2.quickGetValue(0, c);
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(r, c, v);	
					}
				}
			}
		}
		else // MATRIX - MATRIX
		{
			//dense non-empty vectors
			if( m1.clen==1 && !m1.sparse && !m1.isEmptyBlock(false)   
				&& !m2.sparse && !m2.isEmptyBlock(false)  )
			{
				ret.allocateDenseBlock();
				double[] a = m1.denseBlock;
				double[] b = m2.denseBlock;
				double[] c = ret.denseBlock;
				for( int i=0; i<rlen; i++ ) {
					c[i] = op.fn.execute( a[i], b[i] );
					if( c[i] != 0 ) 
						ret.nonZeros++;
				}
			}
			//general case
			else 
			{
				for(int r=0; r<rlen; r++)
					for(int c=0; c<clen; c++)
					{
						double v1 = m1.quickGetValue(r, c);
						double v2 = m2.quickGetValue(r, c);
						double v = op.fn.execute( v1, v2 );
						ret.appendValue(r, c, v);
					}
			}
		}
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void safeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op)
		throws DMLRuntimeException
	{
		//early abort possible since sparsesafe
		if( m1.isEmptyBlock(false) ) {
			return;
		}
		
		//sanity check input/output sparsity
		if( m1.sparse != ret.sparse )
			throw new DMLRuntimeException("Unsupported safe binary scalar operations over different input/output representation: "+m1.sparse+" "+ret.sparse);
		
		boolean copyOnes = (op.fn instanceof NotEquals && op.getConstant()==0);
		
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
					SparseRow crow = new SparseRow(alen);
					crow.setSize(alen);
					
					//memcopy/memset of indexes/values (sparseblock guarantees absence of 0s) 
					System.arraycopy(aix, apos, crow.indexes(), 0, alen);
					Arrays.fill(crow.values(), 0, alen, 1);
					c.set(r, crow, false);
					nnz += alen;
				}
				else { //GENERAL CASE
					//create sparse row without repeated resizing for specific ops
					if( op.fn instanceof Multiply || op.fn instanceof Multiply2 
						|| op.fn instanceof Power2  ) {
						c.allocate(r, alen);
					}
					
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
	 * @param m1
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void unsafeBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op)
		throws DMLRuntimeException
	{
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
			double[] c = ret.denseBlock;			
			int m = m1.rlen;
			int n = m1.clen;
			
			//init dense result with unsafe 0-value
			double cval0 = op.executeScalar(0);
			Arrays.fill(c, cval0);
			
			//compute non-zero input values
			int nnz = m*n;
			for(int i=0, cix=0; i<m; i++, cix+=n) {
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for(int j=apos; j<apos+alen; j++) {
						double val = op.executeScalar(avals[j]);
						c[ cix+aix[j] ] = val;
						nnz -= (val==0) ? 1 : 0;
					}
				}
			}
			ret.nonZeros = nnz;
		}
		else { //DENSE MATRIX
			denseBinaryScalar(m1, ret, op);
		}
	}

	/**
	 * 
	 * @param m1
	 * @param ret
	 * @param op
	 * @throws DMLRuntimeException 
	 */
	private static void denseBinaryScalar(MatrixBlock m1, MatrixBlock ret, ScalarOperator op) 
		throws DMLRuntimeException 
	{
		//allocate dense block (if necessary), incl clear nnz
		ret.allocateDenseBlock(true);
		
		double[] a = m1.denseBlock;
		double[] c = ret.denseBlock;
		
		//compute scalar operation, incl nnz maintenance
		int limit = m1.rlen*m1.clen;
		int nnz = 0;
		for( int i=0; i<limit; i++ ) {
			c[i] = op.executeScalar( a[i] );
			nnz += (c[i] != 0) ? 1 : 0;
		}
		ret.nonZeros = nnz;
	}
	
	/**
	 * 
	 * @param m1ret
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void safeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) 
		throws DMLRuntimeException 
	{
		//early abort on skip and empty 
		if( m1ret.isEmptyBlock(false) && m2.isEmptyBlock(false) )
			return; // skip entire empty block
		//special case: start aggregation
		else if( op.fn instanceof Plus && m1ret.isEmptyBlock(false) ){
			m1ret.copy(m2);
			return; 
		}
		
		int rlen = m1ret.rlen;
		int clen = m1ret.clen;
		
		if(m1ret.sparse && m2.sparse)
		{
			if(m1ret.sparseBlock!=null)
				m1ret.allocateSparseRowsBlock(false);
			if(m2.sparseBlock!=null)
				m2.allocateSparseRowsBlock(false);
			
			SparseBlock c = m1ret.sparseBlock;
			SparseBlock b = m2.sparseBlock;
			
			if( c!=null && b!=null )
			{
				for(int r=0; r<rlen; r++)
				{
					if(c.isEmpty(r) && b.isEmpty(r))
						continue;
					
					if( b.isEmpty(r) )
					{
						int apos = c.pos(r);
						int alen = c.size(r);
						double[] values=c.values(r);
						for(int i=apos; i<apos+alen; i++)
							values[i]=op.fn.execute(values[i], 0);
					}else
					{
						int estimateSize=0;
						if( !c.isEmpty(r) )
							estimateSize+=c.size(r);
						if( !b.isEmpty(r))
							estimateSize+=b.size(r);
						estimateSize=Math.min(clen, estimateSize);
						
						//temp
						SparseRow thisRow = c.get(r);
						c.set(r, new SparseRow(estimateSize, clen), false);
						
						if(thisRow!=null)
						{
							m1ret.nonZeros-=thisRow.size();
							mergeForSparseBinary(op, thisRow.values(), thisRow.indexes(), 0, 
									thisRow.size(), b.values(r), b.indexes(r), b.pos(r), b.size(r), r, m1ret);
							
						}
						else
						{
							appendRightForSparseBinary(op, b.values(r), b.indexes(r), b.pos(r), b.size(r), 0, r, m1ret);
						}
					}
				}	
			}
			else if(m1ret.sparseBlock==null)
			{
				m1ret.sparseBlock = SparseBlockFactory.createSparseBlock(rlen);
				
				for(int r=0; r<rlen; r++)
				{
					if( !b.isEmpty(r) ) {
						SparseRow tmp = new SparseRow( b.size(r), clen );
						appendRightForSparseBinary(op, b.values(r), b.indexes(r), b.pos(r), b.size(r), 0, r, m1ret);
						m1ret.sparseBlock.set(r, tmp, false);
					}
				}				
			}
			else //that.sparseRows==null
			{
				if( !(op.fn instanceof Plus || op.fn instanceof Minus || op.fn instanceof Or) ) {
					for(int r=0; r<rlen; r++){
						if( !c.isEmpty(r) )
						{
							SparseRow tmp = c.get(r);
							int alen = tmp.size();
							double[] avals = tmp.values();
							for( int j=0; j<alen; j++ )
								avals[j] = op.fn.execute(avals[j], 0);
							tmp.compact(); //handle removed entries (e.g., mult, and)
							c.set(r, tmp, false);
							
							//NOTE: for left in-place, we cannot use append because it would create duplicates
							//appendLeftForSparseBinary(op, arow.getValueContainer(), arow.getIndexContainer(), arow.size(), 0, r, m1ret);
						}
					}
				}
			}
		}
		else //one side dense
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					double thisvalue = m1ret.quickGetValue(r, c);
					double thatvalue = m2.quickGetValue(r, c);
					double resultvalue = op.fn.execute(thisvalue, thatvalue);
					m1ret.quickSetValue(r, c, resultvalue);
				}	
		}
	}
	
	/**
	 * 
	 * @param m1ret
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void unsafeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) throws DMLRuntimeException 
	{
		int rlen = m1ret.rlen;
		int clen = m1ret.clen;
		BinaryAccessType atype = getBinaryAccessType(m1ret, m2);
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR ) //MATRIX - COL_VECTOR
		{
			for(int r=0; r<rlen; r++)
			{
				//replicated value
				double v2 = m2.quickGetValue(r, 0);
				
				for(int c=0; c<clen; c++)
				{
					double v1 = m1ret.quickGetValue(r, c);	
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR ) //MATRIX - ROW_VECTOR
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					double v1 = m1ret.quickGetValue(r, c);
					double v2 = m2.quickGetValue(0, c); //replicated value
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
		}
		else // MATRIX - MATRIX
		{
			for(int r=0; r<rlen; r++)
				for(int c=0; c<clen; c++)
				{
					double v1 = m1ret.quickGetValue(r, c);
					double v2 = m2.quickGetValue(r, c);
					double v = op.fn.execute( v1, v2 );
					m1ret.quickSetValue(r, c, v);
				}
		}
	}
	
	/**
	 * * like a merge sort
	 * 
	 * @param op
	 * @param values1
	 * @param cols1
	 * @param size1
	 * @param values2
	 * @param cols2
	 * @param size2
	 * @param resultRow
	 * @param result
	 * @throws DMLRuntimeException
	 */
	private static void mergeForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1, 
				double[] values2, int[] cols2, int pos2, int size2, int resultRow, MatrixBlock result) 
		throws DMLRuntimeException
	{
		int p1=0, p2=0, column;
		while( p1<size1 && p2< size2 )
		{
			double value = 0;
			if(cols1[pos1+p1]<cols2[pos2+p2]) {
				value = op.fn.execute(values1[pos1+p1], 0);
				column = cols1[pos1+p1];
				p1++;
			}
			else if(cols1[pos1+p1]==cols2[pos2+p2]) {
				value = op.fn.execute(values1[pos1+p1], values2[pos2+p2]);
				column = cols1[pos1+p1];
				p1++;
				p2++;
			}
			else {
				value = op.fn.execute(0, values2[pos2+p2]);
				column = cols2[pos2+p2];
				p2++;
			}
			result.appendValue(resultRow, column, value);	
		}
		
		//add left over
		appendLeftForSparseBinary(op, values1, cols1, pos1, size1, p1, resultRow, result);
		appendRightForSparseBinary(op, values2, cols2, pos2, size2, p2, resultRow, result);
	}
	
	/**
	 * 
	 * @param op
	 * @param values1
	 * @param cols1
	 * @param pos1
	 * @param size1
	 * @param pos
	 * @param resultRow
	 * @param result
	 * @throws DMLRuntimeException
	 */
	private static void appendLeftForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int pos1, int size1, 
				int pos, int resultRow, MatrixBlock result) 
		throws DMLRuntimeException
	{
		for(int j=pos1+pos; j<pos1+size1; j++) {
			double v = op.fn.execute(values1[j], 0);
			result.appendValue(resultRow, cols1[j], v);
		}
	}
	
	/**
	 * 
	 * @param op
	 * @param values2
	 * @param cols2
	 * @param pos2
	 * @param size2
	 * @param pos
	 * @param resultRow
	 * @param result
	 * @throws DMLRuntimeException
	 */
	private static void appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int pos2, int size2, 
		int pos, int resultRow, MatrixBlock result) throws DMLRuntimeException
	{
		for( int j=pos2+pos; j<pos2+size2; j++ ) {
			double v = op.fn.execute(0, values2[j]);
			result.appendValue(resultRow, cols2[j], v);
		}
	}
	
}

