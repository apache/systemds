/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.matrix.data;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;

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
		boolean isMultiply = (op.fn instanceof Multiply);
		boolean skipEmpty = (isMultiply);
		
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
				if(m1.sparseRows!=null && m2.sparseRows!=null)
				{
					for(int r=0; r<rlen; r++)
					{
						SparseRow lrow = (m1.sparseRows.length>r && m1.sparseRows[r]!=null) ? m1.sparseRows[r] : null; 
						SparseRow rrow = (m2.sparseRows.length>r && m2.sparseRows[r]!=null) ? m2.sparseRows[r] : null; 
						
						if( lrow!=null && rrow!=null)
						{
							mergeForSparseBinary(op, lrow.getValueContainer(), lrow.getIndexContainer(), lrow.size(),
									rrow.getValueContainer(), rrow.getIndexContainer(), rrow.size(), r, ret);	
						}
						else if( rrow!=null )
						{
							appendRightForSparseBinary(op, rrow.getValueContainer(), 
									rrow.getIndexContainer(), rrow.size(), 0, r, ret);
						}
						else if( lrow!=null )
						{
							appendLeftForSparseBinary(op, lrow.getValueContainer(), 
									lrow.getIndexContainer(), lrow.size(), 0, r, ret);
						}
						
						// do nothing if both not existing
					}
				}
				//right sparse block existing
				else if( m2.sparseRows!=null )
				{
					for(int r=0; r<Math.min(rlen, m2.sparseRows.length); r++)
						if(m2.sparseRows[r]!=null)
						{
							appendRightForSparseBinary(op, m2.sparseRows[r].getValueContainer(), 
									m2.sparseRows[r].getIndexContainer(), m2.sparseRows[r].size(), 0, r, ret);
						}
				}
				//left sparse block existing
				else
				{
					for(int r=0; r<rlen; r++)
						if( m1.sparseRows[r]!=null )
						{
							appendLeftForSparseBinary(op, m1.sparseRows[r].getValueContainer(), 
									m1.sparseRows[r].getIndexContainer(), m1.sparseRows[r].size(), 0, r, ret);
						}
				}
			}
			else if( !ret.sparse && (m1.sparse || m2.sparse) &&
					(op.fn instanceof Plus || op.fn instanceof Minus || 
					(op.fn instanceof Multiply && !m2.sparse )))
			{
				//specific case in order to prevent binary search on sparse inputs (see quickget and quickset)
				ret.allocateDenseBlock();
				final int m = ret.rlen;
				final int n = ret.clen;
				double[] c = ret.denseBlock;
				
				//1) process left input: assignment
				int alen;
				int[] aix;
				double[] avals;
				
				if( m1.sparse ) //SPARSE left
				{
					Arrays.fill(ret.denseBlock, 0, ret.denseBlock.length, 0); 
					
					if( m1.sparseRows != null )
					{
						for( int i=0, ix=0; i<m; i++, ix+=n ) {
							SparseRow arow = m1.sparseRows[i];
							if( arow != null && !arow.isEmpty() )
							{
								alen = arow.size();
								aix = arow.getIndexContainer();
								avals = arow.getValueContainer();
								for(int k = 0; k < alen; k++) 
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
					if(m2.sparseRows!=null)
					{
						for( int i=0, ix=0; i<m; i++, ix+=n ) {
							SparseRow arow = m2.sparseRows[i];
							if( arow != null && !arow.isEmpty() )
							{
								alen = arow.size();
								aix = arow.getIndexContainer();
								avals = arow.getValueContainer();
								for(int k = 0; k < alen; k++) 
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
			else if( !ret.sparse && !m1.sparse && !m2.sparse && m1.denseBlock!=null && m2.denseBlock!=null )
			{
				ret.allocateDenseBlock();
				final int m = ret.rlen;
				final int n = ret.clen;
				double[] c = ret.denseBlock;
				
				//int nnz = 0;
				for( int i=0; i<m*n; i++ )
				{
					c[i] = op.fn.execute(m1.denseBlock[i], m2.denseBlock[i]);
					//HotSpot JVM bug causes crash in presence of NaNs 
					//nnz += (c[i]!=0)? 1 : 0;
					if( c[i] != 0 )
						ret.nonZeros++;
				}
				//result.nonZeros = nnz;
			}
			else //generic case
			{
				double thisvalue, thatvalue, resultvalue;
				for(int r=0; r<rlen; r++)
					for(int c=0; c<clen; c++)
					{
						thisvalue=m1.quickGetValue(r, c);
						thatvalue=m2.quickGetValue(r, c);
						if(thisvalue==0 && thatvalue==0)
							continue;
						resultvalue=op.fn.execute(thisvalue, thatvalue);
						ret.appendValue(r, c, resultvalue);
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

		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=0, ix=0; i<rlen; i++, ix+=clen )
			{
				//replicate vector value
				double v2 = (b==null) ? 0 : b[i];
				if( skipEmpty && v2 == 0 ) //skip empty rows
					continue;
					
				if( isMultiply && v2 == 1 ) //ROW COPY
				{
					//a guaranteed to be non-null (see early abort)
					System.arraycopy(a, ix, c, ix, clen);
				}
				else //GENERAL CASE
				{
					if( a != null )
						for( int j=0; j<clen; j++ )
							c[ix+j] = op.fn.execute( a[ix+j], v2 );	
					else
						Arrays.fill(c, ix, ix+clen, op.fn.execute( 0, v2 ));	
				}
			}
		}
		else if( atype == BinaryAccessType.MATRIX_ROW_VECTOR )
		{
			if( a==null && b==null ) //both empty
			{
				double v = op.fn.execute( 0, 0 );
				Arrays.fill(c, 0, rlen*clen, v);
			}
			else if( a==null ) //left empty
			{
				//compute first row
				for( int j=0; j<clen; j++ )
					c[j] = op.fn.execute( 0, b[j] );
				//copy first to all other rows
				for( int i=1, ix=clen; i<rlen; i++, ix+=clen )
					System.arraycopy(c, 0, c, ix, clen);
			}
			else //default case (incl right empty) 
			{
				for( int i=0, ix=0; i<rlen; i++, ix+=clen )
					for( int j=0; j<clen; j++ )
						c[ix+j] = op.fn.execute( a[ix+j], ((b!=null) ? b[j] : 0) );	
			}
		}
		
		ret.recomputeNonZeros();
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
		SparseRow[] a = m1.sparseRows;
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
				SparseRow arow = (a==null) ? null : a[i];
				
				if( (skipEmpty && (arow==null || arow.isEmpty() || v2 == 0 ))
					|| ((arow==null || arow.isEmpty()) && v2 == 0) )
				{
					continue; //skip empty rows
				}
					
				if( isMultiply && v2==1 ) //ROW COPY
				{
					if( arow != null && !arow.isEmpty()  )
						ret.appendRow(i, arow);
				}
				else //GENERAL CASE
				{
					int lastIx = -1;
					if( arow != null && !arow.isEmpty() ) 
					{
						int alen = arow.size();
						int[] aix = arow.getIndexContainer();
						double[] avals = arow.getValueContainer();
						for( int j=0; j<alen; j++ )
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
				SparseRow arow = (a==null) ? null : a[i];
				
				if( skipEmpty && (arow==null || arow.isEmpty()) )
					continue; //skip empty rows
					
				int lastIx = -1;
				if( arow != null && !arow.isEmpty() ) 
				{
					int alen = arow.size();
					int[] aix = arow.getIndexContainer();
					double[] avals = arow.getValueContainer();
					for( int j=0; j<alen; j++ )
					{
						//empty left
						for( int k = lastIx+1; k<aix[j]; k++ ){
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
				for( int k = lastIx+1; k<clen; k++ ){
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
				SparseRow brow = m2.sparseRows[0];
				if( brow != null && !brow.isEmpty() ) 
				{
					int blen = brow.size();
					int[] bix = brow.getIndexContainer();
					double[] bvals = brow.getValueContainer();
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
		
		//TODO performance improvement for relational operations like ">"
		//sort rhs by val, compute cutoff and memset 1/0 for halfs

		for(int r=0; r<rlen; r++) {
			double v1 = m1.quickGetValue(r, 0);		
			for(int c=0; c<clen; c++)
			{
				double v2 = m2.quickGetValue(0, c);
				double v = op.fn.execute( v1, v2 );
				ret.appendValue(r, c, v);	
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
		else // MATRIX - MATRIX
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
			SparseRow[] a = m1.sparseRows;
			SparseRow[] c = ret.sparseRows;
			
			for(int r=0; r<Math.min(m1.rlen, m1.sparseRows.length); r++) {
				if( a[r]!=null && !a[r].isEmpty() )
				{
					int alen = a[r].size();
					int[] aix = a[r].getIndexContainer();
					double[] avals = a[r].getValueContainer();
					
					if( copyOnes ) //SPECIAL CASE: e.g., (X != 0) 
					{
						//create sparse row without repeated resizing
						SparseRow crow = new SparseRow(alen);
						crow.setSize(alen);
						
						//memcopy/memset of indexes and values
						//note: currently we do a safe copy of values because in special cases there
						//might exist zeros in a sparserow and we need to ensure result correctness
						System.arraycopy(aix, 0, crow.getIndexContainer(), 0, alen);
						//Arrays.fill(crow.getValueContainer(), 0, alen, 1);
						double[] cvals = crow.getValueContainer();
						for(int j=0; j<alen; j++)
							cvals[j] = (avals[j] != 0) ? 1 : 0;
						c[r] = crow;
						ret.nonZeros+=alen;
					}
					else //GENERAL CASE
					{
						for(int j=0; j<alen; j++) {
							double val = op.executeScalar(avals[j]);
							ret.appendValue(r, aix[j], val);
						}
					}
				}
			}
		}
		else //DENSE <- DENSE
		{
			//allocate dense block
			ret.allocateDenseBlock(true);
		
			double[] a = m1.denseBlock;
			double[] c = ret.denseBlock;
			
			int limit = m1.rlen*m1.clen;
			for( int i=0; i<limit; i++ )
			{
				c[i] = op.executeScalar( a[i] );
				if( c[i] != 0 )
					ret.nonZeros++;
			}
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
				ret.init(val, ret.rlen, ret.clen);
			return;
		}
		
		//sanity check input/output sparsity
		if( ret.sparse )
			throw new DMLRuntimeException("Unsupported unsafe binary scalar operations over sparse output representation.");
		
		if( m1.sparse ) //SPARSE MATRIX
		{
			ret.allocateDenseBlock();
			
			SparseRow[] a = m1.sparseRows;
			double[] c = ret.denseBlock;			
			int m = m1.rlen;
			int n = m1.clen;
			
			//init dense result with unsafe 0-value
			double cval0 = op.executeScalar(0);
			Arrays.fill(c, cval0);
			
			//compute non-zero input values
			for(int i=0, cix=0; i<m; i++, cix+=n) 
			{
				if( a[i]!=null && !a[i].isEmpty() )
				{
					int alen = a[i].size();
					int[] aix = a[i].getIndexContainer();
					double[] avals = a[i].getValueContainer();
					for(int j=0; j<alen; j++) {
						double val = op.executeScalar(avals[j]);
						c[ cix+aix[j] ] = val;
					}
				}
			}
		
			//recompute non zeros 
			ret.recomputeNonZeros();
		}
		else //DENSE MATRIX
		{
			//allocate dense block (if necessary), incl clear nnz
			ret.allocateDenseBlock(true);
			
			double[] a = m1.denseBlock;
			double[] c = ret.denseBlock;
			
			//compute scalar operation, incl nnz maintenance
			int limit = m1.rlen*m1.clen;
			for( int i=0; i<limit; i++ )
			{
				c[i] = op.executeScalar( a[i] );
				if( c[i] != 0 )
					ret.nonZeros++;
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
			if(m1ret.sparseRows!=null)
				m1ret.allocateSparseRowsBlock(false);
			if(m2.sparseRows!=null)
				m2.allocateSparseRowsBlock(false);
			
			if(m1ret.sparseRows!=null && m2.sparseRows!=null)
			{
				for(int r=0; r<rlen; r++)
				{
					if(m1ret.sparseRows[r]==null && m2.sparseRows[r]==null)
						continue;
					
					if(m2.sparseRows[r]==null)
					{
						double[] values=m1ret.sparseRows[r].getValueContainer();
						for(int i=0; i<m1ret.sparseRows[r].size(); i++)
							values[i]=op.fn.execute(values[i], 0);
					}else
					{
						int estimateSize=0;
						if(m1ret.sparseRows[r]!=null)
							estimateSize+=m1ret.sparseRows[r].size();
						if(m2.sparseRows[r]!=null)
							estimateSize+=m2.sparseRows[r].size();
						estimateSize=Math.min(clen, estimateSize);
						
						//temp
						SparseRow thisRow=m1ret.sparseRows[r];
						m1ret.sparseRows[r]=new SparseRow(estimateSize, clen);
						
						if(thisRow!=null)
						{
							m1ret.nonZeros-=thisRow.size();
							mergeForSparseBinary(op, thisRow.getValueContainer(), 
									thisRow.getIndexContainer(), thisRow.size(),
									m2.sparseRows[r].getValueContainer(), 
									m2.sparseRows[r].getIndexContainer(), m2.sparseRows[r].size(), r, m1ret);
							
						}else
						{
							appendRightForSparseBinary(op, m2.sparseRows[r].getValueContainer(), 
									m2.sparseRows[r].getIndexContainer(), m2.sparseRows[r].size(), 0, r, m1ret);
						}
					}
				}	
			}
			else if(m1ret.sparseRows==null)
			{
				m1ret.sparseRows=new SparseRow[rlen];
				for(int r=0; r<rlen; r++)
				{
					SparseRow brow = m2.sparseRows[r];
					if( brow!=null && !brow.isEmpty() )
					{
						m1ret.sparseRows[r] = new SparseRow( brow.size(), clen );
						appendRightForSparseBinary(op, brow.getValueContainer(), brow.getIndexContainer(), brow.size(), 0, r, m1ret);
					}
				}				
			}
			else //that.sparseRows==null
			{
				if( !(op.fn instanceof Plus || op.fn instanceof Minus || op.fn instanceof Or) ){
					for(int r=0; r<rlen; r++){
						SparseRow arow = m1ret.sparseRows[r];
						if( arow!=null && !arow.isEmpty() )
						{
							int alen = arow.size();
							double[] avals = arow.getValueContainer();
							for( int j=0; j<alen; j++ )
								avals[j] = op.fn.execute(avals[j], 0);
							arow.compact(); //handle removed entries (e.g., mult, and)
							
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
	private static void mergeForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int size1, 
				double[] values2, int[] cols2, int size2, int resultRow, MatrixBlock result) 
		throws DMLRuntimeException
	{
		int p1=0, p2=0, column;
		while( p1<size1 && p2< size2 )
		{
			double value = 0;
			if(cols1[p1]<cols2[p2])
			{
				value = op.fn.execute(values1[p1], 0);
				column = cols1[p1];
				p1++;
			}
			else if(cols1[p1]==cols2[p2])
			{
				value = op.fn.execute(values1[p1], values2[p2]);
				column = cols1[p1];
				p1++;
				p2++;
			}
			else
			{
				value = op.fn.execute(0, values2[p2]);
				column = cols2[p2];
				p2++;
			}
			result.appendValue(resultRow, column, value);	
		}
		
		//add left over
		appendLeftForSparseBinary(op, values1, cols1, size1, p1, resultRow, result);
		appendRightForSparseBinary(op, values2, cols2, size2, p2, resultRow, result);
	}
	
	/**
	 * 
	 * @param op
	 * @param values1
	 * @param cols1
	 * @param size1
	 * @param pos
	 * @param resultRow
	 * @param result
	 * @throws DMLRuntimeException
	 */
	private static void appendLeftForSparseBinary(BinaryOperator op, double[] values1, int[] cols1, int size1, 
				int pos, int resultRow, MatrixBlock result) 
		throws DMLRuntimeException
	{
		for(int j=pos; j<size1; j++)
		{
			double v = op.fn.execute(values1[j], 0);
			result.appendValue(resultRow, cols1[j], v);
		}
	}
	
	/**
	 * 
	 * @param op
	 * @param values2
	 * @param cols2
	 * @param size2
	 * @param pos
	 * @param resultRow
	 * @param result
	 * @throws DMLRuntimeException
	 */
	private static void appendRightForSparseBinary(BinaryOperator op, double[] values2, int[] cols2, int size2, 
		int pos, int resultRow, MatrixBlock result) throws DMLRuntimeException
	{
		for( int j=pos; j<size2; j++ )
		{
			double v = op.fn.execute(0, values2[j]);
			result.appendValue(resultRow, cols2[j], v);
		}
	}
	
}

