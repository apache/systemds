/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.Minus;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Or;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public enum BinaryAccessType {
		MATRIX_MATRIX,
		MATRIX_COL_VECTOR,
		MATRIX_ROW_VECTOR,
		INVALID,
	}
	
	///////////////////////////////////
	// public matrix bincell interface
	///////////////////////////////////
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @param ret
	 * @throws DMLRuntimeException
	 */
	public static void bincellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, BinaryOperator op) 
		throws DMLRuntimeException
	{
		if(op.sparseSafe)
			safeBinary(m1, m2, ret, op);
		else
			unsafeBinary(m1, m2, ret, op);
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
		if(op.sparseSafe)
			safeBinaryInPlace(m1ret, m2, op);
		else
			unsafeBinaryInPlace(m1ret, m2, op);
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
		else if( rlen1 > 1 && rlen2 == 1 )
			return BinaryAccessType.MATRIX_ROW_VECTOR;
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
		
		//currently we support MM (where both dimensions need to match) and
		//MV operations w/ V always being a column vector (where row dimensions 
		//need to match, and the second input has exactly 1 column)
		return (   ( rlen1 == rlen2 || (rlen1 > 1 && rlen2 == 1) ) 
				&& ( clen1 == clen2 || (clen1 > 1 && clen2 == 1) ) );
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
		//skip empty blocks (since sparse-safe)
		if( m1.isEmptyBlock(false) && m2.isEmptyBlock(false) )
			return;
	
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
		else //MATRIX - MATRIX
		{
			if(m1.sparse && m2.sparse)
			{
				if(ret.sparse)
					ret.adjustSparseRows(ret.rlen-1);	
				
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
		boolean skipEmpty = (op.fn instanceof Multiply);
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
				if( !skipEmpty || v2 != 0 ) //skip empty rows
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
	 * TODO: custom implementation for sparse-sparse (row vector) possible
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
		boolean skipEmpty = (op.fn instanceof Multiply);
		
		int rlen = m1.rlen;
		int clen = m1.clen;
		SparseRow[] a = m1.sparseRows;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		//early abort on skip and empty
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		//allocate once in order to prevent repeated reallocation
		if( ret.sparse )
			ret.adjustSparseRows(ret.rlen-1);
		
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
		boolean skipEmpty = (op.fn instanceof Multiply);
		int rlen = m1.rlen;
		int clen = m1.clen;
		BinaryAccessType atype = getBinaryAccessType(m1, m2);
		
		//early abort on skip and empy
		if( skipEmpty && (m1.isEmptyBlock(false) || m2.isEmptyBlock(false) ) )
			return; // skip entire empty block
		
		//allocate once in order to prevent repeated reallocation 
		if( ret.sparse )
			ret.adjustSparseRows(ret.rlen-1);
		
		if( atype == BinaryAccessType.MATRIX_COL_VECTOR )
		{
			for( int i=0; i<rlen; i++ )
			{
				//replicate vector value
				double v2 = m2.quickGetValue(i, 0);
				if( !skipEmpty || v2 != 0 ) {//skip zero rows
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
			for( int i=0; i<rlen; i++ )
				for( int j=0; j<clen; j++ )
				{
					double v1 = m1.quickGetValue(i, j);
					double v2 = m2.quickGetValue(0, j); //replicated vector value
					double v = op.fn.execute( v1, v2 );
					ret.appendValue(i, j, v);		
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
	 * @param m1ret
	 * @param m2
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void safeBinaryInPlace(MatrixBlock m1ret, MatrixBlock m2, BinaryOperator op) throws DMLRuntimeException 
	{
		int rlen = m1ret.rlen;
		int clen = m1ret.clen;
		
		if(m1ret.sparse && m2.sparse)
		{
			//special case, if both matrices are all 0s, just return
			if(m1ret.sparseRows==null && m2.sparseRows==null)
				return;
			
			if(m1ret.sparseRows!=null)
				m1ret.adjustSparseRows(rlen-1);
			if(m2.sparseRows!=null)
				m2.adjustSparseRows(rlen-1);
			
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

