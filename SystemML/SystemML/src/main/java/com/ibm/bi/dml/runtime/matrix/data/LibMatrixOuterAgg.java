/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

/**
 * ACS:
 * Purpose of this library is to make some of the unary outer aggregate operator more efficient.
 * Today these operators are being handled through common operations.
 * This library will expand per need and priority to include these operators through this support.
 * To begin with, first operator being handled is unary aggregate for less than (<), rowsum operation.
 * Other list will be added soon are rowsum on >, <=, >=, ==, and != operation.  
 */
public class LibMatrixOuterAgg 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";



	private LibMatrixOuterAgg() {
		//prevent instantiation via private constructor
	}
	
	
	
	
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isSupportedUnaryAggregateOperator( AggregateUnaryOperator uaggOp, BinaryOperator bOp )
	{
		boolean bSupported = false;
		
		if( (bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals || bOp.fn instanceof GreaterThanEquals
				|| bOp.fn instanceof Equals || bOp.fn instanceof NotEquals)				
				&& uaggOp.aggOp.increOp.fn instanceof KahanPlus
				&& uaggOp.indexFn instanceof ReduceCol ) //special case: rowsSums(outer(A,B,<))
			bSupported = true;
		
		return bSupported;
			
	}
	
	/**
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	public static void aggregateMatrix(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		//step1: prepare output (last col corr)
		outIx.setIndexes(in1Ix.getRowIndex(), 1); 
		outVal.reset(in1Val.getNumRows(), 2, false);
		
		//step2: compute unary aggregate outer chain
		if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals) {
			aggregateMatrixForLessAndGreaterThanEquals(in1Ix, in1Val, outIx, outVal, bv, bOp);
		} else if(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals) {
			aggregateMatrixForGreaterAndLessThanEquals(in1Ix, in1Val, outIx, outVal, bv, bOp);
		} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
			aggregateMatrixForEqualAndNotEquals(in1Ix, in1Val, outIx, outVal, bv, bOp);
		}
	}
	
	/**
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void aggregateMatrixForLessAndGreaterThanEquals(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		MatrixBlock a = in1Val;
		MatrixBlock c = outVal;

		if(!(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals))
			return;
		
		for( int i=0; i<a.getNumRows(); i++ ) {
			double ai = a.quickGetValue(i, 0);
			int ix = Arrays.binarySearch(bv, ai);
			int cnt = 0;
			
			if( ix >= 0 ){ //match, scan to next val
				while( ai==bv[ix++] && ix<bv.length );
				ix += (ai==bv[bv.length-1])?1:0;
			}
				
			if (!(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals))
				cnt = bv.length-Math.abs(ix)+1;
			
			if (bOp.fn instanceof GreaterThanEquals || bOp.fn instanceof GreaterThan || bOp.fn instanceof NotEquals)
				cnt = bv.length - cnt;
			c.quickSetValue(i, 0, cnt);
		}
	}
	
	/**
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void aggregateMatrixForGreaterAndLessThanEquals(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		MatrixBlock a = in1Val;
		MatrixBlock c = outVal;
		
		if(!(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals))
			return;
					
		for( int i=0; i<a.getNumRows(); i++ ) {
			double ai = a.quickGetValue(i, 0);
			int ix = Arrays.binarySearch(bv, ai);
			int cnt = 0;
			
			if( ix >= 0 ){ //match, scan to next val
				while( ix > 0 && ai==bv[ix-1]) --ix;
				ix++;	//Readjust index to match subsenquent index calculation.
			}
				
			cnt = bv.length-Math.abs(ix)+1;
			
			if (bOp.fn instanceof GreaterThan)
				cnt = bv.length - cnt;
			c.quickSetValue(i, 0, cnt);
		}
	}
	

	/**
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void aggregateMatrixForEqualAndNotEquals(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		MatrixBlock a = in1Val;
		MatrixBlock c = outVal;

		if(!(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals))
			return;
			
		for( int i=0; i<a.getNumRows(); i++ ) {
			double ai = a.quickGetValue(i, 0);
			int ix = Arrays.binarySearch(bv, ai);
			int cnt = 0;
			
			if( ix >= 0 ){ //match, scan to next val
				while( ix > 0 && ai==bv[ix-1]) --ix;
				while( ix<bv.length && ai==bv[ix]) {ix++; cnt++;}
			}
				
			if (bOp.fn instanceof NotEquals)
				cnt = bv.length - cnt;
			c.quickSetValue(i, 0, cnt);
		}
	}
	
}