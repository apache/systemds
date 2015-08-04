/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

/**
 * ACS:
 * Library for matrix aggregations including TODO ak+, uak+ for all
 * combinations of dense and sparse representations, and corrections.
 * Those are performance-critical operations because they are used
 * on combiners/reducers of important operations like tsmm, mvmult,
 * indexing, but also basic sum/min/max/mean, row*, col*, etc. Specific
 * handling is especially required for all non sparse-safe operations
 * in order to prevent unnecessary worse asymptotic behavior.
 *
 * This library currently covers the following opcodes:
 * TODO ak+, uak+, uark+, uack+, uamin, uarmin, uacmin, uamax, uarmax, uacmax,
 * TODO ua*, uamean, uarmean, uacmean, uarimax, uaktrace.
 * 
 * TODO next opcode extensions: a+, colindexmax
 * TODO low level optimization (potential 3x, sum non-conclusive yet)
 */
public class LibMatrixUaggOuter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";



	private LibMatrixUaggOuter() {
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
		
		if( bOp.fn instanceof LessThan && uaggOp.aggOp.increOp.fn instanceof KahanPlus
				&& uaggOp.indexFn instanceof ReduceCol ) //special case: rowsSums(outer(A,B,<))
			bSupported = true;
		
		return bSupported;
			
	}
	
	/**
	 * 
	 * @param in
	 * @param aggVal
	 * @param aggCorr
	 * @throws DMLRuntimeException
	 */
	public static void aggregateMatrix(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv) 
			throws DMLRuntimeException
	{
		
		//step1: prepare output (last col corr)
		outIx.setIndexes(in1Ix.getRowIndex(), 1); 
		outVal.reset(in1Val.getNumRows(), 2, false);
		
		//step2: compute unary aggregate outer chain
		MatrixBlock a = (MatrixBlock)in1Val;
		MatrixBlock c = (MatrixBlock)outVal;
		for( int i=0; i<a.getNumRows(); i++ ) {
			double ai = a.quickGetValue(i, 0);
			int ix = Arrays.binarySearch(bv, ai);
			if( ix >= 0 ){ //match, scan to next val
				while( ai==bv[ix++] && ix<bv.length );
				ix += (ai==bv[bv.length-1])?1:0;
			}
			int cnt = bv.length-Math.abs(ix)+1;
			c.quickSetValue(i, 0, cnt);
		}

	}
	

}

