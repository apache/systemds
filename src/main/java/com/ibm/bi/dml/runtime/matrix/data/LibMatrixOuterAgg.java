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
import com.ibm.bi.dml.runtime.functionobjects.Equals;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThan;
import com.ibm.bi.dml.runtime.functionobjects.GreaterThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.functionobjects.LessThan;
import com.ibm.bi.dml.runtime.functionobjects.LessThanEquals;
import com.ibm.bi.dml.runtime.functionobjects.NotEquals;
import com.ibm.bi.dml.runtime.functionobjects.ReduceAll;
import com.ibm.bi.dml.runtime.functionobjects.ReduceCol;
import com.ibm.bi.dml.runtime.functionobjects.ReduceRow;
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

	private LibMatrixOuterAgg() {
		//prevent instantiation via private constructor
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isSupportedUaggOp( AggregateUnaryOperator uaggOp, BinaryOperator bOp )
	{
		boolean bSupported = false;
		
		if( (bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals || bOp.fn instanceof GreaterThanEquals
				|| bOp.fn instanceof Equals || bOp.fn instanceof NotEquals)				
				&& uaggOp.aggOp.increOp.fn instanceof KahanPlus
				&& (uaggOp.indexFn instanceof ReduceCol || uaggOp.indexFn instanceof ReduceRow || uaggOp.indexFn instanceof ReduceAll)) //special case: rowsSums(outer(A,B,<))
			bSupported = true;
		
		return bSupported;
			
	}
	
	/**
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param outVal
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	public static void aggregateMatrix(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, double[] bv, 
			BinaryOperator bOp, AggregateUnaryOperator uaggOp) 
			throws DMLRuntimeException
	{		
		//step1: prepare output (last col corr)
		if(uaggOp.indexFn instanceof ReduceCol) {
			outIx.setIndexes(in1Ix.getRowIndex(), 1); 
			outVal.reset(in1Val.getNumRows(), 2, false);
		} else if(uaggOp.indexFn instanceof ReduceRow) {
			outIx.setIndexes(1, in1Ix.getColumnIndex()); 
			outVal.reset(2, in1Val.getNumColumns(), false);
		} else if(uaggOp.indexFn instanceof ReduceAll) {
			outIx.setIndexes(1,1); 
			outVal.reset(1, 2, false);
		}
		
		//step2: compute unary aggregate outer chain
		if(uaggOp.indexFn instanceof ReduceCol) {
			if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals) {
				uaRowSumLtGe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals) {
				uaRowSumGtLe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
				uaRowSumEqNe(in1Val, outVal, bv, bOp);
			}
		} else if(uaggOp.indexFn instanceof ReduceRow) {
			if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals) {
				uaColSumLtGe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals) {
				uaColSumGtLe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
				uaColSumEqNe(in1Val, outVal, bv, bOp);
			}
		} else if(uaggOp.indexFn instanceof ReduceAll) {
			if(bOp.fn instanceof LessThan || bOp.fn instanceof GreaterThanEquals) {
				uaSumLtGe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof GreaterThan || bOp.fn instanceof LessThanEquals) {
				uaSumGtLe(in1Val, outVal, bv, bOp);
			} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
				uaSumEqNe(in1Val, outVal, bv, bOp);
			}
		}
	}
	
	/**
	 * UAgg rowSums for LessThan and GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRowSumLtGe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumRowSumLtGeColSumGtLe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumRowSumLtGeColSumGtLe(ai, bv, bOp);
			out.quickSetValue(i, 0, cnt);
		}
	}
	
	/**
	 * UAgg rowSums for GreaterThan and LessThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRowSumGtLe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumRowSumGtLeColSumLtGe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumRowSumGtLeColSumLtGe(ai, bv, bOp);
			out.quickSetValue(i, 0, cnt);
		}
	}
	
	
	/**
	 * UAgg rowSums for Equal and NotEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRowSumEqNe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumEqNe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumEqNe(ai, bv, bOp);
			out.quickSetValue(i, 0, cnt);
		}
	}

	/**
	 * UAgg colSums for LessThan and GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaColSumLtGe(MatrixBlock in1Val, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		if (in1Val.isInSparseFormat())
			s_uaColSumLtGe(in1Val, outVal, bv, bOp);
		else
			d_uaColSumLtGe(in1Val, outVal, bv, bOp);
	}

	/**
	 * UAgg colSums for GreaterThan and LessThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaColSumGtLe(MatrixBlock in1Val, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		if (in1Val.isInSparseFormat())
			s_uaColSumGtLe(in1Val, outVal, bv, bOp);
		else
			d_uaColSumGtLe(in1Val, outVal, bv, bOp);
	}

	
	/**
	 * UAgg colSums for Equal and NotEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaColSumEqNe(MatrixBlock in1Val, MatrixBlock outVal, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		if (in1Val.isInSparseFormat())
			s_uaColSumEqNe(in1Val, outVal, bv, bOp);
		else
			d_uaColSumEqNe(in1Val, outVal, bv, bOp);
	}

	
	/**
	 * UAgg sums for LessThan and GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaSumLtGe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumRowSumLtGeColSumGtLe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumRowSumLtGeColSumGtLe(ai, bv, bOp);
			cnt += (int)out.quickGetValue(0, 0);
			out.quickSetValue(0, 0, cnt);
		}
	}
	
	/**
	 * UAgg sums for GreaterThan and LessThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaSumGtLe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumRowSumGtLeColSumLtGe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumRowSumGtLeColSumLtGe(ai, bv, bOp);
			cnt += (int)out.quickGetValue(0, 0);
			out.quickSetValue(0, 0, cnt);
		}
	}
	
	
	/**
	 * UAgg sums for Equal and NotEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaSumEqNe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumEqNe(0.0, bv, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int cnt = (ai == 0) ? agg0: sumEqNe(ai, bv, bOp);
			cnt += (int)out.quickGetValue(0, 0);
			out.quickSetValue(0, 0, cnt);
		}
	}

	
	/**
	 * UAgg colSums Dense Matrix for LessThan and GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void d_uaColSumLtGe(MatrixBlock in, MatrixBlock out, double[] bv, 
			BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int agg0 = sumRowSumGtLeColSumLtGe(0.0, bv, bOp);
		int m = in.clen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(0, i);
			int cnt = (ai == 0) ? agg0: sumRowSumGtLeColSumLtGe(ai, bv, bOp);
			out.quickSetValue(0, i, cnt);
		}
	}

	/**
	 * UAgg colSums Sparse Matrix for LessThan and GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void s_uaColSumLtGe(MatrixBlock in, MatrixBlock out, double[] bv,	BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int agg0 = sumRowSumGtLeColSumLtGe(0.0, bv, bOp);

		//allocate and initialize output values (not indices) 
		out.allocateDenseBlock(true);
		Arrays.fill(out.getDenseArray(), 0, out.getNumColumns(), agg0);
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseRow[] aSparseRows = in.getSparseRows();		
		for (int j = 0; j < aSparseRows.length; ++j)
		if( aSparseRows[j]!=null && !aSparseRows[j].isEmpty() )
		{
			double [] aValues = aSparseRows[j].getValueContainer();
			int [] aIndexes = aSparseRows[j].getIndexContainer();
			
			for (int i=0; i < aValues.length; ++i)
			{
				int cnt = sumRowSumGtLeColSumLtGe(aValues[i], bv, bOp);
				out.quickSetValue(0, aIndexes[i], cnt);
			}
		}		
	}

	/**
	 * UAgg colSums Dense Matrix for GreaterThan and LessThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void d_uaColSumGtLe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int agg0 = sumRowSumLtGeColSumGtLe(0.0, bv, bOp);
		int m = in.clen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(0, i);
			int cnt = (ai == 0) ? agg0: sumRowSumLtGeColSumGtLe(ai, bv, bOp);
			out.quickSetValue(0, i, cnt);
		}
	}

	/**
	 * UAgg colSums Sparse Matrix for GreaterThan and LessThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void s_uaColSumGtLe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int agg0 = sumRowSumLtGeColSumGtLe(0.0, bv, bOp);

		//allocate and initialize output values (not indices) 
		out.allocateDenseBlock(true);
		Arrays.fill(out.getDenseArray(), 0, out.getNumColumns(), agg0);
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseRow[] aSparseRows = in.getSparseRows();		
		for (int j = 0; j < aSparseRows.length; ++j)
		if( aSparseRows[j]!=null && !aSparseRows[j].isEmpty() )
		{
			double [] aValues = aSparseRows[j].getValueContainer();
			int [] aIndexes = aSparseRows[j].getIndexContainer();
			
			for (int i=0; i < aValues.length; ++i)
			{
				int cnt = sumRowSumLtGeColSumGtLe(aValues[i], bv, bOp);
				out.quickSetValue(0, aIndexes[i], cnt);
			}
		}		
	}

	/**
	 * UAgg colSums Dense Matrix for Equal and NotEqual operator 
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void d_uaColSumEqNe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumEqNe(0.0, bv, bOp);
		int m = in.clen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(0, i);
			int cnt = (ai == 0) ? agg0: sumEqNe(ai, bv, bOp);
			out.quickSetValue(0, i, cnt);
		}
	}

	/**
	 * UAgg colSums Sparse Matrix for Equal and NotEqual operator 
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void s_uaColSumEqNe(MatrixBlock in, MatrixBlock out, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int agg0 = sumEqNe(0.0, bv, bOp);

		//allocate and initialize output values (not indices) 
		out.allocateDenseBlock(true);
		Arrays.fill(out.getDenseArray(), 0, out.getNumColumns(), agg0);
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseRow[] aSparseRows = in.getSparseRows();		
		for (int j = 0; j < aSparseRows.length; ++j)
		if( aSparseRows[j]!=null && !aSparseRows[j].isEmpty() )
		{
			double [] aValues = aSparseRows[j].getValueContainer();
			int [] aIndexes = aSparseRows[j].getIndexContainer();
			
			for (int i=0; i < aValues.length; ++i)
			{
				int cnt = sumEqNe(aValues[i], bv, bOp);
				out.quickSetValue(0, aIndexes[i], cnt);
			}
		}		
	}

	
	/**
	 * Calculates the sum of number for rowSum of GreaterThan and LessThanEqual, and 
	 * 									colSum of LessThan and GreaterThanEqual operators.
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int sumRowSumGtLeColSumLtGe(double value, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ix = Arrays.binarySearch(bv, value);
		int cnt = 0;
		
		if( ix >= 0 ){ //match, scan to next val
			while( ix > 0 && value==bv[ix-1]) --ix;
			ix++;	//Readjust index to match subsenquent index calculation.
		}

		cnt = bv.length-Math.abs(ix)+1;

		//cnt = Math.abs(ix) - 1;
		if ((bOp.fn instanceof LessThan) || (bOp.fn instanceof GreaterThan))
			cnt = bv.length - cnt;

		return cnt;
	}

	
	/**
	 * Calculates the sum of number for rowSum of LessThan and GreaterThanEqual, and 
	 * 									colSum of GreaterThan and LessThanEqual operators.
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int sumRowSumLtGeColSumGtLe(double value, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ix = Arrays.binarySearch(bv, value);
		int cnt = 0;
		
		if( ix >= 0 ){ //match, scan to next val
			while( value==bv[ix++] && ix<bv.length );
			ix += (value==bv[bv.length-1])?1:0;
		}
			
		cnt = bv.length-Math.abs(ix)+1;
		
		if (bOp.fn instanceof LessThanEquals || bOp.fn instanceof GreaterThanEquals)
			cnt = bv.length - cnt;		

		return cnt;
	}

	
	/**
	 * Calculates the sum of number for Equal and NotEqual operators 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int sumEqNe(double value, double[] bv, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ix = Arrays.binarySearch(bv, value);
		int cnt = 0;
		
		if( ix >= 0 ){ //match, scan to next val
			while( ix > 0 && value==bv[ix-1]) --ix;
			while( ix<bv.length && value==bv[ix]) {
				ix++; cnt++;
			}
		}
			
		if (bOp.fn instanceof NotEquals)
			cnt = bv.length - cnt;

		return cnt;
	}
}