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
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.LessThanEquals;
import org.apache.sysml.runtime.functionobjects.NotEquals;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.SortUtils;

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
	 * This will return if uaggOp is of type RowIndexMax
	 * 
	 * @param uaggOp
	 * @return
	 */
	public static boolean isRowIndexMax(AggregateUnaryOperator uaggOp)
	{
		return 	(uaggOp.aggOp.increOp.fn instanceof Builtin														
			    && (((Builtin)(uaggOp.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX));						
	}
	
	/**
	 * This will return if uaggOp is of type RowIndexMin
	 * 
	 * @param uaggOp
	 * @return
	 */
	public static boolean isRowIndexMin(AggregateUnaryOperator uaggOp)
	{
		return 	(uaggOp.aggOp.increOp.fn instanceof Builtin									
			    && (((Builtin)(uaggOp.aggOp.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX));						
	}
	
	
	/**
	 * This will return if uaggOp is of type RowIndexMin
	 * 
	 * @param bOp
	 * @return true/false, based on if its one of the six operators (<, <=, >, >=, == and !=)
	 */
	public static boolean isCompareOperator(BinaryOperator bOp)
	{
		return ( bOp.fn instanceof LessThan || bOp.fn instanceof LessThanEquals		// For operators <, <=,  
			|| bOp.fn instanceof GreaterThan || bOp.fn instanceof GreaterThanEquals //				 >, >=
			|| bOp.fn instanceof Equals || bOp.fn instanceof NotEquals);				//				==, !=
	}
		
		
			/**
	 * @param uaggOp
	 * @param bOp
	 * @return
	 */
	public static boolean isSupportedUaggOp( AggregateUnaryOperator uaggOp, BinaryOperator bOp )
	{
		boolean bSupported = false;
		
		if(isCompareOperator(bOp)
			&& 
				(uaggOp.aggOp.increOp.fn instanceof KahanPlus					// Kahanplus
			    ||		
				(isRowIndexMin(uaggOp) || isRowIndexMax(uaggOp)))				// RowIndexMin or RowIndexMax 						
			&&			
				(uaggOp.indexFn instanceof ReduceCol							// ReduceCol 
					|| uaggOp.indexFn instanceof ReduceRow 						// ReduceRow
					|| uaggOp.indexFn instanceof ReduceAll))					// ReduceAll
			
			bSupported = true;
		
		return bSupported;
			
	}

	/*
	 * 
	 * @param iCols
	 * @param vmb
	 * @param bOp
	 * @param uaggOp
	 * 
	 */
	public static int[] prepareRowIndices(int iCols, double vmb[], BinaryOperator bOp, AggregateUnaryOperator uaggOp) throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		return (isRowIndexMax(uaggOp)?prepareRowIndicesMax(iCols, vmb, bOp):prepareRowIndicesMin(iCols, vmb, bOp));
	}
	
	/**
	 * This function will return max indices, based on column vector data. 
	 * This indices will be computed based on operator. 
	 * These indices can be used to compute max index for a given input value in subsequent operation.
	 * 
	 *  e.g. Right Vector has data (V1)    :                6   3   9   7   2   4   4   3
	 *       Original indices for this data will be (I1):   1   2   3   4   5   6   7   8		
	 * 
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 * CumMax of I2 will be A:  (CumMin(I2))                5   5   8   8   8   8   8   8
	 * CumMax of I2 in reverse order be B:                  8   8   8   7   7   4   4   3
	 * 
	 * Values from vector A is used to compute RowIndexMax for > & >= operators
	 * Values from vector B is used to compute RowIndexMax for < & <= operators
	 * Values from I2 is used to compute RowIndexMax for == operator.
	 * Original values are directly used to compute RowIndexMax for != operator
	 * 
	 * Shifting values from vector A or B is required to compute final indices.
	 * Once indices are shifted from vector A or B, their cell value corresponding to input data will be used. 
	 *  
	 * 
	 * @param iCols
	 * @param vmb
	 * @param bOp
	 * @return vixCumSum
	 */
	public static int[] prepareRowIndicesMax(int iCols, double vmb[], BinaryOperator bOp) throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		int[] vixCumSum = null;
		int[] vix = new int[iCols];
		
		//sort index vector on extracted data (unstable)
		if(!(bOp.fn instanceof NotEquals)){
			for( int i=0; i<iCols; i++ )
				vix[i] = i;

			SortUtils.sortByValueStable(0, iCols, vmb, vix);
		} 
	
		if(bOp.fn instanceof LessThan || bOp.fn instanceof LessThanEquals 
				|| bOp.fn instanceof GreaterThan || bOp.fn instanceof GreaterThanEquals) {
	
			boolean bPrimeCumSum = false;
			if(bOp.fn instanceof LessThan || bOp.fn instanceof LessThanEquals )
				bPrimeCumSum = true;
			
			double dvix[] = new double[vix.length];
			if (bPrimeCumSum)
				for (int i = 0; i< vix.length; ++i)
					dvix[vix.length-1-i] = vix[i];
			else
				for (int i = 0; i< vix.length; ++i)
					dvix[i] = vix[i];
			
			MatrixBlock mbix = DataConverter.convertToMatrixBlock(dvix, true);
			
			UnaryOperator u_op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinFunctionCode.CUMMAX));
			MatrixBlock mbResult = (MatrixBlock) mbix.unaryOperations(u_op, new MatrixBlock());
			
			vixCumSum = DataConverter.convertToIntVector(mbResult);  
			if (bPrimeCumSum)
				for (int i = 0; i< (vixCumSum.length+1)/2; ++i) {
					int iTemp = vixCumSum[vixCumSum.length-1-i];
					vixCumSum[vixCumSum.length-1-i] = vixCumSum[i];
					vixCumSum[i] = iTemp;
				}
							
			adjustRowIndicesMax(vixCumSum, vmb, bOp);
			
		} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
			adjustRowIndicesMax(vix, vmb, bOp);
			vixCumSum = vix;
		}
		
		return vixCumSum;
	}
	

	/**
	 * This function will return min indices, based on column vector data. 
	 * This indices will be computed based on operator. 
	 * These indices can be used to compute min index for a given input value in subsequent operation.
	 * 
	 *  e.g. Right Vector has data (V1)    :                6   3   9   7   2   4   4   3
	 *       Original indices for this data will be (I1):   1   2   3   4   5   6   7   8		
	 * 
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 * CumMin of I2 will be A:  (CumMin(I2))                5   2   2   2   2   1   1   1
	 * CumMin of I2 in reverse order be B:                  1   1   1   1   1   1   3   3
	 * 
	 * Values from vector A is used to compute RowIndexMin for > operator
	 * Values from vector B is used to compute RowIndexMin for <, <= and >= operators
	 * Values from I2 is used to compute RowIndexMax for == operator.
	 * Original values are directly used to compute RowIndexMax for != operator
	 * 
	 * Shifting values from vector A or B is required to compute final indices.
	 * Once indices are shifted from vector A or B, their cell value corresponding to input data will be used. 
	 *  
	 * 
	 * @param iCols
	 * @param vmb
	 * @param bOp
	 * @return vixCumSum
	 */
	public static int[] prepareRowIndicesMin(int iCols, double vmb[], BinaryOperator bOp) throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		int[] vixCumSum = null;
		int[] vix = new int[iCols];
		
		//sort index vector on extracted data (unstable)
		if(!(bOp.fn instanceof NotEquals || bOp.fn instanceof Equals )){
			for( int i=0; i<iCols; i++ )
				vix[i] = i;
	
			SortUtils.sortByValueStable(0, iCols, vmb, vix);
		} 
	
		if(bOp.fn instanceof LessThan || bOp.fn instanceof LessThanEquals 
				|| bOp.fn instanceof GreaterThan || bOp.fn instanceof GreaterThanEquals) {
	
			boolean bPrimeCumSum = false;
			if(bOp.fn instanceof GreaterThan || bOp.fn instanceof GreaterThanEquals )
				bPrimeCumSum = true;
			
			double dvix[] = new double[vix.length];
			if (bPrimeCumSum)
				for (int i = 0; i< vix.length; ++i)
					dvix[vix.length-1-i] = vix[i];
			else
				for (int i = 0; i< vix.length; ++i)
					dvix[i] = vix[i];
			
			MatrixBlock mbix = DataConverter.convertToMatrixBlock(dvix, true);
			
			UnaryOperator u_op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinFunctionCode.CUMMIN));
			MatrixBlock mbResult = (MatrixBlock) mbix.unaryOperations(u_op, new MatrixBlock());
			
			vixCumSum = DataConverter.convertToIntVector(mbResult);  
			if (bPrimeCumSum)
				for (int i = 0; i< (vixCumSum.length+1)/2; ++i) {
					int iTemp = vixCumSum[vixCumSum.length-1-i];
					vixCumSum[vixCumSum.length-1-i] = vixCumSum[i];
					vixCumSum[i] = iTemp;
				}
							
			adjustRowIndicesMin(vixCumSum, vmb, bOp);
			
		} else if(bOp.fn instanceof Equals || bOp.fn instanceof NotEquals) {
			adjustRowIndicesMin(vix, vmb, bOp);
			vixCumSum = vix;
		}
		
		return vixCumSum;
	}
	
	
	/**
	 * ReSet output matrix
	 * 
	 * @param in1Ix
	 * @param in1Val
	 * @param outIx
	 * @param outVal
	 * @param uaggOp
	 */
	public static void resetOutputMatix(MatrixIndexes in1Ix, MatrixBlock in1Val, MatrixIndexes outIx, MatrixBlock outVal, AggregateUnaryOperator uaggOp) 
	{		
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
	}
	
	
	/**
	 * 
	 * @param in1Val
	 * @param outVal
	 * @param bv
	 * @param bOp
	 * @param uaggOp
	 * @throws DMLRuntimeException
	 */
	public static void aggregateMatrix(MatrixBlock in1Val, MatrixBlock outVal, double[] bv, int[] bvi, BinaryOperator bOp, AggregateUnaryOperator uaggOp) 
			throws DMLRuntimeException
	{		
		// compute unary aggregate outer chain
		if(isRowIndexMax(uaggOp)) 
		{
			if(bOp.fn instanceof LessThan) {
				uaRIMLt(in1Val, outVal, bv, bvi, bOp);
			} else if(bOp.fn instanceof LessThanEquals) {
				uaRIMLe(in1Val, outVal, bv, bvi, bOp);
			} else if(bOp.fn instanceof GreaterThan) { 
				uaRIMGt(in1Val, outVal, bv, bvi, bOp);
			} else if(bOp.fn instanceof GreaterThanEquals) {
				uaRIMGe(in1Val, outVal, bv, bvi, bOp);
			} else if(bOp.fn instanceof Equals){ 
				uaRIMEq(in1Val, outVal, bv, bvi, bOp);	
			} else if (bOp.fn instanceof NotEquals) {
				uaRIMNe(in1Val, outVal, bv, bvi, bOp);
			}
		} else if(isRowIndexMin(uaggOp)) 
		{
				if(bOp.fn instanceof LessThan) {
					uaRIMinLt(in1Val, outVal, bv, bvi, bOp);
				} else if(bOp.fn instanceof LessThanEquals) {
					uaRIMinLe(in1Val, outVal, bv, bvi, bOp);
				} else if(bOp.fn instanceof GreaterThan) { 
					uaRIMinGt(in1Val, outVal, bv, bvi, bOp);
				} else if(bOp.fn instanceof GreaterThanEquals) {
					uaRIMinGe(in1Val, outVal, bv, bvi, bOp);
				} else if(bOp.fn instanceof Equals){ 
					uaRIMinEq(in1Val, outVal, bv, bvi, bOp);	
				} else if (bOp.fn instanceof NotEquals) {
					uaRIMinNe(in1Val, outVal, bv, bvi, bOp);
				}
		} else if(uaggOp.indexFn instanceof ReduceCol) {
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
	 * UAgg rowIndexMax for LessThan operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMLt(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxLt(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxLt(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	/**
	 * UAgg rowIndexMax for LessThanEquals operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMLe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxLe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxLe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	/**
	 * UAgg rowIndexMax for GreaterThan operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMGt(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxGt(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxGt(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	
	/**
	 * UAgg rowIndexMax for GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMGe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxGe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxGe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	
	/**
	 * UAgg rowIndexMax for Equal operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMEq(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxEq(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxEq(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}


	
	/**
	 * UAgg rowIndexMax for NotEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMNe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uarimaxNe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uarimaxNe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}


	/**
	 * UAgg rowIndexMin for LessThan operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinLt(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminLt(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminLt(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	/**
	 * UAgg rowIndexMin for LessThanEquals operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinLe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminLe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminLe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	/**
	 * UAgg rowIndexMin for GreaterThan operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinGt(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminGt(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminGt(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	
	/**
	 * UAgg rowIndexMin for GreaterThanEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinGe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminGe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminGe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}
	
	
	/**
	 * UAgg rowIndexMin for Equal operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinEq(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminEq(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminEq(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
		}
	}


	
	/**
	 * UAgg rowIndexMin for NotEqual operator
	 * 
	 * @param in
	 * @param out
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static void uaRIMinNe(MatrixBlock in, MatrixBlock out, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{		
		int ind0 = uariminNe(0.0, bv, bvi, bOp);
		int m = in.rlen;
		
		for( int i=0; i<m; i++ ) {
			double ai = in.quickGetValue(i, 0);
			int ind = (ai == 0) ? ind0: uariminNe(ai, bv, bvi, bOp);
			out.quickSetValue(i, 0, ind);
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
		Arrays.fill(out.getDenseBlock(), 0, out.getNumColumns(), agg0);
		if(agg0 != 0.0)
			out.setNonZeros(out.getNumColumns());
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseBlock sblock = in.getSparseBlock();		
		for( int j = 0; j < sblock.numRows(); j++)
		if( !sblock.isEmpty(j) ) {
			int apos = sblock.pos(j);
			int alen = sblock.size(j);
			int[] aix = sblock.indexes(j);
			double [] avals = sblock.values(j);
			
			for (int i=apos; i < apos+alen; i++) {
				int cnt = sumRowSumGtLeColSumLtGe(avals[i], bv, bOp);
				out.quickSetValue(0, aix[i], cnt);
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
		Arrays.fill(out.getDenseBlock(), 0, out.getNumColumns(), agg0);
		if(agg0 != 0.0)
			out.setNonZeros(out.getNumColumns());
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseBlock sblock = in.getSparseBlock();		
		for (int j = 0; j < sblock.numRows(); j++)
		if( !sblock.isEmpty(j) ) {
			int apos = sblock.pos(j);
			int alen = sblock.size(j);
			int[] aix = sblock.indexes(j);
			double [] avals = sblock.values(j);
			
			for (int i=apos; i < apos+alen; i++) {
				int cnt = sumRowSumLtGeColSumGtLe(avals[i], bv, bOp);
				out.quickSetValue(0, aix[i], cnt);
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
		Arrays.fill(out.getDenseBlock(), 0, out.getNumColumns(), agg0);
		if(agg0 != 0.0)
			out.setNonZeros(out.getNumColumns());
		
		if( in.isEmptyBlock(false) )
			return;
			
		SparseBlock sblock = in.getSparseBlock();		
		for (int j = 0; j < sblock.numRows(); j++)
		if( !sblock.isEmpty(j) ) {
			int apos = sblock.pos(j);
			int alen = sblock.size(j);
			int[] aix = sblock.indexes(j);
			double [] avals = sblock.values(j);
			
			for (int i=apos; i < apos+alen; ++i) {
				int cnt = sumEqNe(avals[i], bv, bOp);
				out.quickSetValue(0, aix[i], cnt);
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
	
	/**
	 * Find out rowIndexMax for Equal operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxEq(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ix = Arrays.binarySearch(bv, value);
		int ixMax = bv.length;
		
		if( ix >= 0 ) 
			ixMax = bvi[ix]+1;
		return ixMax;
	}

	
	/**
	 * Find out rowIndexMax for NotEqual operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxNe(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMax = bv.length;
		
		if( bv[bv.length-1] == value ) 
			ixMax = bvi[0]+1;
		return ixMax;
	}

	
	/**
	 * Find out rowIndexMax for GreaterThan operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxGt(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMax = bv.length;
		
		if(value <= bv[0] || value > bv[bv.length-1]) 
			return ixMax;
		
		int ix = Arrays.binarySearch(bv, value);
		ix = Math.abs(ix)-1;
		ixMax = bvi[ix-1]+1; 
		
		return ixMax;
	}

	
	/**
	 * Find out rowIndexMax for GreaterThanEqual operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxGe(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMax = bv.length;
		
		if(value < bv[0] || value >= bv[bv.length-1]) 
			return ixMax;
		
		int ix = Arrays.binarySearch(bv, value);
		ix = Math.abs(ix)-1;
		ixMax = bvi[ix-1]+1; 
		
		return ixMax;
	}

	
	/**
	 * Find out rowIndexMax for LessThan operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxLt(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMax = bv.length;
		
		if(value < bv[0] || value >= bv[bv.length-1]) 
			return ixMax;
		
		int ix = Arrays.binarySearch(bv, value);
		if (ix < 0) 
			ix = Math.abs(ix)-1;
		ixMax = bvi[ix-1]+1; 
		
		return ixMax;
	}

	/**
	 * Find out rowIndexMax for LessThanEquals operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uarimaxLe(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMax = bv.length;
		
		if(value < bv[0] || value > bv[bv.length-1]) 
			return ixMax;
		
		int ix = Arrays.binarySearch(bv, value);
		ix = Math.abs(ix);
		ixMax = bvi[ix-1]+1; 
		
		return ixMax;
	}
	

	/**
	 * Find out rowIndexMin for Equal operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminEq(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if(value == bv[0])
			ixMin = bvi[0]+1;
		return ixMin;
	}

	
	/**
	 * Find out rowIndexMin for NotEqual operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminNe(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if( bv[0] != value ) 
			ixMin = bvi[0]+1;
		return ixMin;
	}

	
	/**
	 * Find out rowIndexMin for GreaterThan operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminGt(double value, double[] bv, int bvi[], BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if(value <= bv[0] || value > bv[bv.length-1]) 
			return ixMin;
		
		int ix = Arrays.binarySearch(bv, value);
		ix = Math.abs(ix)-1;
		ixMin = bvi[ix]+1; 
		
		return ixMin;
	}

	
	/**
	 * Find out rowIndexMin for GreaterThanEqual operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminGe(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if(value <= bv[0] || value > bv[bv.length-1]) 
			return ixMin;
		
		int ix = Arrays.binarySearch(bv, value);
		if(ix < 0)
			ix = Math.abs(ix)-1;
		ixMin = bvi[ix-1]+1; 
		
		return ixMin;
	}

	
	/**
	 * Find out rowIndexMin for LessThan operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminLt(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if(value < bv[0] || value >= bv[bv.length-1]) 
			return ixMin;
		
		int ix = Arrays.binarySearch(bv, value);
		if (ix < 0) 
			ix = Math.abs(ix)-1;
		ixMin = bvi[ix-1]+1; 
		
		return ixMin;
	}

	/**
	 * Find out rowIndexMin for LessThanEquals operator. 
	 * 
	 * @param value
	 * @param bv
	 * @param bOp
	 * @throws DMLRuntimeException
	 */
	private static int uariminLe(double value, double[] bv, int[] bvi, BinaryOperator bOp) 
			throws DMLRuntimeException
	{
		int ixMin = 1;
		
		if(value < bv[0] || value > bv[bv.length-1]) 
			return ixMin;
		
		int ix = Arrays.binarySearch(bv, value);
		if (ix < 0) 
			ix = Math.abs(ix)-1;
		ixMin = bvi[ix]+1; 
		
		return ixMin;
	}
	
	
	/**
	 * This function adjusts indices to be leveraged in uarimaxXX functions.
	 * Initially vector containing indices are sorted based on value and then CumMax/CumMin 
	 * per need for <, <=, >, >= operator, where as just sorted indices based on value for ==, and != operators.
	 * There is need to shift these indices for different operators, which is handled through this function. 
	 * 
	 * @param vix
	 * @param vmb
	 * @param bOp
	 */
	public static void adjustRowIndicesMax(int[] vix, double[] vmb,BinaryOperator bOp)
    {
    	if (bOp.fn instanceof LessThan) {
        	shiftLeft(vix, vmb);
    	} else if ((bOp.fn instanceof GreaterThanEquals) || (bOp.fn instanceof Equals)) {
    		setMaxIndexInPartition(vix,vmb);
    	} else if(bOp.fn instanceof NotEquals) {
    		double dLastValue = vmb[vmb.length-1];
    		int i=vmb.length-1;
    		while(i>0 && dLastValue == vmb[i-1]) --i;
    		if (i > 0) 
    			vix[0] = i-1;
    		else	
    			vix[0] = vix.length-1;
    	}
    }

	/**
	 * This function adjusts indices to be leveraged in uariminXX functions.
	 * Initially vector containing indices are sorted based on value and then CumMin 
	 * per need for <, <=, >, >= operator, where as just sorted indices based on value for ==, and != operators.
	 * There is need to shift these indices for different operators, which is handled through this function. 
	 * 
	 * @param vix
	 * @param vmb
	 * @param bOp
	 */
	public static void adjustRowIndicesMin(int[] vix, double[] vmb,BinaryOperator bOp)
    {
		if (bOp.fn instanceof GreaterThan) {
			setMinIndexInPartition(vix, vmb);
		}
		else if (bOp.fn instanceof GreaterThanEquals) {
        	shiftLeft(vix, vmb);
    	}
        else if (bOp.fn instanceof LessThanEquals) {
        	shiftRight(vix, vmb);
    	} else if(bOp.fn instanceof Equals) {
    		double dFirstValue = vmb[0];
    		int i=0;
    		while(i<vmb.length-1 && dFirstValue == vmb[i+1]) ++i;
    		if (i < vmb.length-1) 
    			vix[0] = i+1;
    		else	
    			vix[0] = 0;
    	} else if(bOp.fn instanceof NotEquals) {
    		double dFirstValue = vmb[0];
    		int i=0;
    		while(i<vmb.length-1 && dFirstValue == vmb[i+1]) ++i;
    		if (i < vmb.length-1) 
    			vix[0] = i-1;
    		else	
    			vix[0] = 0;
    	}
    }
	

	/**
	 * This function will shift indices from one partition to next in right direction.
	 * 
	 *  For an example, if there are two sorted vector based on value like following, where
	 *   V2 is sorted data, and I2 are its corresponding indices.
	 *   
	 *   Then this function will shift indices to right by one partition I2". 
	 *     Left most partition remained untouched.
	 *   
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 *    Shift Right by one partition (I2")                5   5   2   8   6   7   1   4
	 * 
	 * @param vix
	 * @param vmb
	 */
	
	public static void shiftRight(int[] vix, double[] vmb)
	{
    	for (int i = vix.length-1; i>0;)
    	{
    		int iPrevInd = i;
    		double dPrevVal = vmb[iPrevInd];
			while(i>=0 && dPrevVal == vmb[i]) --i;
			
			if(i >= 0) {
				for (int j = i+1; j<= iPrevInd; ++j)
					vix[j] = vix[i];
			}
    	}
	}

	
	/**
	 * This function will shift indices from one partition to next in left direction.
	 * 
	 *  For an example, if there are two sorted vector based on value like following, where
	 *   V2 is sorted data, and I2 are its corresponding indices.
	 *   
	 *   Then this function will shift indices to right by one partition I2". 
	 *     Right most partition remained untouched.
	 *   
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 *    Shift Left by one partition (I2")                 2   8   6   7   1   4   3   3
	 * 
	 * @param vix
	 * @param vmb
	 */
	
	public static void shiftLeft(int[] vix, double[] vmb)
	{
    	int iCurInd = 0;
		
    	for (int i = 0; i < vix.length;++i)
    	{
    		double dPrevVal = vmb[iCurInd];
			while(i<vix.length && dPrevVal == vmb[i]) ++i;
			
			if(i < vix.length) {
				for(int j=iCurInd; j<i; ++j) vix[j] = vix[i];
				iCurInd = i;
			}
    	}
	}

	
	/**
	 * This function will set minimum index in the partition to all cells in partition.
	 * 
	 *  For an example, if there are two sorted vector based on value like following, where
	 *   V2 is sorted data, and I2 are its corresponding indices.
	 *   
	 *   In this case, for partition with value = 4, has two different indices -- 6, and 7.
	 *   This function will set indices to both cells to minimum value of 6.
	 *   
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 *    Minimum indices set in the partition (I2")        5   2   8   6   6   1   4   3
	 * 
	 * @param vix
	 * @param vmb
	 */
	public static void setMinIndexInPartition(int[] vix, double[] vmb)
	{
		int iLastIndex = 0;
		double dLastVal = vix[iLastIndex];

    	for (int i = 0; i < vix.length-1; ++i)
    	{
    		while(i<vmb.length-1 && dLastVal == vmb[i+1]) ++i;
    		for (int j=iLastIndex+1; j<=i; ++j) 
    			vix[j] = vix[iLastIndex];
    		if (i < vix.length-1) {
        		iLastIndex = i+1;
        		dLastVal = vmb[i+1];
    		}
    	}
	}

	/**
	 * This function will set maximum index in the partition to all cells in partition.
	 * 
	 *  For an example, if there are two sorted vector based on value like following, where
	 *   V2 is sorted data, and I2 are its corresponding indices.
	 *   
	 *   In this case, for partition with value = 4, has two different indices -- 6, and 7.
	 *   This function will set indices to both cells to maximum value of 7.
	 *   
	 *  Sorting this data based on value will be (V2):      2   3   3   4   4   6   7   9	
	 *      Then indices will be ordered as (I2):           5   2   8   6   7   1   4   3
	 * 
	 *    Maximum indices set in the partition (I2")        5   2   8   7   7   1   4   3
	 * 
	 * @param vix
	 * @param vmb
	 */
	public static void setMaxIndexInPartition(int[] vix, double[] vmb)
	{
		int iLastIndex = vix.length-1;
		double dLastVal = vix[iLastIndex];

    	for (int i = vix.length-1; i > 0;)
    	{
    		while(i>0 && dLastVal == vmb[i]) --i;
    		for (int j=i+1; j<iLastIndex; ++j) 
    			vix[j] = vix[iLastIndex];
    		if (i > 0) {
        		iLastIndex = i;
        		dLastVal = vmb[i];
    		}
    	}
	}

}