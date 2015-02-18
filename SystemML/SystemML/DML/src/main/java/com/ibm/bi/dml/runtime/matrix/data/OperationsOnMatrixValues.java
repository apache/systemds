/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ScalarOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;

public class OperationsOnMatrixValues 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static void performScalarIgnoreIndexes(MatrixValue valueIn, MatrixValue valueOut, ScalarOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.scalarOperations(op, valueOut);
	}
	
	public static void performUnaryIgnoreIndexes(MatrixValue valueIn, MatrixValue valueOut, UnaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.unaryOperations(op, valueOut);
	}
	
	public static void performUnaryIgnoreIndexesInPlace(MatrixValue valueIn, UnaryOperator op) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.unaryOperationsInPlace(op);
	}
	
	public static void performReorg(MatrixIndexes indexesIn, MatrixValue valueIn, MatrixIndexes indexesOut, 
			         MatrixValue valueOut, ReorgOperator op, int startRow, int startColumn, int length) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.fn.execute(indexesIn, indexesOut);
		
		//operation on the cells inside the value
		valueIn.reorgOperations(op, valueOut, startRow, startColumn, length);
	}

	public static void performAppend(MatrixValue valueIn1, MatrixValue valueIn2,
			ArrayList<IndexedMatrixValue> outlist, int blockRowFactor, int blockColFactor, boolean m2IsLast, int nextNCol) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn1.appendOperations(valueIn2, outlist, blockRowFactor, blockColFactor, m2IsLast, nextNCol);
	}
	
	public static void performZeroOut(MatrixIndexes indexesIn, MatrixValue valueIn, 
			MatrixIndexes indexesOut, MatrixValue valueOut, IndexRange range, boolean complementary) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.zeroOutOperations(valueOut, range, complementary);
		indexesOut.setIndexes(indexesIn);
	}
	
	public static void performSlice(MatrixIndexes indexesIn, MatrixValue valueIn, 
			ArrayList<IndexedMatrixValue> outlist, IndexRange range, 
			int rowCut, int colCut, int blockRowFactor, int blockColFactor, int boundaryRlen, int boundaryClen) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.sliceOperations(outlist, range, rowCut, colCut, blockRowFactor, blockColFactor, boundaryRlen, boundaryClen);
	}
	
	// ------------- Tertiary Operations -------------
	// tertiary where all three inputs are matrices
	public static void performTertiary(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.tertiaryOperations(op, valueIn2, valueIn3, ctableResult, ctableResultBlock);
	}
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public static void performTertiary(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			double scalarIn3, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock, Operator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.tertiaryOperations(op, valueIn2, scalarIn3, false, ctableResult, ctableResultBlock);
	}
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public static void performTertiary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			double scalarIn3, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.tertiaryOperations(op, scalarIn2, scalarIn3, ctableResult, ctableResultBlock);
	}
	
	// tertiary where first input is a matrix, and second is scalars (double)
	public static void performTertiary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, boolean left,
			int brlen, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.tertiaryOperations(op, indexesIn1, scalarIn2, left, brlen, ctableResult, ctableResultBlock);
	}
	
	// tertiary where first and third inputs are matrices, and second is a scalars (double)
	public static void performTertiary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, HashMap<MatrixIndexes, Double> ctableResult, MatrixBlock ctableResultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.tertiaryOperations(op, scalarIn2, valueIn3, ctableResult, ctableResultBlock);
	}
	// -----------------------------------------------------
	
	//binary operations are those that the indexes of both cells have to be matched
	public static void performBinaryIgnoreIndexes(MatrixValue value1, MatrixValue value2, 
			MatrixValue valueOut, BinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value1.binaryOperations(op, value2, valueOut);
	}
	
	/**
	 * 
	 * @param valueOut
	 * @param correction
	 * @param op
	 * @param rlen
	 * @param clen
	 * @param sparseHint
	 * @param imbededCorrection
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void startAggregation(MatrixValue valueOut, MatrixValue correction, AggregateOperator op, 
			int rlen, int clen, boolean sparseHint, boolean imbededCorrection)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		int outRow=0, outCol=0, corRow=0, corCol=0;
		if(op.correctionExists)
		{
			if(!imbededCorrection)
			{
				switch(op.correctionLocation)
				{
				case NONE:
					outRow=rlen;
					outCol=clen;
					corRow=rlen;
					corCol=clen;
					break;
				case LASTROW:
					outRow=rlen-1;
					outCol=clen;
					corRow=1;
					corCol=clen;
					break;
				case LASTCOLUMN:
					if(op.increOp.fn instanceof Builtin 
					   && ( ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MAXINDEX 
					        || ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX) )
					{
						outRow = rlen;
						outCol = 1;
						corRow = rlen;
						corCol = 1;
					}
					else{
						outRow=rlen;
						outCol=clen-1;
						corRow=rlen;
						corCol=1;
					}
					break;
				case LASTTWOROWS:
					outRow=rlen-2;
					outCol=clen;
					corRow=2;
					corCol=clen;
					break;
				case LASTTWOCOLUMNS:
					outRow=rlen;
					outCol=clen-2;
					corRow=rlen;
					corCol=2;
					break;
				default:
						throw new DMLRuntimeException("unrecognized correctionLocation: "+op.correctionLocation);
				}
			}else
			{
				outRow=rlen;
				outCol=clen;
				corRow=rlen;
				corCol=clen;
			}
			
			//set initial values according to operator
			if(op.initialValue==0) {
				valueOut.reset(outRow, outCol, sparseHint);
				correction.reset(corRow, corCol, false);
			}
			else {
				valueOut.resetDenseWithValue(outRow, outCol, op.initialValue);
				correction.resetDenseWithValue(corRow, corCol, op.initialValue);
			}
			
		}
		else
		{
			if(op.initialValue==0)
				valueOut.reset(rlen, clen, sparseHint);
			else
				valueOut.resetDenseWithValue(rlen, clen, op.initialValue);
		}
	}
	
	public static void incrementalAggregation(MatrixValue valueAgg, MatrixValue correction, MatrixValue valueAdd, 
			AggregateOperator op, boolean imbededCorrection) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.correctionExists)
		{
			if(!imbededCorrection || op.correctionLocation==CorrectionLocationType.NONE)
				valueAgg.incrementalAggregate(op, correction, valueAdd);
			else
				valueAgg.incrementalAggregate(op, valueAdd);
		}
		else
			valueAgg.binaryOperationsInPlace(op.increOp, valueAdd);
	}
	
	public static void performRandUnary(MatrixIndexes indexesIn, MatrixValue valueIn, 
			MatrixIndexes indexesOut, MatrixValue valueOut, int brlen, int bclen)
	{
		indexesOut.setIndexes(indexesIn);
		valueOut.copy(valueIn);
	}
	
	public static void performAggregateUnary(MatrixIndexes indexesIn, MatrixValue valueIn, MatrixIndexes indexesOut, 
			MatrixValue valueOut, AggregateUnaryOperator op,int brlen, int bclen)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.indexFn.execute(indexesIn, indexesOut);
		
		//perform on the value
		valueIn.aggregateUnaryOperations(op, valueOut, brlen, bclen, indexesIn);
	}
	
	public static void performAggregateBinary(MatrixIndexes indexes1, MatrixValue value1, MatrixIndexes indexes2, MatrixValue value2, 
			MatrixIndexes indexesOut, MatrixValue valueOut, AggregateBinaryOperator op)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//compute output index
		indexesOut.setIndexes(indexes1.getRowIndex(), indexes2.getColumnIndex());
		
		//perform on the value
		value1.aggregateBinaryOperations(indexes1, value1, indexes2, value2, valueOut, op);
	}

	public static void performAggregateBinaryIgnoreIndexes(
			MatrixValue value1, MatrixValue value2,
			MatrixValue valueOut, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException {
			
		//perform on the value
		value1.aggregateBinaryOperations(value1, value2, valueOut, op);
	}
}
