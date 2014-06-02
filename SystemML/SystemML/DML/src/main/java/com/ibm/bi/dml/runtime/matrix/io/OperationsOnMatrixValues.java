/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction.IndexRange;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static void performScalarIgnoreIndexes(MatrixValue value_in, MatrixValue value_out, ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.scalarOperations(op, value_out);
	}
	
	public static void performScalarIgnoreIndexesInPlace(MatrixValue value_in, ScalarOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.scalarOperationsInPlace(op);
	}
	
	public static void performUnaryIgnoreIndexes(MatrixValue value_in, MatrixValue value_out, UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.unaryOperations(op, value_out);
	}
	
	public static void performUnaryIgnoreIndexesInPlace(MatrixValue value_in, UnaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.unaryOperationsInPlace(op);
	}

/*	public static void performBuiltinIgnoreIndexes(MatrixValue value_in, double constant, 
			MatrixValue value_out, Builtin.SupportedOperation op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		boolean sparseSafe=false;
		if(op==Builtin.SupportedOperation.ABS || op==Builtin.SupportedOperation.SIN 
				|| op==Builtin.SupportedOperation.SQRT || op==Builtin.SupportedOperation.TAN)
			sparseSafe=true;
		value_in.builtinOperations(op, constant, value_out, sparseSafe);
	}*/

	
	public static void performReorg(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, ReorgOperator op, 
			int startRow, int startColumn, int length) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.fn.execute(indexes_in, indexes_out);
		
		//operation on the cells inside the value
		value_out=value_in.reorgOperations(op, value_out, startRow, startColumn, length);
		
	}
/*	public static void performAppend(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, ReorgOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operates only on indexes
		op.fn.execute(indexes_in, indexes_out);
		
		//operation on the cells inside the value
		value_out=value_in.reorgOperations(op, value_out, 0, 0, 0);
	}*/
	
	public static void performAppend(MatrixValue value_in1, MatrixValue value_in2,
			ArrayList<IndexedMatrixValue> outlist, int blockRowFactor, int blockColFactor, boolean m2IsLast, int nextNCol) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in1.appendOperations(value_in2, outlist, blockRowFactor, blockColFactor, m2IsLast, nextNCol);
	}
	
	public static void performZeroOut(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, IndexRange range, boolean complementary) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_out=value_in.zeroOutOperations(value_out, range, complementary);
		indexes_out.setIndexes(indexes_in);
	}
	
	public static void performSlice(MatrixIndexes indexes_in, MatrixValue value_in, 
			ArrayList<IndexedMatrixValue> outlist, IndexRange range, 
			int rowCut, int colCut, int blockRowFactor, int blockColFactor, int boundaryRlen, int boundaryClen) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_in.sliceOperations(outlist, range, rowCut, colCut, blockRowFactor, blockColFactor, boundaryRlen, boundaryClen);
	}
	
	// ------------- Tertiary Operations -------------
	// tertiary where all three inputs are matrices
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, MatrixIndexes indexes_in2, MatrixValue value_in2, 
			MatrixIndexes indexes_in3, MatrixValue value_in3, HashMap<MatrixIndexes, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, value_in2, value_in3, ctableResult);
	}
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, MatrixIndexes indexes_in2, MatrixValue value_in2, 
			double scalar_in3, HashMap<MatrixIndexes, Double> ctableResult, Operator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, value_in2, scalar_in3, ctableResult);
	}
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, double scalar_in2, 
			double scalar_in3, HashMap<MatrixIndexes, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, scalar_in2, scalar_in3, ctableResult);
	}
	
	// tertiary where first input is a matrix, and second is scalars (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, double scalar_in2, boolean left,
			int brlen, HashMap<MatrixIndexes, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, indexes_in1, scalar_in2, left, brlen, ctableResult);
	}
	
	// tertiary where first and third inputs are matrices, and second is a scalars (double)
	public static void performTertiary(MatrixIndexes indexes_in1, MatrixValue value_in1, double scalar_in2, 
			MatrixIndexes indexes_in3, MatrixValue value_in3, HashMap<MatrixIndexes, Double> ctableResult, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		value_in1.tertiaryOperations(op, scalar_in2, value_in3, ctableResult);
	}
	// -----------------------------------------------------
	
	//binary operations are those that the indexes of both cells have to be matched
	public static void performBinaryIgnoreIndexes(MatrixValue value1, MatrixValue value2, 
			MatrixValue value_out, BinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		value_out=value1.binaryOperations(op, value2, value_out);
	}
	
	public static void startAggregation(MatrixValue value_out, MatrixValue correction, AggregateOperator op, 
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
					        || ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinFunctionCode.MININDEX) 
					        ){
						outRow = rlen;
						outCol = 1;
						corRow = rlen;
						corCol = 1;
					}else{
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
			
	/*		if(op.correctionLocation==1)
			{
				outRow=rlen-1;
				outCol=clen;
				corRow=1;
				corCol=clen;
			}else if(op.correctionLocation==2)
			{
				outRow=rlen;
				outCol=clen-1;
				corRow=rlen;
				corCol=1;
			}else if(op.correctionLocation==0)
			{
				outRow=rlen;
				outCol=clen;
				corRow=rlen;
				corCol=clen;
			}
			else
				throw new DMLRuntimeException("unrecognized correctionLocation: "+op.correctionLocation);	*/
			
			value_out.reset(outRow, outCol, sparseHint);
			correction.reset(corRow, corCol, false);
			
		}else
		{
			if(op.initialValue==0)
				value_out.reset(rlen, clen, sparseHint);
			else
				value_out.resetDenseWithValue(rlen, clen, op.initialValue);
		}
	}
	
	public static void incrementalAggregation(MatrixValue value_agg, MatrixValue correction, MatrixValue value_add, 
			AggregateOperator op, boolean imbededCorrection) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(op.correctionExists)
		{
			if(!imbededCorrection || op.correctionLocation==CorrectionLocationType.NONE)
				value_agg.incrementalAggregate(op, correction, value_add);
			else
				value_agg.incrementalAggregate(op, value_add);
		}
		else
			value_agg.binaryOperationsInPlace(op.increOp, value_add);
	}
	
	public static void performRandUnary(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, int brlen, int bclen)
	{
		indexes_out.setIndexes(indexes_in);
		value_out.copy(value_in);
	}
	
	public static void performAggregateUnary(MatrixIndexes indexes_in, MatrixValue value_in, MatrixIndexes indexes_out, 
			MatrixValue value_out, AggregateUnaryOperator op,int brlen, int bclen)
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		op.indexFn.execute(indexes_in, indexes_out);
		
		//perform on the value
		value_out=value_in.aggregateUnaryOperations(op, value_out, brlen, bclen, indexes_in);
	}
	
	public static void performAggregateBinary(MatrixIndexes indexes1, MatrixValue value1, MatrixIndexes indexes2, MatrixValue value2, 
			MatrixIndexes indexes_out, MatrixValue value_out, AggregateBinaryOperator op, boolean partialMult)
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operate on the value indexes first
		//tempCell1.setIndexes(indexes1);
		//tempCell2.setIndexes(indexes2);
		//CellOperations.performMMCJIndexOnly(tempCell1, tempCell2, tempCell1);
		indexes_out.setIndexes(indexes1.getRowIndex(), indexes2.getColumnIndex());
		
		//perform on the value
		value_out=value1.aggregateBinaryOperations(indexes1, value1, indexes2, value2, value_out, op, partialMult);
	}
	
	//including scalar, reorg and aggregateUnary operations
/*	public static void performAllUnary(MatrixIndexes indexes_in, MatrixValue value_in, 
			MatrixIndexes indexes_out, MatrixValue value_out, Instruction ins) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(ins instanceof Scalar.InstructionType)
		{
			performScalarIgnoreIndexes(value_in, ((Scalar.InstructionType) ins).constant, 
					value_out, (Scalar.SupportedOperation)ins.operation);
			indexes_out.setIndexes(indexes_in);
		}
		else if(ins instanceof Reorg.InstructionType)
			performReorg(indexes_in, value_in, indexes_out, value_out, 
					(Reorg.SupportedOperation)ins.operation);
		else if(ins instanceof AggregateUnary.InstructionType)
			performAggregateUnary(indexes_in, value_in, indexes_out, value_out, 
					(AggregateUnary.SupportedOperation)ins.operation,
					numRowsInBlock, numColsInBlock);
		else
			throw new DMLUnsupportedOperationException("Operation unsupported");
	}*/

	public static void performAggregateBinaryIgnoreIndexes(
			MatrixValue value1, MatrixValue value2,
			MatrixValue value_out, AggregateBinaryOperator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException {
			
		//perform on the value
		value_out=value1.aggregateBinaryOperations(value1, value2, value_out, op);
	}
	
	static int numRowsInBlock, numColsInBlock ;
	public static void setNumRowsInBlock(int i) {
		numRowsInBlock = i ;
	}
	public static void setNumColumnsInBlock(int i) {
		numColsInBlock = i;
	}
}
