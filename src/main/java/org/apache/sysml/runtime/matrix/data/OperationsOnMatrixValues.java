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


package org.apache.sysml.runtime.matrix.data;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;

public class OperationsOnMatrixValues 
{
	
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
			ArrayList<IndexedMatrixValue> outlist, int blockRowFactor, int blockColFactor,  boolean cbind, boolean m2IsLast, int nextNCol) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn1.appendOperations(valueIn2, outlist, blockRowFactor, blockColFactor, cbind, m2IsLast, nextNCol);
	}
	
	public static void performZeroOut(MatrixIndexes indexesIn, MatrixValue valueIn, 
			MatrixIndexes indexesOut, MatrixValue valueOut, IndexRange range, boolean complementary) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		valueIn.zeroOutOperations(valueOut, range, complementary);
		indexesOut.setIndexes(indexesIn);
	}
	
	// ------------- Ternary Operations -------------
	// tertiary where all three inputs are matrices
	public static void performTernary(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.ternaryOperations(op, valueIn2, valueIn3, resultMap, resultBlock);
	}
	
	// tertiary where first two inputs are matrices, and third input is a scalar (double)
	public static void performTernary(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			double scalarIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.ternaryOperations(op, valueIn2, scalarIn3, false, resultMap, resultBlock);
	}
	
	// tertiary where first input is a matrix, and second and third inputs are scalars (double)
	public static void performTernary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			double scalarIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.ternaryOperations(op, scalarIn2, scalarIn3, resultMap, resultBlock);
	}
	
	// tertiary where first input is a matrix, and second is scalars (double)
	public static void performTernary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, boolean left,
			int brlen, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.ternaryOperations(op, indexesIn1, scalarIn2, left, brlen, resultMap, resultBlock);
	}
	
	// tertiary where first and third inputs are matrices, and second is a scalars (double)
	public static void performTernary(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//operation on the cells inside the value
		valueIn1.ternaryOperations(op, scalarIn2, valueIn3, resultMap, resultBlock);
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
	
	/**
	 * 
	 * @param val
	 * @param range
	 * @param brlen
	 * @param bclen
	 * @param outlist
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static void performSlice(IndexedMatrixValue in, IndexRange ixrange, int brlen, int bclen, ArrayList<IndexedMatrixValue> outlist) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		long cellIndexTopRow = UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), brlen, 0);
		long cellIndexBottomRow = UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), brlen, in.getValue().getNumRows()-1);
		long cellIndexLeftCol = UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), bclen, 0);
		long cellIndexRightCol = UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), bclen, in.getValue().getNumColumns()-1);
		
		long cellIndexOverlapTop = Math.max(cellIndexTopRow, ixrange.rowStart);
		long cellIndexOverlapBottom = Math.min(cellIndexBottomRow, ixrange.rowEnd);
		long cellIndexOverlapLeft = Math.max(cellIndexLeftCol, ixrange.colStart);
		long cellIndexOverlapRight = Math.min(cellIndexRightCol, ixrange.colEnd);
		
		//check if block is outside the indexing range
		if(cellIndexOverlapTop>cellIndexOverlapBottom || cellIndexOverlapLeft>cellIndexOverlapRight) {
			return;
		}
		
		IndexRange tmpRange = new IndexRange(
			UtilFunctions.cellInBlockCalculation(cellIndexOverlapTop, brlen), 
			UtilFunctions.cellInBlockCalculation(cellIndexOverlapBottom, brlen), 
			UtilFunctions.cellInBlockCalculation(cellIndexOverlapLeft, bclen), 
			UtilFunctions.cellInBlockCalculation(cellIndexOverlapRight, bclen));
		
		int rowCut=UtilFunctions.cellInBlockCalculation(ixrange.rowStart, brlen);
		int colCut=UtilFunctions.cellInBlockCalculation(ixrange.colStart, bclen);
		
		int rowsInLastBlock = (int)((ixrange.rowEnd-ixrange.rowStart+1)%brlen);
		if(rowsInLastBlock==0) 
			rowsInLastBlock=brlen;
		int colsInLastBlock = (int)((ixrange.colEnd-ixrange.colStart+1)%bclen);
		if(colsInLastBlock==0) 
			colsInLastBlock=bclen;
		
		long resultBlockIndexTop=UtilFunctions.blockIndexCalculation(cellIndexOverlapTop-ixrange.rowStart+1, brlen);
		long resultBlockIndexBottom=UtilFunctions.blockIndexCalculation(cellIndexOverlapBottom-ixrange.rowStart+1, brlen);
		long resultBlockIndexLeft=UtilFunctions.blockIndexCalculation(cellIndexOverlapLeft-ixrange.colStart+1, bclen);
		long resultBlockIndexRight=UtilFunctions.blockIndexCalculation(cellIndexOverlapRight-ixrange.colStart+1, bclen);
		
		int boundaryRlen = brlen;
		int boundaryClen = bclen;
		long finalBlockIndexBottom=UtilFunctions.blockIndexCalculation(ixrange.rowEnd-ixrange.rowStart+1, brlen);
		long finalBlockIndexRight=UtilFunctions.blockIndexCalculation(ixrange.colEnd-ixrange.colStart+1, bclen);
		if(resultBlockIndexBottom==finalBlockIndexBottom)
			boundaryRlen=rowsInLastBlock;
		if(resultBlockIndexRight==finalBlockIndexRight)
			boundaryClen=colsInLastBlock;
			
		//allocate space for the output value
		for(long r=resultBlockIndexTop; r<=resultBlockIndexBottom; r++)
			for(long c=resultBlockIndexLeft; c<=resultBlockIndexRight; c++)
			{
				IndexedMatrixValue out=new IndexedMatrixValue(new MatrixIndexes(), new MatrixBlock());
				out.getIndexes().setIndexes(r, c);
				outlist.add(out);
			}
		
		//execute actual slice operation
		in.getValue().sliceOperations(outlist, tmpRange, rowCut, colCut, brlen, bclen, boundaryRlen, boundaryClen);
	}
}
