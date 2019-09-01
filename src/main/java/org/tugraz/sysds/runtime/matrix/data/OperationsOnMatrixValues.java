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


package org.tugraz.sysds.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tugraz.sysds.lops.PartialAggregate.CorrectionLocationType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.functionobjects.Builtin;
import org.tugraz.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.tugraz.sysds.runtime.matrix.mapred.IndexedMatrixValue;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.ReorgOperator;
import org.tugraz.sysds.runtime.util.IndexRange;
import org.tugraz.sysds.runtime.util.UtilFunctions;


public class OperationsOnMatrixValues 
{
	public static void performReorg(MatrixIndexes indexesIn, MatrixValue valueIn, MatrixIndexes indexesOut, 
			MatrixValue valueOut, ReorgOperator op, int startRow, int startColumn, int length) {
		//operate on the value indexes first
		op.fn.execute(indexesIn, indexesOut);
		
		//operation on the cells inside the value
		valueIn.reorgOperations(op, valueOut, startRow, startColumn, length);
	}

	public static void performAppend(MatrixValue valueIn1, MatrixValue valueIn2,
			ArrayList<IndexedMatrixValue> outlist, int blen, boolean cbind, boolean m2IsLast, int nextNCol) {
		valueIn1.append(valueIn2, outlist, blen, cbind, m2IsLast, nextNCol);
	}
	
	public static void performZeroOut(MatrixIndexes indexesIn, MatrixValue valueIn, 
			MatrixIndexes indexesOut, MatrixValue valueOut, IndexRange range, boolean complementary) {
		valueIn.zeroOutOperations(valueOut, range, complementary);
		indexesOut.setIndexes(indexesIn);
	}
	
	// ------------- Ternary Operations -------------
	public static void performCtable(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) {
		//operation on the cells inside the value
		valueIn1.ctableOperations(op, valueIn2, valueIn3, resultMap, resultBlock);
	}
	
	public static void performCtable(MatrixIndexes indexesIn1, MatrixValue valueIn1, MatrixIndexes indexesIn2, MatrixValue valueIn2, 
			double scalarIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op) {
		//operation on the cells inside the value
		valueIn1.ctableOperations(op, valueIn2, scalarIn3, false, resultMap, resultBlock);
	}
	
	public static void performCtable(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			double scalarIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) {
		//operation on the cells inside the value
		valueIn1.ctableOperations(op, scalarIn2, scalarIn3, resultMap, resultBlock);
	}
	
	public static void performCtable(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, boolean left,
			int blen, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) {
		//operation on the cells inside the value
		valueIn1.ctableOperations(op, indexesIn1, scalarIn2, left, blen, resultMap, resultBlock);
	}
	
	public static void performCtable(MatrixIndexes indexesIn1, MatrixValue valueIn1, double scalarIn2, 
			MatrixIndexes indexesIn3, MatrixValue valueIn3, CTableMap resultMap, MatrixBlock resultBlock, Operator op ) {
		//operation on the cells inside the value
		valueIn1.ctableOperations(op, scalarIn2, valueIn3, resultMap, resultBlock);
	}
	// -----------------------------------------------------
	
	//binary operations are those that the indexes of both cells have to be matched
	public static void performBinaryIgnoreIndexes(MatrixValue value1, MatrixValue value2, 
			MatrixValue valueOut, BinaryOperator op) {
		value1.binaryOperations(op, value2, valueOut);
	}

	public static void startAggregation(MatrixValue valueOut, MatrixValue correction, AggregateOperator op, 
			int rlen, int clen, boolean sparseHint, boolean imbededCorrection) {
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
					   && ( ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinCode.MAXINDEX 
					        || ((Builtin)(op.increOp.fn)).bFunc == Builtin.BuiltinCode.MININDEX) )
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
				case LASTFOURROWS:
					outRow=rlen-4;
					outCol=clen;
					corRow=4;
					corCol=clen;
					break;
				case LASTFOURCOLUMNS:
					outRow=rlen;
					outCol=clen-4;
					corRow=rlen;
					corCol=4;
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
				valueOut.reset(Math.max(outRow,0), Math.max(outCol,0), sparseHint);
				correction.reset(Math.max(corRow,0), Math.max(corCol,0), false);
			}
			else {
				valueOut.reset(Math.max(outRow, 0), Math.max(outCol,0), op.initialValue);
				correction.reset(Math.max(corRow,0), Math.max(corCol,0), op.initialValue);
			}
		}
		else
		{
			if(op.initialValue==0)
				valueOut.reset(rlen, clen, sparseHint);
			else
				valueOut.reset(rlen, clen, op.initialValue);
		}
	}
	
	public static void incrementalAggregation(MatrixValue valueAgg, MatrixValue correction, MatrixValue valueAdd, 
			AggregateOperator op, boolean imbededCorrection) {
		incrementalAggregation(valueAgg, correction, valueAdd, op, imbededCorrection, true);
	}
	
	
	public static void incrementalAggregation(MatrixValue valueAgg, MatrixValue correction, MatrixValue valueAdd, 
			AggregateOperator op, boolean imbededCorrection, boolean deep)
	{
		if(op.correctionExists)
		{
			if(!imbededCorrection || op.correctionLocation==CorrectionLocationType.NONE)
				valueAgg.incrementalAggregate(op, correction, valueAdd, deep);
			else
				valueAgg.incrementalAggregate(op, valueAdd);
		}
		else
			valueAgg.binaryOperationsInPlace(op.increOp, valueAdd);
	}
	
	public static void performAggregateUnary(MatrixIndexes indexesIn, MatrixValue valueIn, MatrixIndexes indexesOut, 
			MatrixValue valueOut, AggregateUnaryOperator op,int blen) {
		//operate on the value indexes first
		op.indexFn.execute(indexesIn, indexesOut);
		
		//perform on the value
		valueIn.aggregateUnaryOperations(op, valueOut, blen, indexesIn);
	}
	
	public static MatrixBlock matMult(MatrixIndexes indexes1, MatrixBlock value1, MatrixIndexes indexes2, MatrixBlock value2, 
			MatrixIndexes indexesOut, MatrixBlock valueOut, AggregateBinaryOperator op) {
		//compute output index
		indexesOut.setIndexes(indexes1.getRowIndex(), indexes2.getColumnIndex());
		//perform on the value
		return value1.aggregateBinaryOperations(indexes1, value1, indexes2, value2, valueOut, op);
	}

	public static MatrixBlock matMult(MatrixBlock value1, MatrixBlock value2,
			MatrixBlock valueOut, AggregateBinaryOperator op) {
		//perform on the value
		return value1.aggregateBinaryOperations(value1, value2, valueOut, op);
	}

	@SuppressWarnings("rawtypes")
	public static List performSlice(IndexRange ixrange, int blen, int iix, int jix, CacheBlock in) {
		if( in instanceof MatrixBlock )
			return performSlice(ixrange, blen, iix, jix, (MatrixBlock)in);
		else if( in instanceof FrameBlock )
			return performSlice(ixrange, blen, iix, jix, (FrameBlock)in);
		throw new DMLRuntimeException("Unsupported cache block type: "+in.getClass().getName());
	}
	
	@SuppressWarnings("rawtypes")
	public static List performSlice(IndexRange ixrange, int blen, int iix, int jix, MatrixBlock in) {
		IndexedMatrixValue imv = new IndexedMatrixValue(new MatrixIndexes(iix, jix), (MatrixBlock)in);
		ArrayList<IndexedMatrixValue> outlist = new ArrayList<>();
		performSlice(imv, ixrange, blen, outlist);
		return SparkUtils.fromIndexedMatrixBlockToPair(outlist);
	}

	public static void performSlice(IndexedMatrixValue in, IndexRange ixrange, int blen, ArrayList<IndexedMatrixValue> outlist) {
		long cellIndexTopRow = UtilFunctions.computeCellIndex(in.getIndexes().getRowIndex(), blen, 0);
		long cellIndexBottomRow = UtilFunctions.computeCellIndex(in.getIndexes().getRowIndex(), blen, in.getValue().getNumRows()-1);
		long cellIndexLeftCol = UtilFunctions.computeCellIndex(in.getIndexes().getColumnIndex(), blen, 0);
		long cellIndexRightCol = UtilFunctions.computeCellIndex(in.getIndexes().getColumnIndex(), blen, in.getValue().getNumColumns()-1);
		
		long cellIndexOverlapTop = Math.max(cellIndexTopRow, ixrange.rowStart);
		long cellIndexOverlapBottom = Math.min(cellIndexBottomRow, ixrange.rowEnd);
		long cellIndexOverlapLeft = Math.max(cellIndexLeftCol, ixrange.colStart);
		long cellIndexOverlapRight = Math.min(cellIndexRightCol, ixrange.colEnd);
		
		//check if block is outside the indexing range
		if(cellIndexOverlapTop>cellIndexOverlapBottom || cellIndexOverlapLeft>cellIndexOverlapRight) {
			return;
		}
		
		IndexRange tmpRange = new IndexRange(
			UtilFunctions.computeCellInBlock(cellIndexOverlapTop, blen), 
			UtilFunctions.computeCellInBlock(cellIndexOverlapBottom, blen), 
			UtilFunctions.computeCellInBlock(cellIndexOverlapLeft, blen), 
			UtilFunctions.computeCellInBlock(cellIndexOverlapRight, blen));
		
		int rowCut=UtilFunctions.computeCellInBlock(ixrange.rowStart, blen);
		int colCut=UtilFunctions.computeCellInBlock(ixrange.colStart, blen);
		
		int rowsInLastBlock = (int)((ixrange.rowEnd-ixrange.rowStart+1)%blen);
		if(rowsInLastBlock==0) 
			rowsInLastBlock=blen;
		int colsInLastBlock = (int)((ixrange.colEnd-ixrange.colStart+1)%blen);
		if(colsInLastBlock==0) 
			colsInLastBlock=blen;
		
		long resultBlockIndexTop=UtilFunctions.computeBlockIndex(cellIndexOverlapTop-ixrange.rowStart+1, blen);
		long resultBlockIndexBottom=UtilFunctions.computeBlockIndex(cellIndexOverlapBottom-ixrange.rowStart+1, blen);
		long resultBlockIndexLeft=UtilFunctions.computeBlockIndex(cellIndexOverlapLeft-ixrange.colStart+1, blen);
		long resultBlockIndexRight=UtilFunctions.computeBlockIndex(cellIndexOverlapRight-ixrange.colStart+1, blen);
		
		int boundaryRlen = blen;
		int boundaryClen = blen;
		long finalBlockIndexBottom=UtilFunctions.computeBlockIndex(ixrange.rowEnd-ixrange.rowStart+1, blen);
		long finalBlockIndexRight=UtilFunctions.computeBlockIndex(ixrange.colEnd-ixrange.colStart+1, blen);
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
		in.getValue().slice(outlist, tmpRange, rowCut, colCut, blen, boundaryRlen, boundaryClen);
	}

	public static void performShift(IndexedMatrixValue in, IndexRange ixrange, int blen, long rlen, long clen, ArrayList<IndexedMatrixValue> outlist) {
		MatrixIndexes ix = in.getIndexes();
		MatrixBlock mb = (MatrixBlock)in.getValue();
		
		long start_lhs_globalRowIndex = ixrange.rowStart + (ix.getRowIndex()-1)*blen;
		long start_lhs_globalColIndex = ixrange.colStart + (ix.getColumnIndex()-1)*blen;
		long end_lhs_globalRowIndex = start_lhs_globalRowIndex + mb.getNumRows() - 1;
		long end_lhs_globalColIndex = start_lhs_globalColIndex + mb.getNumColumns() - 1;
		
		long start_lhs_rowIndex = UtilFunctions.computeBlockIndex(start_lhs_globalRowIndex, blen);
		long end_lhs_rowIndex = UtilFunctions.computeBlockIndex(end_lhs_globalRowIndex, blen);
		long start_lhs_colIndex = UtilFunctions.computeBlockIndex(start_lhs_globalColIndex, blen);
		long end_lhs_colIndex = UtilFunctions.computeBlockIndex(end_lhs_globalColIndex, blen);
		
		for(long leftRowIndex = start_lhs_rowIndex; leftRowIndex <= end_lhs_rowIndex; leftRowIndex++) {
			for(long leftColIndex = start_lhs_colIndex; leftColIndex <= end_lhs_colIndex; leftColIndex++) {
				
				// Calculate global index of right hand side block
				long lhs_rl = Math.max((leftRowIndex-1)*blen+1, start_lhs_globalRowIndex);
				long lhs_ru = Math.min(leftRowIndex*blen, end_lhs_globalRowIndex);
				long lhs_cl = Math.max((leftColIndex-1)*blen+1, start_lhs_globalColIndex);
				long lhs_cu = Math.min(leftColIndex*blen, end_lhs_globalColIndex);
				
				int lhs_lrl = UtilFunctions.computeCellInBlock(lhs_rl, blen);
				int lhs_lru = UtilFunctions.computeCellInBlock(lhs_ru, blen);
				int lhs_lcl = UtilFunctions.computeCellInBlock(lhs_cl, blen);
				int lhs_lcu = UtilFunctions.computeCellInBlock(lhs_cu, blen);
				
				long rhs_rl = lhs_rl - ixrange.rowStart + 1;
				long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
				long rhs_cl = lhs_cl - ixrange.colStart + 1;
				long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
				
				int rhs_lrl = UtilFunctions.computeCellInBlock(rhs_rl, blen);
				int rhs_lru = UtilFunctions.computeCellInBlock(rhs_ru, blen);
				int rhs_lcl = UtilFunctions.computeCellInBlock(rhs_cl, blen);
				int rhs_lcu = UtilFunctions.computeCellInBlock(rhs_cu, blen);
				
				MatrixBlock slicedRHSBlk = mb.slice(rhs_lrl, rhs_lru, rhs_lcl, rhs_lcu, new MatrixBlock());
				
				int lbrlen = UtilFunctions.computeBlockSize(rlen, leftRowIndex, blen);
				int lbclen = UtilFunctions.computeBlockSize(clen, leftColIndex, blen);
				MatrixBlock resultBlock = new MatrixBlock(lbrlen, lbclen, false);
				resultBlock = resultBlock.leftIndexingOperations(slicedRHSBlk, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, null, UpdateType.COPY);
				outlist.add(new IndexedMatrixValue(new MatrixIndexes(leftRowIndex, leftColIndex), resultBlock));
			}
		}
	}

	public static void performMapGroupedAggregate( Operator op, IndexedMatrixValue inTarget, MatrixBlock groups, int ngroups, int blen, ArrayList<IndexedMatrixValue> outlist )
	{
		MatrixIndexes ix = inTarget.getIndexes();
		MatrixBlock target = (MatrixBlock)inTarget.getValue();
		
		//execute grouped aggregate operations
		MatrixBlock out = groups.groupedAggOperations(target, null, new MatrixBlock(), ngroups, op);
		
		if( out.getNumRows()<=blen && out.getNumColumns()<=blen )
		{
			//single output block
			outlist.add( new IndexedMatrixValue(new MatrixIndexes(1,ix.getColumnIndex()), out) );
		}
		else
		{
			//multiple output blocks (by op def, single column block )
			for(int blockRow = 0; blockRow < (int)Math.ceil(out.getNumRows()/(double)blen); blockRow++)
			{
				int maxRow = (blockRow*blen + blen < out.getNumRows()) ? blen : out.getNumRows() - blockRow*blen;
				int row_offset = blockRow*blen;

				//copy submatrix to block
				MatrixBlock tmp = out.slice(row_offset, row_offset+maxRow-1);
				
				//append block to result cache
				outlist.add(new IndexedMatrixValue(new MatrixIndexes(blockRow+1,ix.getColumnIndex()), tmp));
			}
		}
	}
	
	@SuppressWarnings("rawtypes")
	public static ArrayList performSlice(IndexRange ixrange, int blen, int iix, int jix, FrameBlock in) {
		Pair<Long, FrameBlock> lfp = new Pair<>(new Long(((iix-1)*blen)+1), in);
		ArrayList<Pair<Long, FrameBlock>> outlist = new ArrayList<>();
		performSlice(lfp, ixrange, blen, outlist);
	
		return outlist;
	}

	
	/**
	 * This function will get slice of the input frame block overlapping in overall slice(Range), slice has requested for.
	 * 
	 * @param in ?
	 * @param ixrange index range
	 * @param blen number of rows in a block
	 * @param blen number of columns in a block
	 * @param outlist list of pairs of frame blocks
	 */
	public static void performSlice(Pair<Long,FrameBlock> in, IndexRange ixrange, int blen, ArrayList<Pair<Long,FrameBlock>> outlist) {
		long index = in.getKey();
		FrameBlock block = in.getValue();
		
		// Get Block indexes (rows and columns boundaries)
		long cellIndexTopRow = index;
		long cellIndexBottomRow = index+block.getNumRows()-1;
		long cellIndexLeftCol = 1;
		long cellIndexRightCol = block.getNumColumns();
		
		// Calculate block boundaries with range of slice to be performed (Global index)
		long cellIndexOverlapTop = Math.max(cellIndexTopRow, ixrange.rowStart);
		long cellIndexOverlapBottom = Math.min(cellIndexBottomRow, ixrange.rowEnd);
		long cellIndexOverlapLeft = Math.max(cellIndexLeftCol, ixrange.colStart);
		long cellIndexOverlapRight = Math.min(cellIndexRightCol, ixrange.colEnd);
		
		//check if block is outside the indexing range
		if(cellIndexOverlapTop>cellIndexOverlapBottom || cellIndexOverlapLeft>cellIndexOverlapRight) {
			return;
		}
		
		// Create IndexRange for the slice to be performed on this block.
		IndexRange tmpRange = new IndexRange(
				cellIndexOverlapTop - index,
				cellIndexOverlapBottom - index,
				cellIndexOverlapLeft -1,
				cellIndexOverlapRight - 1);
		
		// Get Top Row and Left column cutting point. 
		int rowCut=(int)(ixrange.rowStart-index);
		
		// Get indices for result block
		long resultBlockIndexTop=UtilFunctions.computeBlockIndex(cellIndexOverlapTop, blen);
		long resultBlockIndexBottom=UtilFunctions.computeBlockIndex(cellIndexOverlapBottom, blen);
		
		//allocate space for the output value
		for(long r=resultBlockIndexTop; r<=resultBlockIndexBottom; r++)
		{
			ValueType[] schema = Arrays.copyOfRange(block.getSchema(), (int)tmpRange.colStart, (int)tmpRange.colEnd+1);
			long iResultIndex = Math.max(((r-1)*blen - ixrange.rowStart + 1), 0);
			Pair<Long,FrameBlock> out=new Pair<>(new Long(iResultIndex+1), new FrameBlock(schema));
			outlist.add(out);
		}
		
		//execute actual slice operation
		block.slice(outlist, tmpRange, rowCut);
	}

	public static void performShift(Pair<Long,FrameBlock> in, IndexRange ixrange, int blenLeft, long rlen, long clen, ArrayList<Pair<Long,FrameBlock>> outlist) {
		Long ix = in.getKey();
		FrameBlock fb = in.getValue();
		long start_lhs_globalRowIndex = ixrange.rowStart + (ix-1);
		long start_lhs_globalColIndex = ixrange.colStart;
		long end_lhs_globalRowIndex = start_lhs_globalRowIndex + fb.getNumRows() - 1;
		long end_lhs_globalColIndex = ixrange.colEnd;
		
		long start_lhs_rowIndex = UtilFunctions.computeBlockIndex(start_lhs_globalRowIndex, blenLeft);
		long end_lhs_rowIndex = UtilFunctions.computeBlockIndex(end_lhs_globalRowIndex, blenLeft);

		for(long leftRowIndex = start_lhs_rowIndex; leftRowIndex <= end_lhs_rowIndex; leftRowIndex++) {
				
			// Calculate global index of right hand side block
			long lhs_rl = Math.max((leftRowIndex-1)*blenLeft+1, start_lhs_globalRowIndex);
			long lhs_ru = Math.min(leftRowIndex*blenLeft, end_lhs_globalRowIndex);
			long lhs_cl = start_lhs_globalColIndex;
			long lhs_cu = end_lhs_globalColIndex;
			
			int lhs_lrl = UtilFunctions.computeCellInBlock(lhs_rl, blenLeft);
			int lhs_lru = UtilFunctions.computeCellInBlock(lhs_ru, blenLeft);
			int lhs_lcl = (int)lhs_cl-1;
			int lhs_lcu = (int)lhs_cu-1;
			
			long rhs_rl = lhs_rl - (ixrange.rowStart-1) - (ix-1);
			long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
			long rhs_cl = lhs_cl - ixrange.colStart + 1;
			long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
			
			// local indices are 0 (zero) based.
			int rhs_lrl = (int) (UtilFunctions.computeCellInBlock(rhs_rl, fb.getNumRows()));
			int rhs_lru = (int) (UtilFunctions.computeCellInBlock(rhs_ru, fb.getNumRows()));
			int rhs_lcl = (int)rhs_cl-1;
			int rhs_lcu = (int)rhs_cu-1;
			
			FrameBlock slicedRHSBlk = fb.slice(rhs_lrl, rhs_lru, rhs_lcl, rhs_lcu, new FrameBlock());
			
			int lblen = blenLeft;
			
			ValueType[] schemaPartialLeft = UtilFunctions.nCopies(lhs_lcl, ValueType.STRING);
			ValueType[] schemaRHS = Arrays.copyOfRange(fb.getSchema(), (int)(rhs_lcl), (int)(rhs_lcl-lhs_lcl+lhs_lcu+1));
			ValueType[] schema = UtilFunctions.copyOf(schemaPartialLeft, schemaRHS);
			ValueType[] schemaPartialRight = UtilFunctions.nCopies(lblen-schema.length, ValueType.STRING);
			schema = UtilFunctions.copyOf(schema, schemaPartialRight);
			FrameBlock resultBlock = new FrameBlock(schema);
			int iRHSRows = (int)(leftRowIndex<=rlen/blenLeft?blenLeft:rlen-(rlen/blenLeft)*blenLeft);
			resultBlock.ensureAllocatedColumns(iRHSRows);
			
			resultBlock = resultBlock.leftIndexingOperations(slicedRHSBlk, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, new FrameBlock());
			outlist.add(new Pair<>((leftRowIndex-1)*blenLeft+1, resultBlock));
		}
	}
}
