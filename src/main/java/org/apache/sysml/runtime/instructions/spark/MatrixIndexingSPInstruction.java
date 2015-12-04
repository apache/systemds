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

package org.apache.sysml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcastMatrix;
import org.apache.sysml.runtime.instructions.spark.functions.IsBlockInRange;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;

public class MatrixIndexingSPInstruction  extends UnarySPInstruction
{
	
	/*
	 * This class implements the matrix indexing functionality inside CP.  
	 * Example instructions: 
	 *     rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
	 *         input=mVar1, output=mVar6, 
	 *         bounds = (Var2,Var3,Var4,Var5)
	 *         rowindex_lower: Var2, rowindex_upper: Var3 
	 *         colindex_lower: Var4, colindex_upper: Var5
	 *     leftIndex:mVar1:mVar2:Var3:Var4:Var5:Var6:mVar7
	 *         triggered by "mVar1[Var3:Var4, Var5:Var6] = mVar2"
	 *         the result is stored in mVar7
	 *  
	 */
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	protected SparkAggType _aggType = null;
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, 
			                          CPOperand out, SparkAggType aggtype, String opcode, String istr)
	{
		super(op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;

		_aggType = aggtype;
	}
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, 
			                          CPOperand out, String opcode, String istr)
	{
		super(op, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 8 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in = new CPOperand(parts[1]);
				CPOperand rl = new CPOperand(parts[2]);
				CPOperand ru = new CPOperand(parts[3]);
				CPOperand cl = new CPOperand(parts[4]);
				CPOperand cu = new CPOperand(parts[5]);
				CPOperand out = new CPOperand(parts[6]);
				SparkAggType aggtype = SparkAggType.valueOf(parts[7]);
				return new MatrixIndexingSPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, aggtype, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase("leftIndex") || opcode.equalsIgnoreCase("mapLeftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput = new CPOperand(parts[1]);
				CPOperand rhsInput = new CPOperand(parts[2]);
				CPOperand rl = new CPOperand(parts[3]);
				CPOperand ru = new CPOperand(parts[4]);
				CPOperand cl = new CPOperand(parts[5]);
				CPOperand cu = new CPOperand(parts[6]);
				CPOperand out = new CPOperand(parts[7]);
				return new MatrixIndexingSPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingSPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		//get indexing range
		long rl = ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral()).getLongValue();
		long ru = ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral()).getLongValue();
		long cl = ec.getScalarInput(colLower.getName(), colLower.getValueType(), colLower.isLiteral()).getLongValue();
		long cu = ec.getScalarInput(colUpper.getName(), colUpper.getValueType(), colUpper.isLiteral()).getLongValue();
		IndexRange ixrange = new IndexRange(rl, ru, cl, cu);
		
		//right indexing
		if( opcode.equalsIgnoreCase("rangeReIndex") )
		{
			//update and check output dimensions
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			mcOut.set(ru-rl+1, cu-cl+1, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock());
			checkValidOutputDimensions(mcOut);
			
			//execute right indexing operation (partitioning-preserving if possible)
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			if( isPartitioningPreservingRightIndexing(mcIn, ixrange) ) {
				out = in1.mapPartitionsToPair(
						new SliceBlockPartitionFunction(ixrange, mcOut), true);
			}
			else{
				out = in1.filter(new IsBlockInRange(rl, ru, cl, cu, mcOut))
			             .flatMapToPair(new SliceBlock(ixrange, mcOut));
				
				//aggregation if required 
				if( _aggType != SparkAggType.NONE )
					out = RDDAggregateUtils.mergeByKey(out);
			}
				
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase("leftIndex") || opcode.equalsIgnoreCase("mapLeftIndex"))
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			PartitionedBroadcastMatrix broadcastIn2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			
			//update and check output dimensions
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			MatrixCharacteristics mcLeft = ec.getMatrixCharacteristics(input1.getName());
			mcOut.set(mcLeft.getRows(), mcLeft.getCols(), mcLeft.getRowsPerBlock(), mcLeft.getColsPerBlock());
			checkValidOutputDimensions(mcOut);
			
			//note: always matrix rhs, scalars are preprocessed via cast to 1x1 matrix
			MatrixCharacteristics mcRight = ec.getMatrixCharacteristics(input2.getName());
				
			//sanity check matching index range and rhs dimensions
			if(!mcRight.dimsKnown()) {
				throw new DMLRuntimeException("The right input matrix dimensions are not specified for MatrixIndexingSPInstruction");
			}
			if(!(ru-rl+1 == mcRight.getRows() && cu-cl+1 == mcRight.getCols())) {
				throw new DMLRuntimeException("Invalid index range of leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"] vs ["+mcRight.getRows()+"x"+mcRight.getCols()+"]." );
			}
			
			if(opcode.equalsIgnoreCase("mapLeftIndex")) 
			{
				broadcastIn2 = sec.getBroadcastForVariable( input2.getName() );
				
				//partitioning-preserving mappartitions (key access required for broadcast loopkup)
				out = in1.mapPartitionsToPair(
						new LeftIndexPartitionFunction(broadcastIn2, ixrange, mcOut), true);
			}
			else {
				// Zero-out LHS
				in1 = in1.mapToPair(new ZeroOutLHS(false, mcLeft.getRowsPerBlock(), 
								mcLeft.getColsPerBlock(), rl, ru, cl, cu));
				
				// Slice RHS to merge for LHS
				in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() )
					    .flatMapToPair(new SliceRHSForLeftIndexing(rl, cl, mcLeft.getRowsPerBlock(), mcLeft.getColsPerBlock(), mcLeft.getRows(), mcLeft.getCols()));
				
				out = RDDAggregateUtils.mergeByKey(in1.union(in2));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			if( broadcastIn2 != null)
				sec.addLineageBroadcast(output.getName(), input2.getName());
			if(in2 != null) 
				sec.addLineageRDD(output.getName(), input2.getName());
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingSPInstruction.");		
	}
		
	/**
	 * 
	 * @param mcOut
	 * @throws DMLRuntimeException
	 */
	private static void checkValidOutputDimensions(MatrixCharacteristics mcOut) 
		throws DMLRuntimeException
	{
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("MatrixIndexingSPInstruction: The updated output dimensions are invalid: " + mcOut);
		}
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param ixrange
	 * @return
	 */
	private boolean isPartitioningPreservingRightIndexing(MatrixCharacteristics mcIn, IndexRange ixrange)
	{
		return ( mcIn.dimsKnown() &&
				(ixrange.rowStart==1 && ixrange.rowEnd==mcIn.getRows() && mcIn.getCols()<=mcIn.getColsPerBlock() )   //1-1 column block indexing
			  ||(ixrange.colStart==1 && ixrange.colEnd==mcIn.getCols() && mcIn.getRows()<=mcIn.getRowsPerBlock() )); //1-1 row block indexing
	}
	
	
	/**
	 * 
	 */
	private static class SliceRHSForLeftIndexing implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5724800998701216440L;
		
		private long rl; 
		private long cl; 
		private int brlen; 
		private int bclen;
		private long lhs_rlen;
		private long lhs_clen;
		
		public SliceRHSForLeftIndexing(long rl, long cl, int brlen, int bclen, long lhs_rlen, long lhs_clen) {
			this.rl = rl;
			this.cl = cl;
			this.brlen = brlen;
			this.bclen = bclen;
			this.lhs_rlen = lhs_rlen;
			this.lhs_clen = lhs_clen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> rightKV) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
	
			long start_lhs_globalRowIndex = rl + (rightKV._1.getRowIndex()-1)*brlen;
			long start_lhs_globalColIndex = cl + (rightKV._1.getColumnIndex()-1)*bclen;
			long end_lhs_globalRowIndex = start_lhs_globalRowIndex + rightKV._2.getNumRows() - 1;
			long end_lhs_globalColIndex = start_lhs_globalColIndex + rightKV._2.getNumColumns() - 1;
			
			long start_lhs_rowIndex = UtilFunctions.blockIndexCalculation(start_lhs_globalRowIndex, brlen);
			long end_lhs_rowIndex = UtilFunctions.blockIndexCalculation(end_lhs_globalRowIndex, brlen);
			long start_lhs_colIndex = UtilFunctions.blockIndexCalculation(start_lhs_globalColIndex, bclen);
			long end_lhs_colIndex = UtilFunctions.blockIndexCalculation(end_lhs_globalColIndex, bclen);
			
			for(long leftRowIndex = start_lhs_rowIndex; leftRowIndex <= end_lhs_rowIndex; leftRowIndex++) {
				for(long leftColIndex = start_lhs_colIndex; leftColIndex <= end_lhs_colIndex; leftColIndex++) {
					
					// Calculate global index of right hand side block
					long lhs_rl = Math.max((leftRowIndex-1)*brlen+1, start_lhs_globalRowIndex);
					long lhs_ru = Math.min(leftRowIndex*brlen, end_lhs_globalRowIndex);
					long lhs_cl = Math.max((leftColIndex-1)*bclen+1, start_lhs_globalColIndex);
					long lhs_cu = Math.min(leftColIndex*bclen, end_lhs_globalColIndex);
					
					int lhs_lrl = UtilFunctions.cellInBlockCalculation(lhs_rl, brlen);
					int lhs_lru = UtilFunctions.cellInBlockCalculation(lhs_ru, brlen);
					int lhs_lcl = UtilFunctions.cellInBlockCalculation(lhs_cl, bclen);
					int lhs_lcu = UtilFunctions.cellInBlockCalculation(lhs_cu, bclen);
					
					long rhs_rl = lhs_rl - rl + 1;
					long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
					long rhs_cl = lhs_cl - cl + 1;
					long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
					
					int rhs_lrl = UtilFunctions.cellInBlockCalculation(rhs_rl, brlen);
					int rhs_lru = UtilFunctions.cellInBlockCalculation(rhs_ru, brlen);
					int rhs_lcl = UtilFunctions.cellInBlockCalculation(rhs_cl, bclen);
					int rhs_lcu = UtilFunctions.cellInBlockCalculation(rhs_cu, bclen);
					
					MatrixBlock slicedRHSBlk = rightKV._2.sliceOperations(rhs_lrl, rhs_lru, rhs_lcl, rhs_lcu, new MatrixBlock());
					
					int lbrlen = UtilFunctions.computeBlockSize(lhs_rlen, leftRowIndex, brlen);
					int lbclen = UtilFunctions.computeBlockSize(lhs_clen, leftColIndex, bclen);
					MatrixBlock resultBlock = new MatrixBlock(lbrlen, lbclen, false);
					resultBlock = resultBlock.leftIndexingOperations(slicedRHSBlk, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, null, false);
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(leftRowIndex, leftColIndex), resultBlock));
				}
			}
			return retVal;
		}
		
	}
	
	/**
	 * 
	 */
	private static class ZeroOutLHS implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -3581795160948484261L;
		
		private boolean complementary = false;
		private int brlen; int bclen;
		private IndexRange indexRange;
		private long rl; long ru; long cl; long cu;
		
		public ZeroOutLHS(boolean complementary, int brlen, int bclen, long rl, long ru, long cl, long cu) {
			this.complementary = complementary;
			this.brlen = brlen;
			this.bclen = bclen;
			this.rl = rl;
			this.ru = ru;
			this.cl = cl;
			this.cu = cu;
			this.indexRange = new IndexRange(rl, ru, cl, cu);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{
			if( !UtilFunctions.isInBlockRange(kv._1(), brlen, bclen, rl, ru, cl, cu) ) {
				return kv;
			}
			
			IndexRange range = UtilFunctions.getSelectedRangeForZeroOut(new IndexedMatrixValue(kv._1, kv._2), brlen, bclen, indexRange);
			if(range.rowStart == -1 && range.rowEnd == -1 && range.colStart == -1 && range.colEnd == -1) {
				throw new Exception("Error while getting range for zero-out");
			}
			
			MatrixBlock zeroBlk = (MatrixBlock) kv._2.zeroOutOperations(new MatrixBlock(), range, complementary);
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, zeroBlk);
		}
		
	}
	
	/**
	 * 
	 */
	private static class LeftIndexPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1757075506076838258L;
		
		private PartitionedBroadcastMatrix _binput;
		private IndexRange _ixrange;
		private int _brlen;
		private int _bclen;
		
		
		public LeftIndexPartitionFunction(PartitionedBroadcastMatrix binput, IndexRange ixrange, MatrixCharacteristics mc) 
		{
			_binput = binput;
			_ixrange = ixrange;
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new LeftIndexPartitionIterator(arg0);
		}
		
		/**
		 * 
		 */
		private class LeftIndexPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public LeftIndexPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}
			
			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) 
				throws Exception 
			{
				if(!UtilFunctions.isInBlockRange(arg._1(), _brlen, _bclen, _ixrange)) {
					return arg;
				}
				
				// Calculate global index of left hand side block
				long lhs_rl = Math.max(_ixrange.rowStart, (arg._1.getRowIndex()-1)*_brlen + 1);
				long lhs_ru = Math.min(_ixrange.rowEnd, arg._1.getRowIndex()*_brlen);
				long lhs_cl = Math.max(_ixrange.colStart, (arg._1.getColumnIndex()-1)*_bclen + 1);
				long lhs_cu = Math.min(_ixrange.colEnd, arg._1.getColumnIndex()*_bclen);
				
				// Calculate global index of right hand side block
				long rhs_rl = lhs_rl - _ixrange.rowStart + 1;
				long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
				long rhs_cl = lhs_cl - _ixrange.colStart + 1;
				long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
				
				// Provide global zero-based index to sliceOperations
				MatrixBlock slicedRHSMatBlock = _binput.sliceOperations(rhs_rl, rhs_ru, rhs_cl, rhs_cu, new MatrixBlock());
				
				// Provide local zero-based index to leftIndexingOperations
				int lhs_lrl = UtilFunctions.cellInBlockCalculation(lhs_rl, _brlen);
				int lhs_lru = UtilFunctions.cellInBlockCalculation(lhs_ru, _brlen);
				int lhs_lcl = UtilFunctions.cellInBlockCalculation(lhs_cl, _bclen);
				int lhs_lcu = UtilFunctions.cellInBlockCalculation(lhs_cu, _bclen);
				MatrixBlock ret = arg._2.leftIndexingOperations(slicedRHSMatBlock, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, new MatrixBlock(), false);
				return new Tuple2<MatrixIndexes, MatrixBlock>(arg._1, ret);
			}
		}
	}

	/**
	 * 
	 */
	private static class SliceBlock implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5733886476413136826L;
		
		private IndexRange _ixrange;
		private int _brlen; 
		private int _bclen;
		
		public SliceBlock(IndexRange ixrange, MatrixCharacteristics mcOut) {
			_ixrange = ixrange;
			_brlen = mcOut.getRowsPerBlock();
			_bclen = mcOut.getColsPerBlock();
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{	
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(kv);
			
			ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
			OperationsOnMatrixValues.performSlice(in, _ixrange, _brlen, _bclen, outlist);
			
			return SparkUtils.fromIndexedMatrixBlock(outlist);
		}		
	}
	
	/**
	 * 
	 */
	private static class SliceBlockPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8111291718258309968L;
		
		private IndexRange _ixrange;
		private int _brlen; 
		private int _bclen;
		
		public SliceBlockPartitionFunction(IndexRange ixrange, MatrixCharacteristics mcOut) {
			_ixrange = ixrange;
			_brlen = mcOut.getRowsPerBlock();
			_bclen = mcOut.getColsPerBlock();
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new SliceBlockPartitionIterator(arg0);
		}	
		
		private class SliceBlockPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public SliceBlockPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg)
				throws Exception
			{
				IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg);
				
				ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
				OperationsOnMatrixValues.performSlice(in, _ixrange, _brlen, _bclen, outlist);
				
				assert(outlist.size() == 1); //1-1 row/column block indexing
				return SparkUtils.fromIndexedMatrixBlock(outlist.get(0));
			}			
		}
	}
}
