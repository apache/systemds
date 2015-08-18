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

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class AppendGSPInstruction extends BinarySPInstruction
{
	
	//TODO cleanup: use offsets and sp/mr consistency
	//private CPOperand _offset = null;
	//private CPOperand _offset2 = null;
	
	public AppendGSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand offset, CPOperand offset2, CPOperand out, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.GAppend;
		
		//_offset = offset;
		//_offset2 = offset2;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and 2xoffset = 5
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand out = new CPOperand(parts[5]);
		
		if(!opcode.equalsIgnoreCase("gappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendGSPInstruction: " + str);
		else
			return new AppendGSPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, in4, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		// general case append (map-extend, aggregate)
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
		
		if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
			throw new DMLRuntimeException("The dimensions unknown for inputs");
		}
		else if(mc1.getRows() != mc2.getRows()) {
			throw new DMLRuntimeException("The number of rows of inputs should match for append instruction");
		}
		else if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
			throw new DMLRuntimeException("The block sizes donot match for input matrices");
		}
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		
		if(mc1.getCols() % mc1.getColsPerBlock() == 0) {
			throw new DMLRuntimeException("Incorrect append instruction when mc1.getCols() % mc1.getColsPerBlock() == 0. Should have used AppendGAlignedSP");
		}
		else {
			// General case: This one needs shifting and merging and hence has huge performance hit.
			long indexOfLastColumn = (long) Math.ceil( (double)mc1.getCols() / (double)mc1.getColsPerBlock() );
			int widthOfLastColumn = UtilFunctions.computeBlockSize(mc1.getCols(), indexOfLastColumn, mc1.getColsPerBlock());
			JavaPairRDD<MatrixIndexes,MatrixBlock> shifted_in2 = 
					in2.flatMapToPair(new ShiftMatrix(widthOfLastColumn, mc1.getCols(), mc1.getColsPerBlock(), mc1.getCols() + mc2.getCols()));
			out = in1.cogroup(shifted_in2).mapToPair(new MergeWithShiftedBlocks(mc1.getCols(), mc1.getColsPerBlock(), mc1.getCols() + mc2.getCols()));
		}
		
		//put output RDD handle into symbol table
		updateBinaryAppendOutputMatrixCharacteristics(sec);
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		sec.addLineageRDD(output.getName(), input2.getName());
	}
	
	public static class MergeWithShiftedBlocks implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 848955582909209400L;
		
		long lastColumnOfLeft; long clenOutput; int bclen;
		
		public MergeWithShiftedBlocks(long clenLeft, int bclen, long clenOutput) {
			lastColumnOfLeft = (long) Math.ceil((double)clenLeft / (double)bclen);
			this.clenOutput = clenOutput;
			this.bclen = bclen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> kv)
				throws Exception {
			MatrixBlock firstBlk = null;
			MatrixBlock secondBlk = null;
			for(MatrixBlock blk : kv._2._1) {
				// Atmost one block from LHS matrix 
				if(firstBlk == null) {
					firstBlk = blk;
				}
				else throw new Exception("Expected exactly one block in MergeWithShiftedBlocks");
			}
			for(MatrixBlock blk : kv._2._2) {
				// Atmost two blocks from shifted RHS matrix
				// However, if one block received from LHS, then atmost one block from shifted RHS matrix
				if(firstBlk == null) {
					firstBlk = blk;
				}
				else if(secondBlk == null) {
					secondBlk = blk;
				}
				else throw new Exception("Expected atmost 2 blocks in MergeWithShiftedBlocks");
			}
			
			int lclen = UtilFunctions.computeBlockSize(clenOutput, kv._1.getColumnIndex(), bclen);
			
			if(firstBlk != null && secondBlk == null) {
				if(firstBlk.getNumColumns() != lclen) {
					// Sanity check
					throw new DMLRuntimeException("Incorrect dimensions of the input block while merging shifted blocks:" + firstBlk.getNumColumns() + " != " +  lclen);
				}
				return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, firstBlk);
			}
			else if(firstBlk == null && secondBlk != null) {
				if(secondBlk.getNumColumns() != lclen) {
					// Sanity check
					throw new DMLRuntimeException("Incorrect dimensions of the input block while merging shifted blocks:" + secondBlk.getNumColumns() + " != " +  lclen);
				}
				return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, secondBlk);
			}
			else if(firstBlk == null || secondBlk == null) {
				throw new Exception("Expected exactly one block in MergeWithShiftedBlocks");
			}
			
			// Since merge requires the dimensions matching
			if(kv._1.getColumnIndex() == lastColumnOfLeft && firstBlk.getNumColumns() < secondBlk.getNumColumns()) {
				// This case occurs for last block of LHS matrix
				MatrixBlock replaceFirstBlk = new MatrixBlock(firstBlk.getNumRows(), secondBlk.getNumColumns(), true);
				replaceFirstBlk = replaceFirstBlk.leftIndexingOperations(firstBlk, 0, firstBlk.getNumRows()-1, 0, firstBlk.getNumColumns()-1, new MatrixBlock(), true);
				firstBlk = replaceFirstBlk;
			}
			
			//merge with sort since blocks might be in any order
			firstBlk.merge(secondBlk, false);
			if(firstBlk.getNumColumns() != lclen) {
				// Sanity check
				throw new DMLRuntimeException("Incorrect dimensions of the input block while merging shifted blocks:" + firstBlk.getNumColumns() + " != " +  lclen);
			}
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, firstBlk);
		}
		
	}
	
	public static class ShiftMatrix implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {

		private static final long serialVersionUID = 3524189212798209172L;
		
		int shiftBy; // width of last column
		long startColumnIndex; // number of columns in left matrix
		int bclen;
		long clenOutput;
		
		public ShiftMatrix(int shiftBy, long startColumnIndex, int bclen, long clenOutput) throws DMLRuntimeException {
			this.shiftBy = shiftBy;
			this.startColumnIndex = startColumnIndex;
			if(shiftBy < 1) {
				throw new DMLRuntimeException("ShiftMatrix is applicable only for shiftBy < 1");
			}
			this.bclen = bclen;
			this.clenOutput = clenOutput;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			long columnIndex = (long) (kv._1.getColumnIndex() + Math.ceil( (double)startColumnIndex / (double)bclen)  - 1);
			MatrixIndexes firstIndex = new MatrixIndexes(kv._1.getRowIndex(), columnIndex);
			MatrixIndexes secondIndex = new MatrixIndexes(kv._1.getRowIndex(), columnIndex + 1);
			
			int cutAt = bclen - shiftBy;
			
			int lclen1 = UtilFunctions.computeBlockSize(clenOutput, firstIndex.getColumnIndex(), bclen);
			if(cutAt >= kv._2.getNumColumns()) {
				// The block is too small to be cut
				MatrixBlock firstBlk = new MatrixBlock(kv._2.getNumRows(), lclen1, true);
				firstBlk = firstBlk.leftIndexingOperations(kv._2, 0, kv._2.getNumRows()-1, lclen1-kv._2.getNumColumns(), lclen1-1, new MatrixBlock(), true);
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(firstIndex, firstBlk));
			}
			else {
				// Since merge requires the dimensions matching, shifting = slicing + left indexing
				MatrixBlock firstSlicedBlk = kv._2.sliceOperations(0, kv._2.getNumRows()-1, 0, cutAt-1, new MatrixBlock());
				MatrixBlock firstBlk = new MatrixBlock(kv._2.getNumRows(), lclen1, true);
				firstBlk = firstBlk.leftIndexingOperations(firstSlicedBlk, 0, kv._2.getNumRows()-1, shiftBy, bclen-1, new MatrixBlock(), true);
				
				MatrixBlock secondSlicedBlk = kv._2.sliceOperations(0, kv._2.getNumRows()-1, cutAt, kv._2.getNumColumns()-1, new MatrixBlock());
				int lclen2 = UtilFunctions.computeBlockSize(clenOutput, secondIndex.getColumnIndex(), bclen);
				MatrixBlock secondBlk = new MatrixBlock(kv._2.getNumRows(), lclen2, true);
				secondBlk = secondBlk.leftIndexingOperations(secondSlicedBlk, 0, kv._2.getNumRows()-1, 0, secondSlicedBlk.getNumColumns()-1, new MatrixBlock(), true);
				
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(firstIndex, firstBlk));
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(secondIndex, secondBlk));
			}
			return retVal;
		}
		
	}
}