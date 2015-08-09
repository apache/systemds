package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

public class AppendMSPInstruction extends BinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum AppendType{
		CBIND,
		STRING,
	}
	
	//type (matrix cbind / scalar string concatenation)
	private AppendType _type;
	
	public AppendMSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.MAppend;
		
		_type = type;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		//String offset_str = parts[4];
		 
		AppendType type = (in1.getDataType()==DataType.MATRIX) ? AppendType.CBIND : AppendType.STRING;
		
		
		if(!opcode.equalsIgnoreCase("mappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendMSPInstruction: " + str);
		else
			return new AppendMSPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if( _type == AppendType.CBIND )
		{		
			// map-only append (rhs must be vector and fit in mapper mem)
			SparkExecutionContext sec = (SparkExecutionContext)ec;
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			
			if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
				throw new DMLRuntimeException("The dimensions unknown for inputs");
			}
			else if(mc1.getRows() != mc2.getRows()) {
				throw new DMLRuntimeException("The number of rows of inputs should match for append instruction");
			}
			else if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
				throw new DMLRuntimeException("The block sizes donot match for input matrices");
			}
			
			if(!mcOut.dimsKnown()) {
				mcOut.set(mc1.getRows(), mc1.getCols() + mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			Broadcast<PartitionedMatrixBlock> in2 = sec.getBroadcastForVariable( input2.getName() );
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.flatMapToPair(
					new MapSideAppend(in2, mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock()));
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			sec.addLineageBroadcast(output.getName(), input2.getName());
		}
		else //STRING
		{
			throw new DMLRuntimeException("Error: String append only valid as CPInstruction");
			// InstructionUtils.processStringAppendInstruction(ec, input1, input2, output);
		}		
	}
	
	public static class MapSideAppend implements  PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 2738541014432173450L;
		private Broadcast<PartitionedMatrixBlock> pm = null;
		long left_rlen; long left_clen; int left_brlen; int left_bclen;
		//long right_rlen; long right_clen; int right_brlen; int right_bclen;
		long rightMostBlockIndex;
		
		public MapSideAppend(Broadcast<PartitionedMatrixBlock> binput, 
				long left_rlen, long left_clen, int left_brlen, int left_bclen)  {
			pm = binput;
			this.left_rlen = left_rlen; this.left_clen = left_clen;
			this.left_brlen = left_brlen; this.left_bclen = left_bclen;
			
			rightMostBlockIndex = (long) Math.ceil( (double) left_clen / (double) left_bclen);
			
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			if(kv._1.getColumnIndex() == rightMostBlockIndex) {
				MatrixBlock lhsMatrixBlock = kv._2;
				PartitionedMatrixBlock partitionedMat = pm.value();
				
				if(lhsMatrixBlock.getNumColumns() == left_bclen) {
					// No need to perform append, just add an additional blocks in RHS.
					retVal.add(kv);
					
					long colIndex = kv._1.getColumnIndex() + 1;
					for(int i = 1; i <= partitionedMat.getNumColumnBlocks(); i++) {
						MatrixBlock part = partitionedMat.getMatrixBlock((int)kv._1.getRowIndex(), i);
						retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(
								new MatrixIndexes(kv._1.getRowIndex(), colIndex), part));
						colIndex++;
					}
				}
				else {
					// Unaligned matrix blocks
					// First get the entire RHS Matrix
					MatrixBlock rhsMatBlock = null;
					if(partitionedMat.getNumColumnBlocks() == 1) {
						// Avoids an unnecessary left indexing operation.
						rhsMatBlock = partitionedMat.getMatrixBlock((int)kv._1.getRowIndex(), 1);
					}
					else {
						rhsMatBlock = new MatrixBlock(lhsMatrixBlock.getNumRows(), (int)partitionedMat.getNumCols(), false);
						long colLower = 1;
						long colUpper = 0;
						for(int i = 1; i <= partitionedMat.getNumColumnBlocks(); i++) {
							MatrixBlock part = partitionedMat.getMatrixBlock((int)kv._1.getRowIndex(), i);
							colUpper += part.getNumColumns();
							rhsMatBlock = rhsMatBlock.leftIndexingOperations(
									part, 1, lhsMatrixBlock.getNumRows(), colLower, colUpper, 
									null, true);
							colLower = colUpper + 1;
						}
					}
					
					// Then perform append operations 
					MatrixBlock retBlk = lhsMatrixBlock.appendOperations(rhsMatBlock, new MatrixBlock());
					
					// Output the appended blocks
					if(retBlk.getNumColumns() > left_bclen) {
						// Since retBlk is greater than left_bclen, slice it
						long numBlocks = (long) Math.ceil((double)retBlk.getNumColumns() / left_bclen);
						long colLower = 1;
						long colUpper = 0;
						long colIndex = kv._1.getColumnIndex();
						for(int i = 1; i <= numBlocks; i++) {
							colUpper = Math.min(colUpper + left_bclen, retBlk.getNumColumns());
							// Slice blocks so as to get them of size "left_bclen" 
							MatrixBlock blk = retBlk.sliceOperations(1, retBlk.getNumRows(), colLower, colUpper, new MatrixBlock());
							colLower = colUpper + 1;
							retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(
									new MatrixIndexes(kv._1.getRowIndex(), colIndex), blk));
							colIndex++;
						}
					}
					else {
						// appended block has only 1 block
						retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, retBlk));
					}
				}
			}
			else if(kv._1.getColumnIndex() > rightMostBlockIndex) {
				throw new Exception("Incorrect computation of rightMostBlockIndex:" + kv._1.getColumnIndex() + " " + rightMostBlockIndex);
			}
			else {
				// No need to perform append operation on this block as it is not rightMostBlockIndex
				retVal.add(kv); 
			}
			return retVal;
		}
		
	}
}
