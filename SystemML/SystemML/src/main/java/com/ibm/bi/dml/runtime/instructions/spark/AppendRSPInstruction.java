package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

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
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

public class AppendRSPInstruction extends BinarySPInstruction
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
	
	public AppendRSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.RAppend;
		
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
		
		
		if(!opcode.equalsIgnoreCase("rappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendRSPInstruction: " + str);
		else
			return new AppendRSPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if( _type == AppendType.CBIND )
		{		
			// reduce-only append (output must have at most one column block):
			// cogroup operation
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
			
			if(mc1.getRows() + mc2.getRows() > mc1.getRowsPerBlock()) {
				throw new DMLRuntimeException("In AppendRSPInstruction, output must have at most one column block"); 
			}
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.cogroup(in2).mapToPair(new ReduceSideAppend());
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			sec.addLineageRDD(output.getName(), input2.getName());
		}
		else //STRING
		{
			throw new DMLRuntimeException("Error: String append only valid as CPInstruction");
			// InstructionUtils.processStringAppendInstruction(ec, input1, input2, output);
		}		
	}
	
	public class ReduceSideAppend implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = -6763904972560309095L;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(
				Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> kv)
				throws Exception {
			MatrixIndexes indx = kv._1;
			MatrixBlock left = null; MatrixBlock right = null;
			for(MatrixBlock blk : kv._2._1) {
				if(left != null) {
					throw new Exception("Only 1 block in ReduceSideAppend on left");
				}
				left = blk;
			}
			for(MatrixBlock blk : kv._2._2) {
				if(right != null) {
					throw new Exception("Only 1 block in ReduceSideAppend on right");
				}
				right = blk;
			}
			if(left == null || right == null) {
				throw new Exception("Only 1 block in ReduceSideAppend on left or right");
			}
			return new Tuple2<MatrixIndexes, MatrixBlock>(indx, left.appendOperations(right, new MatrixBlock()));
		}
		
	}
}

