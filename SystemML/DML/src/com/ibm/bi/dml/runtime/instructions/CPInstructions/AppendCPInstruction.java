package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class AppendCPInstruction extends BinaryCPInstruction{
	//can be a var name or constant
	String offset_str;
	
	public AppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, String offset_str, CPOperand out, String istr){
		super(op, in1, in2, out, istr);
		this.offset_str = offset_str;
		cptype = CPINSTRUCTION_TYPE.Append;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		String offset_str = parts[4];
		
		if(!opcode.equalsIgnoreCase("append"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendCPInstruction: " + str);
		else
			return new AppendCPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, 
										   in2,
										   offset_str,
										   out,
										   str);
	}
	
	@Override
	public void processInstruction(ProgramBlock pb)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get inputs
		MatrixBlock matBlock1 = pb.getMatrixInput(input1.get_name());
		MatrixBlock matBlock2 = pb.getMatrixInput(input2.get_name());
		
		//check input dimensions
		if(matBlock1.getNumRows() != matBlock2.getNumRows())
			throw new DMLRuntimeException("Append is not possible for input matrices " 
										  + input1.get_name()
										  + " and "
										  + input2.get_name()
										  + "with unequal number of rows");
		
		//create output properties
		ReorgOperator r_op = (ReorgOperator) optr;
		OffsetColumnIndex off = ((OffsetColumnIndex)((ReorgOperator)optr).fn);
		off.setOutputSize(matBlock1.getNumRows(), matBlock1.getNumColumns() + matBlock2.getNumColumns());
		
		//execute append operations (append both inputs to initially empty output)
		off.setOffset(0);
		MatrixBlock tmp = (MatrixBlock) matBlock1.appendOperations(r_op, new MatrixBlock(), 0, 0, 0 );
		off.setOffset(matBlock1.getNumColumns());
		MatrixBlock ret = (MatrixBlock) matBlock2.appendOperations(r_op, tmp, 0, 0, 0 );
		
		//set output
		String output_name = output.get_name();
		pb.setMatrixOutput(output_name, ret);

		//release inputs 
        pb.releaseMatrixInput(input1.get_name());
		pb.releaseMatrixInput(input2.get_name());
	
	}
}
