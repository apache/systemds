package dml.runtime.instructions.CPInstructions;

import dml.runtime.functionobjects.Not;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SimpleOperator;
import dml.utils.DMLRuntimeException;

public class UnaryCPInstruction extends ComputationCPInstruction{
	public UnaryCPInstruction(Operator op,
							  CPOperand in,
							  CPOperand out,
							  String instr){
		super(op, in, null, out);
		instString = instr;
	}
	
	static String parseUnaryInstruction(String instr, CPOperand in, CPOperand out)
		throws DMLRuntimeException{
	
		InstructionUtils.checkNumFields ( instr, 2 );
	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in.split(parts[1]);
		out.split(parts[2]);
		
		return opcode;
	}

	static SimpleOperator getSimpleUnaryOperator(String opcode) throws DMLRuntimeException{
		if(opcode.equalsIgnoreCase("!"))
			return new SimpleOperator(Not.getNotFnObject());
		
		throw new DMLRuntimeException("Unknown unary operator " + opcode);
	}
}
