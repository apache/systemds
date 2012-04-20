package dml.runtime.instructions.CPInstructions;

import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CPInstruction extends Instruction {

	public enum CPINSTRUCTION_TYPE { INVALID, AggregateBinary, AggregateUnary, ArithmeticBinary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, Reorg, RelationalBinary, File, Variable, External, ParameterizedBuiltin, Builtin, Append, Rand}; 
	CPINSTRUCTION_TYPE cptype;
	
	Operator optr;
	
	public CPInstruction(Operator op) {
		type = INSTRUCTION_TYPE.CONTROL_PROGRAM;
		optr = op;
	}
	
	public CPINSTRUCTION_TYPE getCPInstructionType() {
		return cptype;
	}
	
	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		return null;
	}

	public Data processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		throw new DMLRuntimeException ( "processInstruction(ProgramBlock): should not be invoked in the base class.");
	}

	@Override
	public String getGraphString() {
		return InstructionUtils.getOpCode(instString);
	}
	
}
