package dml.runtime.instructions.CPInstructions;

import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CPInstruction extends Instruction {

	public enum CPINSTRUCTION_TYPE { INVALID, Scalar, Arithmetic, Boolean, Relational, Builtin, File, Variable, External, ParameterizedBuiltin }; 
	
	CPINSTRUCTION_TYPE cptype;
	
	Operator optr;
	
	public CPInstruction(Operator op) {
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
