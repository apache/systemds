package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class CPInstruction extends Instruction {

	public enum CPINSTRUCTION_TYPE { INVALID, AggregateBinary, AggregateUnary, ArithmeticBinary, Tertiary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, Reorg, RelationalBinary, File, Variable, External, ParameterizedBuiltin, Builtin, Append, Rand, Sort, MatrixIndexing }; 
	CPINSTRUCTION_TYPE cptype;
	
	Operator optr;
	
	public CPInstruction() {
		type = INSTRUCTION_TYPE.CONTROL_PROGRAM;
	}
	public CPInstruction(Operator op) {
		this();
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

	public void processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		throw new DMLRuntimeException ( "processInstruction(ProgramBlock): should not be invoked in the base class.");
	}

	@Override
	public String getGraphString() {
		return InstructionUtils.getOpCode(instString);
	}
	
}
