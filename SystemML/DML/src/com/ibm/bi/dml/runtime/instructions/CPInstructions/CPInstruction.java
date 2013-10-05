/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class CPInstruction extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum CPINSTRUCTION_TYPE { INVALID, AggregateBinary, AggregateUnary, ArithmeticBinary, Tertiary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, Reorg, RelationalBinary, File, Variable, External, ParameterizedBuiltin, Builtin, Append, Rand, Sort, MatrixIndexing, MMTSJ, MatrixReshape }; 
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

	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		throw new DMLRuntimeException ( "processInstruction(ProgramBlock): should not be invoked in the base class.");
	}

	public void processInstruction(ProgramBlock pb, ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException {
		throw new DMLRuntimeException ( "processInstruction(ProgramBlock): should not be invoked in the base class.");
	}

	@Override
	public String getGraphString() {
		return InstructionUtils.getOpCode(instString);
	}
	
}
