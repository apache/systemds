/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class CPInstruction extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum CPINSTRUCTION_TYPE { INVALID, AggregateBinary, AggregateUnary, ArithmeticBinary, Tertiary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, ParameterizedBuiltin, MultiReturnBuiltin, Builtin, Reorg, RelationalBinary, File, Variable, External, Append, Rand, Sort, MatrixIndexing, MMTSJ, PMMJ, MatrixReshape, Partition, StringInit }; 
	
	protected CPINSTRUCTION_TYPE cptype;
	protected Operator optr;
	
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

	/**
	 * This method should be used to execute the instruction. It's abstract to force 
	 * subclasses to override it. 
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public abstract void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;

	@Override
	public String getGraphString() {
		return InstructionUtils.getOpCode(instString);
	}
}
