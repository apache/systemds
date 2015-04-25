/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Not;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

public abstract class UnarySPInstruction extends ComputationSPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public UnarySPInstruction(Operator op, CPOperand in, CPOperand out,
			String opcode, String instr) {
		this (op, in, null, null, out, opcode, instr);
	}

	public UnarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String instr) {
		this (op, in1, in2, null, out, opcode, instr);
	}

	public UnarySPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String instr) {
		super(op, in1, in2, in3, out, opcode, instr);
	}

	static String parseUnaryInstruction(String instr, CPOperand in,
			CPOperand out) throws DMLRuntimeException {
		InstructionUtils.checkNumFields(instr, 2);
		return parse(instr, in, null, null, out);
	}

	static String parseUnaryInstruction(String instr, CPOperand in1,
			CPOperand in2, CPOperand out) throws DMLRuntimeException {
		InstructionUtils.checkNumFields(instr, 3);
		return parse(instr, in1, in2, null, out);
	}

	static String parseUnaryInstruction(String instr, CPOperand in1,
			CPOperand in2, CPOperand in3, CPOperand out) throws DMLRuntimeException {
		InstructionUtils.checkNumFields(instr, 4);
		return parse(instr, in1, in2, in3, out);
	}

	private static String parse(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) throws DMLRuntimeException {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		
		// first part is the opcode, last part is the output, middle parts are input operands
		String opcode = parts[0];
		out.split(parts[parts.length-1]);
		
		switch(parts.length) {
		case 3:
			in1.split(parts[1]);
			in2 = null;
			in3 = null;
			break;
		case 4:
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3 = null;
			break;
		case 5:
			in1.split(parts[1]);
			in2.split(parts[2]);
			in3.split(parts[3]);
			break;
		default:
			throw new DMLRuntimeException("Unexpected number of operands in the instruction: " + instr);
		}
		return opcode;
	}
	
	static SimpleOperator getSimpleUnaryOperator(String opcode)
			throws DMLRuntimeException {
		if (opcode.equalsIgnoreCase("!"))
			return new SimpleOperator(Not.getNotFnObject());

		throw new DMLRuntimeException("Unknown unary operator " + opcode);
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException 
	 */
	protected void updateOutputMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			if(!mc1.dimsKnown())
				throw new DMLRuntimeException("The output dimensions are not specified and cannot be inferred from input:" + mc1.toString() + " " + mcOut.toString());
			else
				mcOut.set(mc1.getRows(), mc1.getCols(), mc1.getColsPerBlock(), mc1.getRowsPerBlock());
		}
	}
}
