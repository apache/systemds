/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class BinaryCPInstruction extends ComputationCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public BinaryCPInstruction(Operator op, 
							 CPOperand in1, 
							 CPOperand in2, 
							 CPOperand out, 
							 String opcode,
						     String istr ){
		super(op, in1, in2, out, opcode, istr);
	}

	public BinaryCPInstruction(Operator op, 
			 CPOperand in1, 
			 CPOperand in2, 
			 CPOperand in3, 
			 CPOperand out, 
			 String opcode,
		     String istr ){
		super(op, in1, in2, in3, out, opcode, istr);
	}

	static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out)
		throws DMLRuntimeException{
		
		InstructionUtils.checkNumFields ( instr, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		out.split(parts[3]);
		
		return opcode;
	}
	
	static String parseBinaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out)
	throws DMLRuntimeException{
	
		InstructionUtils.checkNumFields ( instr, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		
		return opcode;
	}
	
}
