/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction.SPINSTRUCTION_TYPE;


public class InstructionParser 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public static Instruction parseSingleInstruction ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( str == null || str.isEmpty() )
			return null;
		
		String execType = str.split(Instruction.OPERAND_DELIM)[0]; 
		if (   execType.equalsIgnoreCase(ExecType.CP.toString()) 
			|| execType.equalsIgnoreCase(ExecType.CP_FILE.toString()) ) 
		{
			CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
			return CPInstructionParser.parseSingleInstruction (cptype, str);
		}
		else if (   execType.equalsIgnoreCase(ExecType.SPARK.toString()) ) 
		{
			SPINSTRUCTION_TYPE sptype = InstructionUtils.getSPType(str); 
			return SPInstructionParser.parseSingleInstruction (sptype, str);
		}
		else if ( execType.equalsIgnoreCase("MR") ) {
			MRINSTRUCTION_TYPE mrtype = InstructionUtils.getMRType(str); 
			if ( mrtype == null )
				throw new DMLRuntimeException("Can not determine MRType for instruction: " + str);
			return MRInstructionParser.parseSingleInstruction (mrtype, str);
		}
		else {
			throw new DMLRuntimeException("Unknown execution type in instruction: " + str);
		}
	}
	
	public static Instruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		String[] strlist = str.split(Instruction.INSTRUCTION_DELIM);
		Instruction[] inst = new Instruction[strlist.length];
		
		for ( int i=0; i < inst.length; i++ ) {
			inst[i] = parseSingleInstruction ( strlist[i] );
		}
		
		return inst;
	}
	
	public static void printInstructions(Instruction[] instructions)
	{
		for(Instruction ins: instructions)
			System.out.println(ins.toString());
	}

}
