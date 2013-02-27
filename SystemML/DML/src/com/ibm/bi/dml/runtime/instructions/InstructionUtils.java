package com.ibm.bi.dml.runtime.instructions;

import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class InstructionUtils {


	public static int checkNumFields ( String str, int expected ) throws DMLRuntimeException {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		
		if ( parts.length - 2 != expected ) // -2 accounts for execType and opcode
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected + ") != is not equal to actual number(" + (parts.length-2) + ").");
		
		return parts.length - 2; 
	}
	
	/**
	 * Given an instruction string, strip-off the execution type and return 
	 * opcode and all input/output operands WITHOUT their data/value type. 
	 * i.e., ret.length = parts.length-1 (-1 for execution type)
	 * 
	 * @param str
	 * @return 
	 */
	public static String[] getInstructionParts ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length-1]; // stripping-off the exectype
		
		ret[0] = parts[1]; // opcode
		for ( int i=2; i < parts.length; i++ ) {
			ret[i-1] = parts[i].split(Instruction.DATATYPE_PREFIX)[0];
		}
		return ret;
	}
	
	/**
	 * Given an instruction string, this function strips-off the 
	 * execution type (CP or MR) and returns the remaining parts, 
	 * which include the opcode as well as the input and output operands.
	 * Each returned part will have the datatype and valuetype associated
	 * with the operand.
	 * 
	 * This function is invoked mainly for parsing CPInstructions.
	 * 
	 * @param str
	 * @return
	 */
	public static String[] getInstructionPartsWithValueType ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length-1]; // stripping-off the exectype
		
		ret[0] = parts[1]; // opcode
		for ( int i=2; i < parts.length; i++ ) {
			ret[i-1] = parts[i];
		}
		return ret;
	}
	
	public static String getOpCode ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		return parts[1]; // parts[0] is the execution type
	}
	
	
	public static MRINSTRUCTION_TYPE getMRType ( String str ) throws DMLUnsupportedOperationException {
		String opcode = getOpCode(str);
		MRINSTRUCTION_TYPE mrtype = MRInstructionParser.String2MRInstructionType.get ( opcode ); 
		return mrtype;
	}
	
	public static CPINSTRUCTION_TYPE getCPType ( String str ) throws DMLUnsupportedOperationException {
		String opcode = getOpCode(str);
		CPINSTRUCTION_TYPE cptype = CPInstructionParser.String2CPInstructionType.get ( opcode ); 
		return cptype;
	}
	
	public static boolean isBuiltinFunction ( String opcode ) {
		Builtin.BuiltinFunctionCode bfc = Builtin.String2BuiltinFunctionCode.get(opcode);
		return (bfc != null);
	}
	
	public static boolean isOperand(String str) {
		String[] parts = str.split(Instruction.DATATYPE_PREFIX);
		if (parts.length > 1){
			return true;
		}
		else {
			return false;
		}
	}
	
	public static boolean isDistributedCacheUsed(String str) {
		for(String inst : str.split(Instruction.INSTRUCTION_DELIM)) {
			String opcode = InstructionUtils.getOpCode(inst);
			if(opcode.equalsIgnoreCase("mvmult") || opcode.equalsIgnoreCase("append")) // || opcode.equalsIgnoreCase("vmmult"))
				return true;
		}
		return false;
	}
	
}
