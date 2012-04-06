package dml.runtime.instructions;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class InstructionUtils {


	public static int checkNumFields ( String str, int expected ) throws DMLRuntimeException {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		
		if ( parts.length - 2 != expected ) // -2 accounts for execType and opcode
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected + ") != is not equal to actual number(" + (parts.length-2) + ").");
		
		return parts.length - 2; 
	}
	
	/**
	 * Parse the instruction string, and return opcode as well as all operands
	 * i.e., ret.length = parts.length-1 (-1 accounts for execution type)
	 * 
	 * @param str
	 * @return 
	 */
	public static String[] getInstructionParts ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length-1]; // -1 accounts for exec type
		
		ret[0] = parts[1]; // opcode
		String[] f;
		for ( int i=2; i < parts.length; i++ ) {
			f = parts[i].split(Instruction.VALUETYPE_PREFIX);
			ret[i-1] = f[0];
		}
		return ret;
	}
	
	public static String[] getInstructionPartsWithValueType ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length-1];
		
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
	
}
