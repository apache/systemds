package dml.runtime.instructions;

import dml.runtime.functionobjects.Builtin;
import dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class InstructionUtils {


	public static int checkNumFields ( String str, int exp ) throws DMLRuntimeException {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		
		if ( parts.length - 1 != exp ) // -1 for opcode
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + exp + ") != is not equal to actual number(" + (parts.length-1) + ").");
		
		return parts.length - 1; 
	}
	
	public static String[] getInstructionParts ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length];
		ret[0] = parts[0];
		
		String[] f;
		for ( int i=1; i < parts.length; i++ ) {
			f = parts[i].split(Instruction.VALUETYPE_PREFIX);
			ret[i] = f[0];
		}
		return ret;
	}
	
	public static String[] getInstructionPartsWithValueType ( String str ) {
		return str.split(Instruction.OPERAND_DELIM);
	}
	
	public static String getOpCode ( String str ) {
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		return parts[0];
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
