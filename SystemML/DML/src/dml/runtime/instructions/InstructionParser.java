package dml.runtime.instructions;

import dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class InstructionParser {
	
	public static Instruction parseSingleInstruction ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( str == null || str.isEmpty() )
			return null;
		
		String execType = str.split(Instruction.OPERAND_DELIM)[0]; 
		if ( execType.equalsIgnoreCase("CP") ) {
			CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
			return CPInstructionParser.parseSingleInstruction (cptype, str);
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
