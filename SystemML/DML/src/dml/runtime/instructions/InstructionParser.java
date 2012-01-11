package dml.runtime.instructions;

import dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class InstructionParser {
	
	public static Instruction parseSingleInstruction ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( str == null || str.isEmpty() )
			return null;
		
		// It is responsibility of the caller to know whether "str" is a MR instruction or a CP instruction
		
		MRINSTRUCTION_TYPE mrtype = InstructionUtils.getMRType(str); 
		CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
		
		if ( mrtype != null ) {
			return MRInstructionParser.parseSingleInstruction (mrtype, str);
		}
		else if ( cptype != null ) {
			return CPInstructionParser.parseSingleInstruction (cptype, str);
		}
		else {
			throw new DMLRuntimeException("Encountered unknown instruction type while parsing \"" + str + "\".");
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
