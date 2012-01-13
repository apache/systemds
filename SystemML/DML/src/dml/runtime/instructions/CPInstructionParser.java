package dml.runtime.instructions;

import java.util.HashMap;

import dml.runtime.instructions.CPInstructions.ArithmeticCPInstruction;
import dml.runtime.instructions.CPInstructions.BooleanCPInstruction;
import dml.runtime.instructions.CPInstructions.BuiltinCPInstruction;
import dml.runtime.instructions.CPInstructions.CPInstruction;
import dml.runtime.instructions.CPInstructions.FileCPInstruction;
import dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import dml.runtime.instructions.CPInstructions.ParameterizedBuiltinCPInstruction;
import dml.runtime.instructions.CPInstructions.RelationalCPInstruction;
import dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class CPInstructionParser extends InstructionParser {

	static public HashMap<String, CPINSTRUCTION_TYPE> String2CPInstructionType;
	static {
		String2CPInstructionType = new HashMap<String, CPINSTRUCTION_TYPE>();

		// Arithmetic Instruction Opcodes 
		String2CPInstructionType.put( "+"    , CPINSTRUCTION_TYPE.Arithmetic);
		String2CPInstructionType.put( "-"    , CPINSTRUCTION_TYPE.Arithmetic);
		String2CPInstructionType.put( "*"    , CPINSTRUCTION_TYPE.Arithmetic);
		String2CPInstructionType.put( "/"    , CPINSTRUCTION_TYPE.Arithmetic);
		String2CPInstructionType.put( "o/"   , CPINSTRUCTION_TYPE.Arithmetic);
		String2CPInstructionType.put( "^"    , CPINSTRUCTION_TYPE.Arithmetic);

		// Boolean Instruction Opcodes 
		String2CPInstructionType.put( "&&"   , CPINSTRUCTION_TYPE.Boolean);
		String2CPInstructionType.put( "||"   , CPINSTRUCTION_TYPE.Boolean);
		String2CPInstructionType.put( "!"    , CPINSTRUCTION_TYPE.Boolean);

		// Relational Instruction Opcodes 
		String2CPInstructionType.put( "=="   , CPINSTRUCTION_TYPE.Relational);
		String2CPInstructionType.put( "!="   , CPINSTRUCTION_TYPE.Relational);
		String2CPInstructionType.put( "<"    , CPINSTRUCTION_TYPE.Relational);
		String2CPInstructionType.put( ">"    , CPINSTRUCTION_TYPE.Relational);
		String2CPInstructionType.put( "<="   , CPINSTRUCTION_TYPE.Relational);
		String2CPInstructionType.put( ">="   , CPINSTRUCTION_TYPE.Relational);

		// File Instruction Opcodes 
		String2CPInstructionType.put( "rm"   , CPINSTRUCTION_TYPE.File);
		String2CPInstructionType.put( "mv"   , CPINSTRUCTION_TYPE.File);

		// Builtin Instruction Opcodes 
		String2CPInstructionType.put( "abs"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "sin"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "cos"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "tan"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "sqrt" , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "log"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "exp"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "max"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "min"  , CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "plogp", CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "print", CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "print2",CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "nrow"  ,CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "ncol"  ,CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "length",CPINSTRUCTION_TYPE.Builtin);
		String2CPInstructionType.put( "round" ,CPINSTRUCTION_TYPE.Builtin);
		
		// Parameterized Builtin Functions
		String2CPInstructionType.put( "cdf"	 , CPINSTRUCTION_TYPE.ParameterizedBuiltin);

		// Variable Instruction Opcodes 
		String2CPInstructionType.put( "assignvar"   , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "mvvar"    	, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "rmvar"    	, CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "rmfilevar"   , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "assignvarwithfile", CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "attachfiletovar"  , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "valuepickCP" , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "iqsize"      , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "spearmanhelper", CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "readScalar"  , CPINSTRUCTION_TYPE.Variable);
		String2CPInstructionType.put( "writeScalar" , CPINSTRUCTION_TYPE.Variable);

		// User-defined function Opcodes
		String2CPInstructionType.put( "extfunct"   	, CPINSTRUCTION_TYPE.External);
	}

	public static CPInstruction parseSingleInstruction (String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
		return CPInstructionParser.parseSingleInstruction(cptype, str);
	}
	
	public static CPInstruction parseSingleInstruction ( CPINSTRUCTION_TYPE cptype, String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() ) 
			return null;
		switch(cptype) {
		case Arithmetic:
			return (CPInstruction) ArithmeticCPInstruction.parseInstruction(str);
			
		case Boolean:
			return (CPInstruction) BooleanCPInstruction.parseInstruction(str);
			
		case Relational:
			return (CPInstruction) RelationalCPInstruction.parseInstruction(str);
			
		case Builtin: 
			return (CPInstruction) BuiltinCPInstruction.parseInstruction(str);
		
		case ParameterizedBuiltin: 
			return (CPInstruction) ParameterizedBuiltinCPInstruction.parseInstruction(str);
		
		case File:
			return (CPInstruction) FileCPInstruction.parseInstruction(str);
			
		case Variable:
			return (CPInstruction) VariableCPInstruction.parseInstruction(str);
			
		case External:
			//return (CPInstruction) ExtBuiltinCPInstruction.parseInstruction(str);
			return (CPInstruction) FunctionCallCPInstruction.parseInstruction(str);
			
		case INVALID:
		default: 
			throw new DMLRuntimeException("Invalid MR Instruction Type: " + cptype );
		}
	}
	
	public static CPInstruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		Instruction[] inst = InstructionParser.parseMixedInstructions(str);
		CPInstruction[] cpinst = new CPInstruction[inst.length];
		for ( int i=0; i < inst.length; i++ ) {
			cpinst[i] = (CPInstruction) inst[i];
		}
		
		return cpinst;
	}
	
	
}
