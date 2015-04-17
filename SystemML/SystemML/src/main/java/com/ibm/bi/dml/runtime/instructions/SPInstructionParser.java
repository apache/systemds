/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Checkpoint;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.cp.BuiltinBinaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.BuiltinUnaryCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.AggregateUnarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.ArithmeticBinarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.BuiltinBinarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.BuiltinUnarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.CSVReblockSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.CheckpointSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.CpmmSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.MapmmSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.MatrixIndexingSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.ReblockSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.RelationalBinarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.ReorgSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction.SPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.spark.TsmmSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.SortSPInstruction;


public class SPInstructionParser extends InstructionParser {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final HashMap<String, SPINSTRUCTION_TYPE> String2SPInstructionType;
	static {
		String2SPInstructionType = new HashMap<String, SPInstruction.SPINSTRUCTION_TYPE>();
		//matrix multiplication operators
		String2SPInstructionType.put( "ba+*"   	, SPINSTRUCTION_TYPE.CPMM);
		String2SPInstructionType.put( "mapmm"   , SPINSTRUCTION_TYPE.MAPMM);
		String2SPInstructionType.put( "tsmm"    , SPINSTRUCTION_TYPE.TSMM);
		
		//unary aggregate operators
		String2SPInstructionType.put( "uak+"   	, SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uark+"   , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uack+"   , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uamean"  , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uarmean" , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uacmean" , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uamax"   , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uarmax"  , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uarimax",  SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uacmax"  , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uamin"   , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uarmin"  , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uarimin" , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uacmin"  , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "ua+"     , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uar+"    , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uac+"    , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "ua*"     , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uatrace" , SPINSTRUCTION_TYPE.AggregateUnary);
		String2SPInstructionType.put( "uaktrace", SPINSTRUCTION_TYPE.AggregateUnary);

		String2SPInstructionType.put( "rangeReIndex"   	, SPINSTRUCTION_TYPE.MatrixIndexing);
		String2SPInstructionType.put( "leftIndex"   	, SPINSTRUCTION_TYPE.MatrixIndexing);
		
		String2SPInstructionType.put( "r'"   	    , SPINSTRUCTION_TYPE.Reorg);
		String2SPInstructionType.put( "rdiag"   	    , SPINSTRUCTION_TYPE.Reorg);
		String2SPInstructionType.put( "rsort"   	    , SPINSTRUCTION_TYPE.Reorg);
		
		String2SPInstructionType.put( "+"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "-"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "*"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "/"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "%%"   , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "%/%"  , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "^"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "^2"   , SPINSTRUCTION_TYPE.ArithmeticBinary); //TODO: special ^ case
		String2SPInstructionType.put( "*2"   , SPINSTRUCTION_TYPE.ArithmeticBinary); //TODO: special * case
		// Relational Instruction Opcodes 
		String2SPInstructionType.put( "=="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "!="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "<"    , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( ">"    , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "<="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( ">="   , SPINSTRUCTION_TYPE.RelationalBinary);
		
		// REBLOCK Instruction Opcodes 
		String2SPInstructionType.put( "rblk"   , SPINSTRUCTION_TYPE.Reblock);
		String2SPInstructionType.put( "csvrblk", SPINSTRUCTION_TYPE.CSVReblock);
	
		// Spark-specific instructions
		String2SPInstructionType.put( Checkpoint.OPCODE, SPINSTRUCTION_TYPE.Checkpoint);
		
		// Builtin Instruction Opcodes 
		String2SPInstructionType.put( "log"  , SPINSTRUCTION_TYPE.Builtin);
		
		String2SPInstructionType.put( "max"  , SPINSTRUCTION_TYPE.BuiltinBinary);
		String2SPInstructionType.put( "min"  , SPINSTRUCTION_TYPE.BuiltinBinary);
		String2SPInstructionType.put( "solve"  , SPINSTRUCTION_TYPE.BuiltinBinary);
		
		String2SPInstructionType.put( "exp"   , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "abs"   , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "sin"   , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "cos"   , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "tan"   , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "asin"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "acos"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "atan"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "sqrt"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "plogp" , SPINSTRUCTION_TYPE.BuiltinUnary);
		// String2SPInstructionType.put( "print" , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "round" , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "ceil"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "floor" , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "ucumk+", SPINSTRUCTION_TYPE.BuiltinUnary);
		// String2SPInstructionType.put( "stop"  , SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "inverse", SPINSTRUCTION_TYPE.BuiltinUnary);
		String2SPInstructionType.put( "sprop", SPINSTRUCTION_TYPE.BuiltinUnary);
		
		String2SPInstructionType.put( "sort"  , SPINSTRUCTION_TYPE.Sort);
		String2SPInstructionType.put( "inmem-iqm"  		, SPINSTRUCTION_TYPE.Variable);
	}

	public static Instruction parseSingleInstruction (String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;

		SPINSTRUCTION_TYPE cptype = InstructionUtils.getSPType(str); 
		if ( cptype == null )
			// return null;
			throw new DMLUnsupportedOperationException("Invalid SP Instruction Type: " + str);
		Instruction cpinst = SPInstructionParser.parseSingleInstruction(cptype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static Instruction parseSingleInstruction ( SPINSTRUCTION_TYPE sptype, String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( str == null || str.isEmpty() ) 
			return null;
		
		String [] parts = null;
		switch(sptype) 
		{
			// Matrix multiplication
			case CPMM:
				return CpmmSPInstruction.parseInstruction(str);
			case MAPMM:
				return MapmmSPInstruction.parseInstruction(str);
			case TSMM:
				return TsmmSPInstruction.parseInstruction(str);
				
			case AggregateUnary:
				return AggregateUnarySPInstruction.parseInstruction(str);
				
			case MatrixIndexing:
				return MatrixIndexingSPInstruction.parseInstruction(str);
			case Reorg:
				return ReorgSPInstruction.parseInstruction(str);
			case ArithmeticBinary:
				return ArithmeticBinarySPInstruction.parseInstruction(str);
			case RelationalBinary:
				return RelationalBinarySPInstruction.parseInstruction(str);
				
			// Reblock instructions	
			case Reblock:
				return ReblockSPInstruction.parseInstruction(str);
			case CSVReblock:
				return CSVReblockSPInstruction.parseInstruction(str);
			
			case Builtin: 
				parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if ( parts[0].equals("log") ) {
					if ( parts.length == 3 ) {
						// B=log(A), y=log(x)
						return (SPInstruction) BuiltinUnarySPInstruction.parseInstruction(str);
					} else if ( parts.length == 4 ) {
						// B=log(A,10), y=log(x,10)
						return (SPInstruction) BuiltinBinarySPInstruction.parseInstruction(str);
					}
				}
				else {
					throw new DMLRuntimeException("Invalid Builtin Instruction: " + str );
				}
				
			case BuiltinBinary:
				parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if ( parts[0].equals("solve") ) {
					return (CPInstruction) BuiltinBinaryCPInstruction.parseInstruction(str);
				}
				return (SPInstruction) BuiltinBinarySPInstruction.parseInstruction(str);
				
			case BuiltinUnary:
				parts = InstructionUtils.getInstructionPartsWithValueType(str);
				if ( parts[0].equals("ucumk+") || parts[0].equals("inverse") ) {
					// For now, ucumk+, inverse are not implemented
					return (CPInstruction) BuiltinUnaryCPInstruction.parseInstruction(str);
				}
				else {
					return (SPInstruction) BuiltinUnarySPInstruction.parseInstruction(str);
				}
				
			case Sort: 
				return (SPInstruction) SortSPInstruction.parseInstruction(str);
				
			case Variable:
				return (CPInstruction) VariableCPInstruction.parseInstruction(str);
				
			case Checkpoint:
				return CheckpointSPInstruction.parseInstruction(str);
				
			case INVALID:
			default:
				throw new DMLUnsupportedOperationException("Invalid SP Instruction Type: " + sptype );
		}
	}
	
	public static SPInstruction[] parseMixedInstructions ( String str ) throws DMLUnsupportedOperationException, DMLRuntimeException {
		if ( str == null || str.isEmpty() )
			return null;
		
		Instruction[] inst = InstructionParser.parseMixedInstructions(str);
		SPInstruction[] cpinst = new SPInstruction[inst.length];
		for ( int i=0; i < inst.length; i++ ) {
			cpinst[i] = (SPInstruction) inst[i];
		}
		
		return cpinst;
	}
	
	
}
