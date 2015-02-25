/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.spark.AggregateUnarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.ArithmeticBinarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.MMCJSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.MapMultSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.MatrixIndexingSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.RelationalBinarySPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.ReorgSPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction.SPINSTRUCTION_TYPE;


public class SPInstructionParser extends InstructionParser {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final HashMap<String, SPINSTRUCTION_TYPE> String2SPInstructionType;
	static {
		String2SPInstructionType = new HashMap<String, SPInstruction.SPINSTRUCTION_TYPE>();
		//matrix multiplication operators
		String2SPInstructionType.put( "ba+*"   	, SPINSTRUCTION_TYPE.MMCJ);
		String2SPInstructionType.put( "mapmult",  SPINSTRUCTION_TYPE.MapMult);
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
		String2SPInstructionType.put( "r'"   	    , SPINSTRUCTION_TYPE.Reorg);
		String2SPInstructionType.put( "+"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "-"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "*"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "/"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "%%"   , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "%/%"  , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "^"    , SPINSTRUCTION_TYPE.ArithmeticBinary);
		String2SPInstructionType.put( "^2"   , SPINSTRUCTION_TYPE.ArithmeticBinary); //TODO: special ^ case
		String2SPInstructionType.put( "^2c-" , SPINSTRUCTION_TYPE.ArithmeticBinary); //TODO: special ^ case
		String2SPInstructionType.put( "*2"   , SPINSTRUCTION_TYPE.ArithmeticBinary); //TODO: special * case
		// Relational Instruction Opcodes 
		String2SPInstructionType.put( "=="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "!="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "<"    , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( ">"    , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( "<="   , SPINSTRUCTION_TYPE.RelationalBinary);
		String2SPInstructionType.put( ">="   , SPINSTRUCTION_TYPE.RelationalBinary);
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
		
		switch(sptype) 
		{
			case MMCJ:
				return MMCJSPInstruction.parseInstruction(str);
			case MapMult:
				return MapMultSPInstruction.parseInstruction(str);
				
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
