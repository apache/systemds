/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.util.StringTokenizer;

import com.ibm.bi.dml.lops.AppendM;
import com.ibm.bi.dml.lops.BinaryM;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.MapMultChain;
import com.ibm.bi.dml.lops.PMMJ;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;


public class InstructionUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param str
	 * @param expected
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int checkNumFields( String str, int expected ) 
		throws DMLRuntimeException 
	{
		//note: split required for empty tokens
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		
		if ( numFields != expected ) 
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected + ") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}
	
	/**
	 * 
	 * @param str
	 * @param expected1
	 * @param expected2
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static int checkNumFields( String str, int expected1, int expected2 ) 
		throws DMLRuntimeException 
	{
		//note: split required for empty tokens
		int numParts = str.split(Instruction.OPERAND_DELIM).length;
		int numFields = numParts - 2; // -2 accounts for execType and opcode
		
		if ( numFields != expected1 && numFields != expected2 ) 
			throw new DMLRuntimeException("checkNumFields() for (" + str + ") -- expected number (" + expected1 + " or "+ expected2 +") != is not equal to actual number (" + numFields + ").");
		
		return numFields; 
	}
	
	/**
	 * Given an instruction string, strip-off the execution type and return 
	 * opcode and all input/output operands WITHOUT their data/value type. 
	 * i.e., ret.length = parts.length-1 (-1 for execution type)
	 * 
	 * @param str
	 * @return 
	 */
	public static String[] getInstructionParts( String str ) 
	{
		StringTokenizer st = new StringTokenizer( str, Instruction.OPERAND_DELIM );
		String[] ret = new String[st.countTokens()-1];
		st.nextToken(); // stripping-off the exectype
		ret[0] = st.nextToken(); // opcode
		int index = 1;
		while( st.hasMoreTokens() ){
			String tmp = st.nextToken();
			int ix = tmp.indexOf(Instruction.DATATYPE_PREFIX);
			ret[index++] = tmp.substring(0,((ix>=0)?ix:tmp.length()));	
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
	public static String[] getInstructionPartsWithValueType( String str ) 
	{
		//note: split required for empty tokens
		String[] parts = str.split(Instruction.OPERAND_DELIM);
		String[] ret = new String[parts.length-1]; // stripping-off the exectype
		ret[0] = parts[1]; // opcode
		for( int i=1; i<parts.length; i++ )
			ret[i-1] = parts[i];
		
		return ret;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 */
	public static String getOpCode( String str ) 
	{
		int ix1 = str.indexOf(Instruction.OPERAND_DELIM);
		int ix2 = str.indexOf(Instruction.OPERAND_DELIM, ix1+1);
		return str.substring(ix1+1, ix2);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLUnsupportedOperationException
	 */
	public static MRINSTRUCTION_TYPE getMRType( String str ) 
		throws DMLUnsupportedOperationException 
	{
		String opcode = getOpCode(str);
		MRINSTRUCTION_TYPE mrtype = MRInstructionParser.String2MRInstructionType.get( opcode ); 
		return mrtype;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLUnsupportedOperationException
	 */
	public static CPINSTRUCTION_TYPE getCPType( String str ) 
		throws DMLUnsupportedOperationException 
	{
		String opcode = getOpCode(str);
		CPINSTRUCTION_TYPE cptype = CPInstructionParser.String2CPInstructionType.get( opcode ); 
		return cptype;
	}
	
	/**
	 * 
	 * @param opcode
	 * @return
	 */
	public static boolean isBuiltinFunction ( String opcode ) 
	{
		Builtin.BuiltinFunctionCode bfc = Builtin.String2BuiltinFunctionCode.get(opcode);
		return (bfc != null);
	}
	
	public static boolean isOperand(String str) 
	{
		//note: split required for empty tokens
		String[] parts = str.split(Instruction.DATATYPE_PREFIX);
		return (parts.length > 1);
	}
	
	/**
	 * Evaluates if at least one instruction of the given instruction set
	 * used the distributed cache.
	 * 
	 * @param str
	 * @return
	 */
	public static boolean isDistributedCacheUsed(String str) 
	{	
		String[] parts = str.split(Instruction.INSTRUCTION_DELIM);
		for(String inst : parts) 
		{
			String opcode = getOpCode(inst);
			if(  opcode.equalsIgnoreCase(AppendM.OPCODE)  
			   || opcode.equalsIgnoreCase(MapMult.OPCODE)
			   || opcode.equalsIgnoreCase(PMMJ.OPCODE)
			   || opcode.equalsIgnoreCase(MapMultChain.OPCODE)
			   || BinaryM.isOpcode( opcode ) ) //multiple opcodes
			{
				return true;
			}
		}
		return false;
	}
	
}
