/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;


import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AppendInstruction extends BinaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	/**
	 * 
	 * @param op
	 * @param in1
	 * @param in2
	 * @param out
	 * @param istr
	 */
	public AppendInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		instString = istr;	
		mrtype = MRINSTRUCTION_TYPE.Append;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if( opcode.equals("mappend") )
			return AppendMInstruction.parseInstruction(str);
		else if( opcode.equals("rappend") )
			return AppendRInstruction.parseInstruction(str);
		else if( opcode.equals("gappend") )
			return AppendGInstruction.parseInstruction(str);
		else
			throw new DMLRuntimeException("Unsupported append operation code: "+opcode);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		throw new DMLUnsupportedOperationException("Operations on base append instruction not supported.");
	}
}
