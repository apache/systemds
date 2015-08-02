/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


/**
 * Supported optcodes: rmempty.
 * 
 */
public class RemoveEmptyMRInstruction extends BinaryInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private long    _len   = -1;
	private boolean _rmRows = true;
	
	
	public RemoveEmptyMRInstruction(Operator op, byte in1, byte in2, long len, boolean rmRows, byte out, String istr)
	{
		super(op, in1, in2, out, istr);
		instString = istr;
		
		_len = len;
		_rmRows = rmRows;
	}
	
	public boolean isRemoveRows()
	{
		return _rmRows;
	}
	
	public long getOutputLen()
	{
		return _len;
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
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		
		if(!opcode.equalsIgnoreCase("rmempty"))
			throw new DMLRuntimeException("Unknown opcode while parsing an RemoveEmptyMRInstruction: " + str);
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long rlen = UtilFunctions.toLong(Double.parseDouble(parts[3]));
		boolean rmRows = parts[4].equals("rows");
		byte out = Byte.parseByte(parts[5]);
		
		return new RemoveEmptyMRInstruction(null, in1, in2, rlen, rmRows, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{			
		//get input and offsets
		IndexedMatrixValue inData = cachedValues.getFirst(input1);
		IndexedMatrixValue inOffset = cachedValues.getFirst(input2);

		//execute remove empty operations
		ArrayList<IndexedMatrixValue> out = new ArrayList<IndexedMatrixValue>();
		LibMatrixReorg.rmempty(inData, inOffset, _rmRows, _len, blockRowFactor, blockColFactor, out);
		
		//put results into cache map
		for( IndexedMatrixValue imv : out )
			cachedValues.add(output, imv);
	}
}
