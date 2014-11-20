/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class AppendRInstruction extends AppendInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public AppendRInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out, istr);
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[3]);
			
		return new AppendRInstruction(null, in1, in2, out, str);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{		
		//get both inputs
		IndexedMatrixValue left = cachedValues.getFirst(input1);
		IndexedMatrixValue right = cachedValues.getFirst(input2);

		//check non-existing block
		if( left == null || right == null )
			throw new DMLRuntimeException("Missing append input.");
		
		//core append operation
		MatrixBlock mbLeft = (MatrixBlock)left.getValue();
		MatrixBlock mbRight = (MatrixBlock)right.getValue();
		
		MatrixBlock ret = mbLeft.appendOperations(mbRight, new MatrixBlock());
		
		//put result into cache
		cachedValues.add(output, new IndexedMatrixValue(left.getIndexes(), ret));
	}
}
