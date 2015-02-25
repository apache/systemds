/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.lops.Tertiary.OperationTypes;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;


public class CombineTertiaryInstruction extends TertiaryInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CombineTertiaryInstruction(OperationTypes op, byte in1, byte in2,
			byte in3, byte out, String istr) {
		super(op, in1, in2, in3, out, -1, -1, istr);
		mrtype = MRINSTRUCTION_TYPE.CombineTertiary;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		// example instruction string - ctabletransform:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE:::3:DOUBLE 
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, in3, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		in3 = Byte.parseByte(parts[3]);
		out = Byte.parseByte(parts[4]);
		
		if ( opcode.equalsIgnoreCase("combinetertiary") ) {
			return new CombineTertiaryInstruction(null, in1, in2, in3, out, str);
		} 
		return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("CombineTertiaryInstruction.processInstruction should never be called!");
	}
}
