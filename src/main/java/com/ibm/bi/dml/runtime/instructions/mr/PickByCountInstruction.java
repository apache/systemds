/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.lops.PickByCount.OperationTypes;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class PickByCountInstruction extends MRInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public byte input1; // used for both valuepick and rangepick
	public byte input2; // used only for valuepick
	public double cst; // used only for rangepick
	public boolean isValuePick=true;
	
	/*
	 *  Constructor for valuepick
	 *  valuepick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE
	 *  0 is data matrix, 1 is the quantile matrix, 2 will have the resulting picked data items 
	 */
	public PickByCountInstruction(Operator op, byte _in1, byte _in2, byte out, String istr) {
		super(op, out);
		input1 = _in1;
		input2 = _in2;
		cst = 0;
		mrtype = MRINSTRUCTION_TYPE.PickByCount;
		instString = istr;
		isValuePick=true;
	}

	/*
	 *  Constructor for rangepick
	 *  rangepick:::0:DOUBLE:::0.25:DOUBLE:::1:DOUBLE
	 *  0 is data matrix, 0.25 is the quantile that needs to be removed from both ends in the PDF, 
	 *  1 will have the resulting picked data items between [Q_1-Q_3]
	 */
	public PickByCountInstruction(Operator op, byte _in1, double _cst, byte out, String istr) {
		super(op, out);
		input1 = _in1;
		cst = _cst;
		mrtype = MRINSTRUCTION_TYPE.PickByCount;
		instString = istr;
		isValuePick=false;
	}

	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		InstructionUtils.checkNumFields ( str, 5 );
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		OperationTypes ptype = OperationTypes.valueOf(parts[4]);
		
		if ( ptype == OperationTypes.VALUEPICK ) 
		{
			byte in1 = Byte.parseByte(parts[1]);
			byte in2 = Byte.parseByte(parts[2]);
			byte out = Byte.parseByte(parts[3]);
			return new PickByCountInstruction(null, in1, in2, out, str);
		} 
		else if ( ptype == OperationTypes.RANGEPICK ) 
		{
			byte in1 = Byte.parseByte(parts[1]);
			double cstant = Double.parseDouble(parts[2]);
			byte out = Byte.parseByte(parts[3]);
			return new PickByCountInstruction(null, in1, cstant, out, str);
		}
		
		return null;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("PickByCountInstruction.processInstruction should never be called!");
		
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		if( isValuePick ) {
			return new byte[]{input1,input2,output};
		}
		else { //range pick
			return new byte[]{input1, output};
		}
		
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		if( isValuePick ) {
			return new byte[]{input1,input2};
		}
		else { //range pick
			return new byte[]{input1};
		}
	}
}
