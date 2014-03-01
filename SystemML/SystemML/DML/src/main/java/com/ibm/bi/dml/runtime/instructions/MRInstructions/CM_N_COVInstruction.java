/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class CM_N_COVInstruction extends UnaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CM_N_COVInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.CM_N_COV;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		int cst;
		String opcode = parts[0];
		
		if (opcode.equalsIgnoreCase("cm") ) 
		{
			in = Byte.parseByte(parts[1]);
			cst = Integer.parseInt(parts[2]);
			out = Byte.parseByte(parts[3]);
			
			if(cst>4 || cst<0 || cst==1)
				throw new DMLRuntimeException("constant for central moment has to be 0, 2, 3, or 4");
			
			AggregateOperationTypes opType = CMOperator.getCMAggOpType(cst);
			CMOperator cm = new CMOperator(CM.getCMFnObject(opType), opType);
			return new CM_N_COVInstruction(cm, in, out, str);
		}else if(opcode.equalsIgnoreCase("cov"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			COVOperator cov = new COVOperator(COV.getCOMFnObject());
			return new CM_N_COVInstruction(cov, in, out, str);
		}/*else if(opcode.equalsIgnoreCase("mean"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			
			CMOperator mean = new CMOperator(CM.getCMFnObject(), CMOperator.AggregateOperationTypes.MEAN);
			return new CM_N_COVInstruction(mean, in, out, str);
		}*/
		else
			throw new DMLRuntimeException("unknown opcode "+opcode);
		
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}
}
