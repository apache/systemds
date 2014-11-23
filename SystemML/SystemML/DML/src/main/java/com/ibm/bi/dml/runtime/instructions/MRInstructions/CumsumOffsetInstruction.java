/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.Builtin;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


public class CumsumOffsetInstruction extends BinaryInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private BinaryOperator _bop = null;
	private UnaryOperator _uop = null;
	
	public CumsumOffsetInstruction(byte in1, byte in2, byte out, String istr)
	{
		super(null, in1, in2, out, istr);
		
		_bop = new BinaryOperator(Plus.getPlusFnObject());
		_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+"));
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[3]);
		
		return new CumsumOffsetInstruction(in1, in2, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		IndexedMatrixValue in1 = cachedValues.getFirst(input1); //original data 
		IndexedMatrixValue in2 = cachedValues.getFirst(input2); //offset row vector
				
		if( in1 == null || in2 == null ) 
			throw new DMLRuntimeException("Unexpected empty input (left="+((in1==null)?"null":in1.getIndexes())
					                                     +", right="+((in2==null)?"null":in2.getIndexes())+").");
		
		//prepare inputs and outputs
		IndexedMatrixValue out = cachedValues.holdPlace(output, valueClass);
		MatrixBlock data = (MatrixBlock) in1.getValue();
		MatrixBlock offset = (MatrixBlock) in2.getValue();
		MatrixBlock blk = (MatrixBlock) out.getValue();
		blk.reset(data.getNumRows(), data.getNumColumns());
		
		//blockwise offset aggregation and prefix sum computation
		MatrixBlock data2 = new MatrixBlock(data); //cp data
		MatrixBlock fdata2 = (MatrixBlock) data2.sliceOperations(1, 1, 1, data2.getNumColumns(), new MatrixBlock()); //1-based
		fdata2.binaryOperationsInPlace(_bop, offset); //sum offset to first row
		data2.copy(0, 0, 0, data2.getNumColumns()-1, fdata2, true); //0-based
		data2.unaryOperations(_uop, blk); //compute columnwise prefix sums

		//set output indexes
		out.getIndexes().setIndexes(in1.getIndexes());		
	}
}
