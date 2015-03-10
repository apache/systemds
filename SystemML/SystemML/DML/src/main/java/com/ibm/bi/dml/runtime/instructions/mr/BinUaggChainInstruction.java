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
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateUnaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;


/**
 * 
 */
public class BinUaggChainInstruction extends UnaryInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//operators
	private BinaryOperator _bOp = null;
	private AggregateUnaryOperator _uaggOp = null;
	
	//reused intermediates  
	private MatrixIndexes _tmpIx = null;
	private MatrixValue _tmpVal = null;
	
	/**
	 * 
	 * @param bop
	 * @param uaggop
	 * @param in1
	 * @param out
	 * @param istr
	 */
	public BinUaggChainInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, byte in1, byte out, String istr)
	{
		super(null, in1, out, istr);
		
		_bOp = bop;
		_uaggOp = uaggop;
		
		_tmpIx = new MatrixIndexes();
		_tmpVal = new MatrixBlock();
		
		mrtype = MRINSTRUCTION_TYPE.BinUaggChain;
		instString = istr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 4 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts( str );		
		
		BinaryOperator bop = BinaryInstruction.parseBinaryOperator(parts[1]);
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[2]);
		byte in1 = Byte.parseByte(parts[3]);
		byte out = Byte.parseByte(parts[4]);
		
		return new BinUaggChainInstruction(bop, uaggop, in1, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			           IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get( input );
		if( blkList == null )
			return;

		for(IndexedMatrixValue imv : blkList)
		{
			if(imv == null)
				continue;
			
			MatrixIndexes inIx = imv.getIndexes();
			MatrixValue inVal = imv.getValue();
			
			//allocate space for the intermediate and output value
			IndexedMatrixValue iout = cachedValues.holdPlace(output, valueClass);
			MatrixIndexes outIx = iout.getIndexes();
			MatrixValue outVal = iout.getValue();
			
			//process instruction
			OperationsOnMatrixValues.performAggregateUnary( inIx, inVal, _tmpIx, _tmpVal, _uaggOp, blockRowFactor, blockColFactor);
			((MatrixBlock)_tmpVal).dropLastRowsOrColums(_uaggOp.aggOp.correctionLocation);
			OperationsOnMatrixValues.performBinaryIgnoreIndexes(inVal, _tmpVal, outVal, _bOp);
			outIx.setIndexes(inIx);
		}
	}
}
