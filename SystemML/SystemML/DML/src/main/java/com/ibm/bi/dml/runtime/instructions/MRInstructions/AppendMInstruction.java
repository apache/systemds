/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.MRInstructions;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AppendMInstruction extends BinaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _offset; 
	public AppendMInstruction(Operator op, byte in1, byte in2, long offset, byte out, String istr)
	{
		super(op, in1, in2, out);
		_offset = offset;
		mrtype = MRINSTRUCTION_TYPE.AppendM;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long offset = (long)(Double.parseDouble(parts[3]));
		byte out = Byte.parseByte(parts[4]);
			
		return new AppendMInstruction(null, in1, in2, offset, out, str);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input1);
		if( blkList == null ) 
			return;
		
		//right now this only deals with appending matrix whith number of column <= blockColFactor
		for(IndexedMatrixValue in1 : blkList)
		{
			if(in1==null)
				continue;
			
			//check if this is a boundary block
			long lastBlockColIndex=_offset/blockColFactor;
			if(_offset%blockColFactor!=0)
				lastBlockColIndex++;
			if(in1.getIndexes().getColumnIndex()!=lastBlockColIndex)
				cachedValues.add(output, in1);
			else
			{
				DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input2);
				MatrixValue value_in2 = dcInput.getDataBlock(in1.getIndexes().getRowIndex(), 1, blockRowFactor, blockColFactor).getValue();
				
				//allocate space for the output value
				ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(2);
				IndexedMatrixValue first=cachedValues.holdPlace(output, valueClass);
				first.getIndexes().setIndexes(in1.getIndexes());
				outlist.add(first);
				
				if(in1.getValue().getNumColumns()+value_in2.getNumColumns()>blockColFactor)
				{
					IndexedMatrixValue second=cachedValues.holdPlace(output, valueClass);
					second.getIndexes().setIndexes(in1.getIndexes().getRowIndex(), in1.getIndexes().getColumnIndex()+1);
					outlist.add(second);
				}
	
				OperationsOnMatrixValues.performAppend(in1.getValue(), value_in2, outlist, 
					blockRowFactor, blockColFactor, true, 0);
			
			}
		}
	}
}
