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
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class AppendGInstruction extends AppendInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _offset = -1; //cols of input1 
	private long _offset2 = -1; //cols of input2
	private long _clen = -1;
	
	
	public AppendGInstruction(Operator op, byte in1, byte in2, long offset, long offset2, byte out, String istr)
	{
		super(op, in1, in2, out, istr);
		_offset = offset;
		_offset2 = offset2;
		_clen = _offset + _offset2;
	}

	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		InstructionUtils.checkNumFields ( str, 5 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		long offset = (long)(Double.parseDouble(parts[3]));
		long clen = (long)(Double.parseDouble(parts[4]));
		byte out = Byte.parseByte(parts[5]);
			
		return new AppendGInstruction(null, in1, in2, offset, clen, out, str);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//Step 1: handle first input (forward blocks, change dim of last block)
		ArrayList<IndexedMatrixValue> blkList1 = cachedValues.get(input1);
		if( blkList1 != null )
			for( IndexedMatrixValue in1 : blkList1 )
			{
				if( in1 == null )
					continue;
				
				if( _offset%bclen == 0 ) //special case: forward only
				{
					cachedValues.add(output, in1);	
				}
				else //general case: change dims and forward
				{					
					MatrixIndexes tmpix = in1.getIndexes();
					MatrixBlock tmpval = (MatrixBlock) in1.getValue(); //always block
					if( _offset/bclen+1 == tmpix.getColumnIndex() ) //border block
					{
						IndexedMatrixValue data = cachedValues.holdPlace(output, valueClass);
						MatrixBlock tmpvalNew = (MatrixBlock)data.getValue(); //always block
						int cols = Math.min(bclen, (int)(_clen-(tmpix.getColumnIndex()-1)*bclen));
						tmpvalNew.reset(tmpval.getNumRows(), cols);						
						tmpvalNew.copy(0, tmpval.getNumRows()-1, 0, tmpval.getNumColumns()-1, tmpval, true);
						data.getIndexes().setIndexes(tmpix);
					}
					else //inner block
					{
						cachedValues.add(output, in1); 
					}	
				}
			}
		
		//Step 2: handle second input (split/forward blocks with new index)
		ArrayList<IndexedMatrixValue> blkList2 = cachedValues.get(input2);
		if( blkList2 != null ) 
			for( IndexedMatrixValue in2 : blkList2 )
			{
				if( in2 == null )
					continue;
				
				if( _offset%bclen == 0 ) //special case no split
				{
					IndexedMatrixValue data = cachedValues.holdPlace(output, valueClass);
					MatrixIndexes ixtmp = data.getIndexes();
					ixtmp.setIndexes(in2.getIndexes().getRowIndex(), 
							         _offset/bclen + in2.getIndexes().getColumnIndex());
					data.set(ixtmp, in2.getValue());
				}
				else //general case: split and forward
				{		
					MatrixIndexes tmpix = in2.getIndexes();
					MatrixBlock tmpval = (MatrixBlock) in2.getValue(); //always block
					
					//first half
					IndexedMatrixValue data1 = cachedValues.holdPlace(output, valueClass);
					MatrixIndexes ix1 = data1.getIndexes();
					MatrixBlock tmpvalNew = (MatrixBlock)data1.getValue(); //always block
					int cix1 = (int)(_offset/bclen + in2.getIndexes().getColumnIndex());
					int cols1 = Math.min(bclen, (int)(_clen-(long)(cix1-1)*bclen));
					ix1.setIndexes( tmpix.getRowIndex(), cix1);
					tmpvalNew.reset( tmpval.getNumRows(), cols1 );
					tmpvalNew.copy(0, tmpval.getNumRows()-1, (int)((_offset+1)%bclen)-1, cols1-1, 
							       tmpval.sliceOperations(1, tmpval.getNumRows(), 1, 
							    		                     cols1-((_offset)%bclen), new MatrixBlock()), true);
					data1.getIndexes().setIndexes(ix1);
					
					if( cols1-((_offset)%bclen)<tmpval.getNumColumns() ) 
					{
						//second half (if required)
						IndexedMatrixValue data2 = cachedValues.holdPlace(output, valueClass);
						MatrixIndexes ix2 = data2.getIndexes();
						MatrixBlock tmpvalNew2 = (MatrixBlock)data2.getValue(); //always block
						int cix2 = (int)(_offset/bclen + 1 + in2.getIndexes().getColumnIndex());
						int cols2 = Math.min(bclen, (int)(_clen-(long)(cix2-1)*bclen));
						ix2.setIndexes( tmpix.getRowIndex(), cix2);
						tmpvalNew2.reset( tmpval.getNumRows(), cols2 );
						tmpvalNew2.copy(0, tmpval.getNumRows()-1, 0, cols2-1, 
								       tmpval.sliceOperations(1, tmpval.getNumRows(), cols1-((_offset)%bclen)+1, 
								    		                     tmpval.getNumColumns(), new MatrixBlock()), true);
						data2.getIndexes().setIndexes(ix2);
					}
				}
			}
	}
}
