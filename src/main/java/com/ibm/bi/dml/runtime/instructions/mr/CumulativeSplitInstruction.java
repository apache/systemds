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
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;

/**
 * 
 * 
 */
public class CumulativeSplitInstruction extends UnaryInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixCharacteristics _mcIn = null;
	private long _lastRowBlockIndex = -1;
	private double _initValue = 0;
	
	public CumulativeSplitInstruction(byte in, byte out, double init, String istr)
	{
		super(null, in, out, istr);
		_initValue = init;
	}

	public void setMatrixCharacteristics( MatrixCharacteristics mcIn )
	{
		_mcIn = mcIn;
		_lastRowBlockIndex = (long)Math.ceil((double)_mcIn.getRows()/_mcIn.getRowsPerBlock());
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		double init = Double.parseDouble(parts[3]);
		
		return new CumulativeSplitInstruction(in, out, init, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList == null ) 
			return;
		
		for(IndexedMatrixValue in1 : blkList)
		{
			if(in1==null) continue;
			
			MatrixIndexes inix = in1.getIndexes();
			MatrixBlock blk = (MatrixBlock)in1.getValue();
			long rixOffset = (inix.getRowIndex()-1)*blockRowFactor;
			boolean firstBlk = (inix.getRowIndex() == 1);
			boolean lastBlk = (inix.getRowIndex() == _lastRowBlockIndex );
			
			//introduce offsets w/ init value for first row 
			if( firstBlk ) { 
				IndexedMatrixValue out = cachedValues.holdPlace(output, valueClass);
				((MatrixBlock) out.getValue()).reset(1,blk.getNumColumns());
				if( _initValue != 0 ){
					for( int j=0; j<blk.getNumColumns(); j++ )
						((MatrixBlock) out.getValue()).appendValue(0, j, _initValue);
				}
				out.getIndexes().setIndexes(1, inix.getColumnIndex());
			}	
			
			//output splitting (shift by one), preaggregated offset used by subsequent block
			for( int i=0; i<blk.getNumRows(); i++ )
				if( !(lastBlk && i==(blk.getNumRows()-1)) ) //ignore last row
				{
					IndexedMatrixValue out = cachedValues.holdPlace(output, valueClass);
					MatrixBlock tmpBlk = (MatrixBlock) out.getValue();
					tmpBlk.reset(1,blk.getNumColumns());
					blk.sliceOperations(i+1, i+1, 1, blk.getNumColumns(), tmpBlk);	
					out.getIndexes().setIndexes(rixOffset+i+2, inix.getColumnIndex());
				}
		}
	}
}
