/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.HashMap;

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
		
		if( !(inData.getValue() instanceof MatrixBlock || inOffset.getValue() instanceof MatrixBlock) )
			throw new DMLRuntimeException("Unsupported input data: expected "+MatrixBlock.class.getName()+" but got "+inData.getValue().getClass().getName()+" and "+inOffset.getValue().getClass().getName());
		if(     _rmRows && inData.getValue().getNumRows()!=inOffset.getValue().getNumRows() 
			|| !_rmRows && inData.getValue().getNumColumns()!=inOffset.getValue().getNumColumns()  ){
			throw new DMLRuntimeException("Dimension mismatch between input data and offsets: ["
					+inData.getValue().getNumRows()+"x"+inData.getValue().getNumColumns()+" vs "+inOffset.getValue().getNumRows()+"x"+inOffset.getValue().getNumColumns());
		}
		
		//compute outputs (at most two output blocks)
		HashMap<MatrixIndexes,IndexedMatrixValue> out = new HashMap<MatrixIndexes,IndexedMatrixValue>();
		MatrixBlock linData = (MatrixBlock) inData.getValue();
		MatrixBlock linOffset = (MatrixBlock) inOffset.getValue();
		MatrixIndexes tmpIx = new MatrixIndexes(-1,-1);
		if( _rmRows ) //margin = "rows"
		{
			long rlen = _len;
			long clen = linData.getNumColumns();
			
			for( int i=0; i<linOffset.getNumRows(); i++ ) {
				long rix = (long)linOffset.quickGetValue(i, 0);
				if( rix > 0 ) //otherwise empty row
				{
					//get single row from source block
					MatrixBlock src = (MatrixBlock) linData.sliceOperations(
							  i+1, i+1, 1, clen, new MatrixBlock());
					long brix = (rix-1)/blockRowFactor+1;
					long lbrix = (rix-1)%blockRowFactor;
					tmpIx.setIndexes(brix, inData.getIndexes().getColumnIndex());
					 //create target block if necessary
					if( !out.containsKey(tmpIx) ) {
						IndexedMatrixValue tmpIMV = cachedValues.holdPlace(output, valueClass);
						tmpIMV.getIndexes().setIndexes(tmpIx);
						((MatrixBlock)tmpIMV.getValue()).reset((int)Math.min(blockRowFactor, rlen-((brix-1)*blockRowFactor)), (int)clen);
						out.put(tmpIMV.getIndexes(), tmpIMV);
					}
					//put single row into target block
					((MatrixBlock)out.get(tmpIx).getValue()).copy(
							  (int)lbrix, (int)lbrix, 0, (int)clen-1, src, false);
				}
			}
		}
		else //margin = "cols"
		{
			long rlen = linData.getNumRows();
			long clen = _len;
			
			for( int i=0; i<linOffset.getNumColumns(); i++ ) {
				long cix = (long)linOffset.quickGetValue(0, i);
				if( cix > 0 ) //otherwise empty row
				{
					//get single row from source block
					MatrixBlock src = (MatrixBlock) linData.sliceOperations(
							  1, rlen, i+1, i+1, new MatrixBlock());
					long bcix = (cix-1)/blockColFactor+1;
					long lbcix = (cix-1)%blockColFactor;
					tmpIx.setIndexes(inData.getIndexes().getRowIndex(), bcix);
					 //create target block if necessary
					if( !out.containsKey(tmpIx) ) {
						IndexedMatrixValue tmpIMV = cachedValues.holdPlace(output, valueClass);
						tmpIMV.getIndexes().setIndexes(tmpIx);
						((MatrixBlock)tmpIMV.getValue()).reset((int)rlen,(int)Math.min(blockRowFactor, clen-((bcix-1)*blockColFactor)));
						out.put(tmpIMV.getIndexes(), tmpIMV);
					}
					//put single row into target block
					((MatrixBlock)out.get(tmpIx).getValue()).copy(
							  0, (int)rlen-1, (int)lbcix, (int)lbcix, src, false);
				}
			}
		}
		
		
		//prepare and return outputs (already in cached values)
		for( IndexedMatrixValue imv : out.values() ){
			((MatrixBlock)imv.getValue()).recomputeNonZeros();
		}
	}
}
