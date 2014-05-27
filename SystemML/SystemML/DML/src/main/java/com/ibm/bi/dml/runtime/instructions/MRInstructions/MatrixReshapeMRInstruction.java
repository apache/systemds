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
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


/**
 * 
 * 
 */
public class MatrixReshapeMRInstruction extends UnaryInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _rows = -1;
	private long _cols = -1;
	private boolean _byrow = false;
	
	private MatrixCharacteristics _mcIn = null;
	private MatrixCharacteristics _mcOut = null;
	
	//MB: cache should be integrated with tempValues, but for n blocks
	private ArrayList<IndexedMatrixValue> _cache = null;
	
	public MatrixReshapeMRInstruction(Operator op, byte in, long rows, long cols, boolean byrow, byte out, String istr)
	{
		super(op, in, out, istr);
		mrtype = MRINSTRUCTION_TYPE.MMTSJ;
		instString = istr;
		
		_rows = rows;
		_cols = cols;
		_byrow = byrow;
	}
	
	public void setMatrixCharacteristics( MatrixCharacteristics mcIn, MatrixCharacteristics mcOut )
	{
		_mcIn = mcIn;
		_mcOut = mcOut;
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
		byte in = Byte.parseByte(parts[1]);
		long rows = UtilFunctions.toLong(Double.parseDouble(parts[2])); //save cast
		long cols = UtilFunctions.toLong(Double.parseDouble(parts[3])); //save cast
		boolean byrow = Boolean.parseBoolean(parts[4]);
		byte out = Byte.parseByte(parts[5]);
		 
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeMRInstruction: " + str);
		else
			return new MatrixReshapeMRInstruction(new Operator(true), in, rows, cols, byrow, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int brlen, int bclen )
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for(IndexedMatrixValue imv : blkList)
			{
				if( imv == null )
					continue;
				
				//get cached blocks
				ArrayList<IndexedMatrixValue> out = _cache;
				//ArrayList<IndexedMatrixValue> out = cachedValues.get(output);
	
				//process instruction
				IndexedMatrixValue in = imv;
				out = LibMatrixReorg.reshape(in, _mcIn.get_rows(), _mcIn.get_cols(), brlen, bclen,
						                     out, _rows, _cols, brlen, bclen, _byrow);
				
				//put the output values in the output cache
				for( IndexedMatrixValue outBlk : out )
					cachedValues.add(output, outBlk);
				
				//put blocks into own cache
				if( LibMatrixReorg.ALLOW_BLOCK_REUSE )
					_cache = out;	
			}
	}
	
	public long getNumRows()
	{
		return _rows;
	}
	
	public long getNumColunms()
	{
		return _cols;
	}
}
