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
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;

/**
 * 
 * 
 */
public class ReplicateInstruction extends UnaryMRInstructionBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _repCols = true;
	private long _lenM = -1; //clen/rlen
	
	public ReplicateInstruction(byte in, byte out, boolean repCols, long lenM, String istr)
	{
		super(null, in, out);
		mrtype = MRINSTRUCTION_TYPE.Reorg;
		instString = istr;
		
		_repCols = repCols;
		_lenM = lenM;
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
		//check instruction format
		InstructionUtils.checkNumFields ( str, 4 );
		
		//parse instruction
		String[] parts = InstructionUtils.getInstructionParts ( str );
		byte in = Byte.parseByte(parts[1]);
		boolean repCols = Boolean.parseBoolean(parts[2]);
		long len = Long.parseLong(parts[3]);
		byte out = Byte.parseByte(parts[4]);
		
		//construct instruction
		return new ReplicateInstruction(in, out, repCols, len, str);
	}

	/**
	 * 
	 */
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		
		if( blkList != null ) {
			for(IndexedMatrixValue in : blkList)
			{
				if( in==null ) continue;
				
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				//process instruction
				MatrixIndexes inIx = in.getIndexes();
				MatrixValue inVal = in.getValue();
				
				if( _repCols ) //replicate columns
				{
					if( inIx.getColumnIndex()>1 || inVal.getNumColumns()>1) //pass-through
					{
						//pass through of index/value (unnecessary rep); decision based on
						//if not column vector (MV binary cell operations only support col vectors)
						out.set(inIx, inVal);
					}
					else
					{
						//compute num additional replicates based on num column blocks lhs matrix
						//(e.g., M is Nx2700, blocksize=1000 -> numRep 2 because original block passed to index 1)
						if( blockColFactor<=1 ) //blocksize should be 1000 or similar
							LOG.warn("Block size of input matrix is: brlen="+blockRowFactor+", bclen="+blockColFactor+".");
						long numRep = (int)Math.ceil((double)_lenM / blockColFactor) - 1; 
						
						//replicate block (number of replicates is potentially unbounded, however,
						//because the vector is not modified we can passed the original data and
						//hence the memory overhead is very small)
						for( long i=0; i<numRep; i++ ){
							IndexedMatrixValue repV = cachedValues.holdPlace(output, valueClass);
							MatrixIndexes repIX= repV.getIndexes();
							repIX.setIndexes(inIx.getRowIndex(), 2+i);
							repV.set(repIX, inVal);
						}
						
						//output original block
						out.set(inIx, inVal);	
					}
				}
				else //replicate rows
				{
					if( inIx.getRowIndex()>1 || inVal.getNumRows()>1) //pass-through
					{
						//pass through of index/value (unnecessary rep); 
						out.set(inIx, inVal);
					}
					else
					{
						//compute num additional replicates based on num column blocks lhs matrix
						//(e.g., M is Nx2700, blocksize=1000 -> numRep 2 because original block passed to index 1)
						if( blockRowFactor<=1 ) //blocksize should be 1000 or similar
							LOG.warn("Block size of input matrix is: brlen="+blockRowFactor+", bclen="+blockColFactor+".");
						long numRep = (int)Math.ceil((double)_lenM / blockRowFactor) - 1; 
						
						//replicate block (number of replicates is potentially unbounded, however,
						//because the vector is not modified we can passed the original data and
						//hence the memory overhead is very small)
						for( long i=0; i<numRep; i++ ){
							IndexedMatrixValue repV = cachedValues.holdPlace(output, valueClass);
							MatrixIndexes repIX= repV.getIndexes();
							repIX.setIndexes(2+i, inIx.getColumnIndex());
							repV.set(repIX, inVal);
						}
						
						//output original block
						out.set(inIx, inVal);	
					}
				}
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
		}
	}
}
