/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;

/**
 * 
 * 
 */
public class ReplicateInstruction extends UnaryMRInstructionBase 
{
	
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
	 * @param mcIn
	 * @param mcOut
	 */
	public void computeOutputDimension(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut)
	{
		if( _repCols )
			mcOut.set(mcIn.getRows(), _lenM, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), mcIn.getCols());
		else
			mcOut.set(_lenM, mcIn.getCols(), mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), mcIn.getRows());
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
					//compute num additional replicates based on num column blocks lhs matrix
					//(e.g., M is Nx2700, blocksize=1000 -> numRep 2 because original block passed to index 1)
					if( blockColFactor<=1 ) //blocksize should be 1000 or similar
						LOG.warn("Block size of input matrix is: brlen="+blockRowFactor+", bclen="+blockColFactor+".");
					long numRep = (long)Math.ceil((double)_lenM / blockColFactor) - 1; 
					
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
				else //replicate rows
				{
					//compute num additional replicates based on num column blocks lhs matrix
					//(e.g., M is Nx2700, blocksize=1000 -> numRep 2 because original block passed to index 1)
					if( blockRowFactor<=1 ) //blocksize should be 1000 or similar
						LOG.warn("Block size of input matrix is: brlen="+blockRowFactor+", bclen="+blockColFactor+".");
					long numRep = (long)Math.ceil((double)_lenM / blockRowFactor) - 1; 
					
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
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
		}
	}
}
