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

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.PMMJ.CacheType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;


/**
 * 
 * 
 */
public class PMMJMRInstruction extends BinaryMRInstructionBase implements IDistributedCacheConsumer
{	
	
	private long _rlen = -1;
	private boolean _outputEmptyBlocks = true;
	
	
	public PMMJMRInstruction(Operator op, byte in1, byte in2, byte out, long nrow, CacheType ctype, boolean outputEmpty, String istr)
	{
		super(op, in1, in2, out);
		instString = istr;
		
		_rlen = nrow;
		_outputEmptyBlocks = outputEmpty;
		
		//NOTE: cache type only used by distributed cache input
	}
	
	public long getNumRows() {
		return _rlen;
	}
	
	public boolean getOutputEmptyBlocks() {
		return _outputEmptyBlocks;
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
		InstructionUtils.checkNumFields ( str, 6 );
		
		String[] parts = InstructionUtils.getInstructionParts(str);
		String opcode = parts[0];
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[4]);
		long nrow = Long.parseLong(parts[3]);
		CacheType ctype = CacheType.valueOf(parts[5]);
		boolean outputEmpty = Boolean.parseBoolean(parts[6]);
		
		if(!opcode.equalsIgnoreCase("pmm"))
			throw new DMLRuntimeException("Unknown opcode while parsing an PmmMRInstruction: " + str);
		
		return new PMMJMRInstruction(new Operator(true), in1, in2, out, nrow, ctype, outputEmpty, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		//get both matrix inputs (left side always permutation)
		DistributedCacheInput dcInput = MRBaseForCommonInstructions.dcValues.get(input1);
		IndexedMatrixValue in2 = cachedValues.getFirst(input2);
		IndexedMatrixValue in1 = dcInput.getDataBlock((int)in2.getIndexes().getRowIndex(), 1);
		MatrixBlock mb1 = (MatrixBlock)in1.getValue();
		MatrixBlock mb2 = (MatrixBlock)in2.getValue();
		
		//compute target block indexes
		long minPos = UtilFunctions.toLong( mb1.minNonZero() );
		long maxPos = UtilFunctions.toLong( mb1.max() );
		long rowIX1 = (minPos-1)/blockRowFactor+1;
		long rowIX2 = (maxPos-1)/blockRowFactor+1;
		boolean multipleOuts = (rowIX1 != rowIX2);
		
		if( minPos >= 1 ) //at least one row selected
		{
			//output sparsity estimate
			double spmb1 = OptimizerUtils.getSparsity(mb1.getNumRows(), 1, mb1.getNonZeros());
			long estnnz = (long) (spmb1 * mb2.getNonZeros());
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(blockRowFactor, mb2.getNumColumns(), estnnz);
			
			//compute and allocate output blocks
			IndexedMatrixValue out1 = cachedValues.holdPlace(output, valueClass);
			IndexedMatrixValue out2 = multipleOuts ? cachedValues.holdPlace(output, valueClass) : null;
			out1.getValue().reset(blockRowFactor, mb2.getNumColumns(), sparse);
			if( out2 != null )
				out2.getValue().reset(UtilFunctions.computeBlockSize(_rlen, rowIX2, blockRowFactor), mb2.getNumColumns(), sparse);
			
			//compute core matrix permutation (assumes that out1 has default blocksize, 
			//hence we do a meta data correction afterwards)
			mb1.permutationMatrixMultOperations(mb2, out1.getValue(), (out2!=null)?out2.getValue():null);
			((MatrixBlock)out1.getValue()).setNumRows(UtilFunctions.computeBlockSize(_rlen, rowIX1, blockRowFactor));
			out1.getIndexes().setIndexes(rowIX1, in2.getIndexes().getColumnIndex());
			if( out2 != null )
				out2.getIndexes().setIndexes(rowIX2, in2.getIndexes().getColumnIndex());
				
			//empty block output filter (enabled by compiler consumer operation is in CP)
			if( !_outputEmptyBlocks && out1.getValue().isEmpty() 
				&& (out2==null || out2.getValue().isEmpty() )  )
			{
				cachedValues.remove(output);
			}
		}
	}
	
	@Override //IDistributedCacheConsumer
	public boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		return (index==input1 && index!=input2);
	}
	
	@Override //IDistributedCacheConsumer
	public void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		indexes.add(input1);
	}
}
