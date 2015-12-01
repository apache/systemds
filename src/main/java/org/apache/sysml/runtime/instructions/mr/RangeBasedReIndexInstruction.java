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
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReIndexOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;


public class RangeBasedReIndexInstruction extends UnaryMRInstructionBase
{
	
	protected boolean forLeftIndexing=false;
	protected long leftMatrixNRows=0;
	protected long leftMatrixNCols=0;
	
	private IndexRange indexRange=null;
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		indexRange=rng;
	}
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, boolean forleft, long leftNRows, long leftNCols, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		indexRange=rng;
		this.forLeftIndexing=forleft;
		this.leftMatrixNRows=leftNRows;
		this.leftMatrixNCols=leftNCols;
	}
	
	public IndexRange getIndexRange() {
		return indexRange;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 8 );
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		boolean forLeft=false;
		if(opcode.equalsIgnoreCase("rangeReIndexForLeft"))
			forLeft=true;
		else if(!opcode.equalsIgnoreCase("rangeReIndex"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		byte in = Byte.parseByte(parts[1]); 
		IndexRange rng=new IndexRange(UtilFunctions.parseToLong(parts[2]), 
									  UtilFunctions.parseToLong(parts[3]), 
									  UtilFunctions.parseToLong(parts[4]),
									  UtilFunctions.parseToLong(parts[5]));
		byte out = Byte.parseByte(parts[6]);
		long leftIndexingNrow=Long.parseLong(parts[7]);
		long leftIndexingNcol=Long.parseLong(parts[8]);
		
		//recalculate the index range for left indexing
		if(forLeft)
		{
			long a=rng.rowStart;
			long b=rng.colStart;
			rng.rowStart=2-a;
			rng.colStart=2-b;
			//don't need to extend to the whole left matrix dimension
			rng.rowEnd=leftIndexingNrow-a+1;
			rng.colEnd=leftIndexingNcol-b+1;
			return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, forLeft, leftIndexingNrow, leftIndexingNcol, str);
		}else
			return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, str);
	}
	
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		
		if(input==output)
			throw new DMLRuntimeException("input cannot be the same for output for "+this.instString);
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null ) {
			for(IndexedMatrixValue in : blkList) 
			{
				if( in == null )
					continue;
	
				//process instruction (incl block allocation)
				ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
				OperationsOnMatrixValues.performSlice(in, indexRange, blockRowFactor, blockColFactor, outlist);
		
				//put blocks into result cache
				for( IndexedMatrixValue ret : outlist )
					cachedValues.add(output, ret);
			}
		}
	}
}
