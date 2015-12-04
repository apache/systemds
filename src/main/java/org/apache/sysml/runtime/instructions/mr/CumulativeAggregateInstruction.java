/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.instructions.mr;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class CumulativeAggregateInstruction extends AggregateUnaryInstruction 
{
	
	private MatrixCharacteristics _mcIn = null;
	
	public CumulativeAggregateInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out, true, istr);
	}
	
	public void setMatrixCharacteristics( MatrixCharacteristics mcIn )
	{
		_mcIn = mcIn;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseCumulativeAggregateUnaryOperator(opcode);
		
		return new CumulativeAggregateInstruction(aggun, in, out, str);
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
			
			//output allocation
			IndexedMatrixValue out = cachedValues.holdPlace(output, valueClass);
			
			//process instruction
			OperationsOnMatrixValues.performAggregateUnary( inix, in1.getValue(), out.getIndexes(), out.getValue(), 
					                            ((AggregateUnaryOperator)optr), blockRowFactor, blockColFactor);
			if( ((AggregateUnaryOperator)optr).aggOp.correctionExists )
				((MatrixBlock)out.getValue()).dropLastRowsOrColums(((AggregateUnaryOperator)optr).aggOp.correctionLocation);
			
			//cumsum expand partial aggregates
			long rlenOut = (long)Math.ceil((double)_mcIn.getRows()/blockRowFactor);
			long rixOut = (long)Math.ceil((double)inix.getRowIndex()/blockRowFactor);
			int rlenBlk = (int) Math.min(rlenOut-(rixOut-1)*blockRowFactor, blockRowFactor);
			int clenBlk = out.getValue().getNumColumns();
			int posBlk = (int) ((inix.getRowIndex()-1) % blockRowFactor);
			MatrixBlock outBlk = new MatrixBlock(rlenBlk, clenBlk, false);
			outBlk.copy(posBlk, posBlk, 0, clenBlk-1, (MatrixBlock) out.getValue(), true);
	
			MatrixIndexes outIx = out.getIndexes(); 
			outIx.setIndexes(rixOut, outIx.getColumnIndex());
			out.set(outIx, outBlk);		
		}
	}
}
