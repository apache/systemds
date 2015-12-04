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
import org.apache.sysml.runtime.functionobjects.ReduceDiag;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class AggregateUnaryInstruction extends UnaryMRInstructionBase 
{
	
	private boolean _dropCorr = false;
	
	public AggregateUnaryInstruction(Operator op, byte in, byte out, boolean dropCorr, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.AggregateUnary;
		instString = istr;
		
		_dropCorr = dropCorr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		byte in = Byte.parseByte(parts[1]);
		byte out = Byte.parseByte(parts[2]);
		boolean drop = Boolean.parseBoolean(parts[3]);
		
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		return new AggregateUnaryInstruction(aggun, in, out, drop, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null )
			for(IndexedMatrixValue in: blkList)
			{
				if(in==null)
					continue;
			
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				MatrixIndexes inix = in.getIndexes();
				
				//prune unnecessary blocks for trace
				if( (((AggregateUnaryOperator)optr).indexFn instanceof ReduceDiag && inix.getColumnIndex()!=inix.getRowIndex()) )
				{
					//do nothing (block not on diagonal); but reset
					out.getValue().reset();
				}
				else //general case
				{
					//process instruction
					AggregateUnaryOperator auop = (AggregateUnaryOperator)optr;
					OperationsOnMatrixValues.performAggregateUnary( inix, in.getValue(), out.getIndexes(), out.getValue(), 
							                            auop, blockRowFactor, blockColFactor);
					if( _dropCorr )
						((MatrixBlock)out.getValue()).dropLastRowsOrColums(auop.aggOp.correctionLocation);
				}
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
	}

}
