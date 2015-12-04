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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class BinaryInstruction extends BinaryMRInstructionBase 
{
	
	public BinaryInstruction(Operator op, byte in1, byte in2, byte out, String istr)
	{
		super(op, in1, in2, out);
		mrtype = MRINSTRUCTION_TYPE.ArithmeticBinary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, out;
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		in2 = Byte.parseByte(parts[2]);
		out = Byte.parseByte(parts[3]);
		
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(opcode);
		if( bop != null )
			return new BinaryInstruction(bop, in1, in2, out, str);
		else
			return null;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in1=cachedValues.getFirst(input1);
		IndexedMatrixValue in2=cachedValues.getFirst(input2);
		if(in1==null && in2==null)
			return;
		
		//allocate space for the output value
		//try to avoid coping as much as possible
		IndexedMatrixValue out;
		if( (output!=input1 && output!=input2)
			|| (output==input1 && in1==null)
			|| (output==input2 && in2==null) )
			out=cachedValues.holdPlace(output, valueClass);
		else
			out=tempValue;
		
		//if one of the inputs is null, then it is a all zero block
		MatrixIndexes finalIndexes=null;
		if(in1==null)
		{
			in1=zeroInput;
			in1.getValue().reset(in2.getValue().getNumRows(), 
					in2.getValue().getNumColumns());
			finalIndexes=in2.getIndexes();
		}else
			finalIndexes=in1.getIndexes();
		
		if(in2==null)
		{
			in2=zeroInput;
			in2.getValue().reset(in1.getValue().getNumRows(), 
					in1.getValue().getNumColumns());
		}
		
		//process instruction
		out.getIndexes().setIndexes(finalIndexes);
		OperationsOnMatrixValues.performBinaryIgnoreIndexes(in1.getValue(), 
				in2.getValue(), out.getValue(), ((BinaryOperator)optr));
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.add(output, out);
		
	}

}
