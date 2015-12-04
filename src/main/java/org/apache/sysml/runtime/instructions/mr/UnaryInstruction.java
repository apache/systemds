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
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


public class UnaryInstruction extends UnaryMRInstructionBase 
{
	
	public UnaryInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Unary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String opcode = InstructionUtils.getOpCode(str);
	 
		InstructionUtils.checkNumFields ( str, 2 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		byte in, out;
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		// Currently, UnaryInstructions are used primarily for executing Builtins like SIN(A), LOG(A,2)
		if ( InstructionUtils.isBuiltinFunction(opcode) ) {
			UnaryOperator unary = new UnaryOperator(Builtin.getBuiltinFnObject(opcode));
			return new UnaryInstruction(unary, in, out, str);
		}
		else 
			return new UnaryInstruction(null, in, out, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		
		if( blkList != null )
			for(IndexedMatrixValue in : blkList )
			{
				if(in==null)
					continue;
				
				//allocate space for the output value
				IndexedMatrixValue out;
				if(input==output)
					out=tempValue;
				else
					out=cachedValues.holdPlace(output, valueClass);
				
				//process instruction
				out.getIndexes().setIndexes(in.getIndexes());
				OperationsOnMatrixValues.performUnaryIgnoreIndexes(in.getValue(), out.getValue(), (UnaryOperator)optr);
				
				//put the output value in the cache
				if(out==tempValue)
					cachedValues.add(output, out);
			}
	}

}
