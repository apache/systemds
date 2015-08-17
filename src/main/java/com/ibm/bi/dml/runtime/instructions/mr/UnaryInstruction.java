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

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.UnaryOperator;


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
			UnaryOperator unary = new UnaryOperator(com.ibm.bi.dml.runtime.functionobjects.Builtin.getBuiltinFnObject(opcode));
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
