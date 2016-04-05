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
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;


public class CumulativeOffsetInstruction extends BinaryInstruction 
{
	
	private BinaryOperator _bop = null;
	private UnaryOperator _uop = null;
	
	public CumulativeOffsetInstruction(byte in1, byte in2, byte out, String opcode, String istr)
	{
		super(null, in1, in2, out, istr);

		if( "bcumoffk+".equals(opcode) ) {
			_bop = new BinaryOperator(Plus.getPlusFnObject());
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucumk+"));
		}
		else if( "bcumoff*".equals(opcode) ){
			_bop = new BinaryOperator(Multiply.getMultiplyFnObject());
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucum*"));	
		}
		else if( "bcumoffmin".equals(opcode) ){
			_bop = new BinaryOperator(Builtin.getBuiltinFnObject("min"));
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummin"));	
		}
		else if( "bcumoffmax".equals(opcode) ){
			_bop = new BinaryOperator(Builtin.getBuiltinFnObject("max"));
			_uop = new UnaryOperator(Builtin.getBuiltinFnObject("ucummax"));	
		}
	}
	
	public static CumulativeOffsetInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[3]);
		
		return new CumulativeOffsetInstruction(in1, in2, out, opcode, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{	
		IndexedMatrixValue in1 = cachedValues.getFirst(input1); //original data 
		IndexedMatrixValue in2 = cachedValues.getFirst(input2); //offset row vector
				
		if( in1 == null || in2 == null ) 
			throw new DMLRuntimeException("Unexpected empty input (left="+((in1==null)?"null":in1.getIndexes())
					                                     +", right="+((in2==null)?"null":in2.getIndexes())+").");
		
		//prepare inputs and outputs
		IndexedMatrixValue out = cachedValues.holdPlace(output, valueClass);
		MatrixBlock data = (MatrixBlock) in1.getValue();
		MatrixBlock offset = (MatrixBlock) in2.getValue();
		MatrixBlock blk = (MatrixBlock) out.getValue();
		blk.reset(data.getNumRows(), data.getNumColumns());
		
		//blockwise offset aggregation and prefix sum computation
		MatrixBlock data2 = new MatrixBlock(data); //cp data
		MatrixBlock fdata2 = data2.sliceOperations(0, 0, 0, data2.getNumColumns()-1, new MatrixBlock()); //1-based
		fdata2.binaryOperationsInPlace(_bop, offset); //sum offset to first row
		data2.copy(0, 0, 0, data2.getNumColumns()-1, fdata2, true); //0-based
		data2.unaryOperations(_uop, blk); //compute columnwise prefix sums/prod/min/max

		//set output indexes
		out.getIndexes().setIndexes(in1.getIndexes());		
	}
}
