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
import org.apache.sysml.runtime.functionobjects.ValueFunctionWithConstant;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class PlusMultInstruction extends BinaryInstruction 
{
	public PlusMultInstruction(Operator op, byte in1, byte in2, byte out, String istr) {
		super(op, in1, in2, out, istr);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static PlusMultInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		String opcode = parts[0];
		byte in1 = Byte.parseByte(parts[1]);
		double scalar = Double.parseDouble(parts[2]);
		byte in2 = Byte.parseByte(parts[3]);
		byte out = Byte.parseByte(parts[4]);
		
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(opcode);
		((ValueFunctionWithConstant) bop.fn).setConstant(scalar);
		return new PlusMultInstruction(bop, in1, in2, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{
		//default binary mr instruction execution (custom logic encoded in operator)
		super.processInstruction(valueClass, cachedValues, tempValue, zeroInput, blockRowFactor, blockColFactor);
	}
}
