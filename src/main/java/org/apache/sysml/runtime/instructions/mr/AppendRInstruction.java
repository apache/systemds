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
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class AppendRInstruction extends AppendInstruction 
{
	public AppendRInstruction(Operator op, byte in1, byte in2, byte out, boolean cbind, String istr)
	{
		super(op, in1, in2, out, cbind, istr);
	}

	public static AppendRInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionParts ( str );
		InstructionUtils.checkNumFields(parts, 4);
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte out = Byte.parseByte(parts[3]);
		boolean cbind = Boolean.parseBoolean(parts[4]);
			
		return new AppendRInstruction(null, in1, in2, out, cbind, str);
	}
	
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int brlen, int bclen)
			throws DMLRuntimeException 
	{		
		//get both inputs
		IndexedMatrixValue left = cachedValues.getFirst(input1);
		IndexedMatrixValue right = cachedValues.getFirst(input2);

		//check non-existing block
		if( left == null || right == null )
			throw new DMLRuntimeException("Missing append input: isNull(left): " + (left==null) + ", isNull(right): " + (right==null));
		
		//core append operation
		MatrixBlock mbLeft = (MatrixBlock)left.getValue();
		MatrixBlock mbRight = (MatrixBlock)right.getValue();
		
		MatrixBlock ret = mbLeft.appendOperations(mbRight, new MatrixBlock(), _cbind);
		
		//put result into cache
		cachedValues.add(output, new IndexedMatrixValue(left.getIndexes(), ret));
	}
}
