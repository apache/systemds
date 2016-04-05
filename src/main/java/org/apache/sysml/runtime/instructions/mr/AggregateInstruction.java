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
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class AggregateInstruction extends UnaryMRInstructionBase 
{
		
	public AggregateInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Aggregate;
		instString = istr;
	}
	
	public static AggregateInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		
		AggregateOperator agg = null;
		if(opcode.equalsIgnoreCase("ak+") || opcode.equalsIgnoreCase("asqk+")
				|| opcode.equalsIgnoreCase("amean") || opcode.equalsIgnoreCase("avar")) {
			InstructionUtils.checkNumFields ( str, 4 );
			agg = InstructionUtils.parseAggregateOperator(opcode, parts[3], parts[4]);
		}
		else {
			InstructionUtils.checkNumFields ( str, 2 );
			agg = InstructionUtils.parseAggregateOperator(opcode, null, null);	
		}
		
		return new AggregateInstruction(agg, in, out, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}

}
