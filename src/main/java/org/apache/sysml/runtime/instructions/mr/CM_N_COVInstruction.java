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
import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.functionobjects.COV;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.COVOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class CM_N_COVInstruction extends UnaryMRInstructionBase 
{
	
	public CM_N_COVInstruction(Operator op, byte in, byte out, String istr)
	{
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.CM_N_COV;
		instString = istr;
	}
	
	public static CM_N_COVInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		int cst;
		String opcode = parts[0];
		
		if (opcode.equalsIgnoreCase("cm") ) 
		{
			in = Byte.parseByte(parts[1]);
			cst = Integer.parseInt(parts[2]);
			out = Byte.parseByte(parts[3]);
			
			if(cst>4 || cst<0 || cst==1)
				throw new DMLRuntimeException("constant for central moment has to be 0, 2, 3, or 4");
			
			AggregateOperationTypes opType = CMOperator.getCMAggOpType(cst);
			CMOperator cm = new CMOperator(CM.getCMFnObject(opType), opType);
			return new CM_N_COVInstruction(cm, in, out, str);
		}else if(opcode.equalsIgnoreCase("cov"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			COVOperator cov = new COVOperator(COV.getCOMFnObject());
			return new CM_N_COVInstruction(cov, in, out, str);
		}/*else if(opcode.equalsIgnoreCase("mean"))
		{
			in = Byte.parseByte(parts[1]);
			out = Byte.parseByte(parts[2]);
			
			CMOperator mean = new CMOperator(CM.getCMFnObject(), CMOperator.AggregateOperationTypes.MEAN);
			return new CM_N_COVInstruction(mean, in, out, str);
		}*/
		else
			throw new DMLRuntimeException("unknown opcode "+opcode);
		
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		throw new DMLRuntimeException("no processInstruction for AggregateInstruction!");
		
	}
}
