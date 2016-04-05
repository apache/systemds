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


package org.apache.sysml.runtime.instructions.spark;

import org.apache.sysml.lops.BinaryM.VectorType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class MatrixBVectorRelationalSPInstruction extends RelationalBinarySPInstruction 
{
	
	private VectorType _vtype = null;
	
	public MatrixBVectorRelationalSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, VectorType vtype, String opcode, String istr) 
		throws DMLRuntimeException
	{
		super(op, in1, in2, out, opcode, istr);
	
		_vtype = vtype;
		
		//sanity check opcodes
		if(!( opcode.equalsIgnoreCase("map==") || opcode.equalsIgnoreCase("map!=") || opcode.equalsIgnoreCase("map<")
			||opcode.equalsIgnoreCase("map>") || opcode.equalsIgnoreCase("map<=") || opcode.equalsIgnoreCase("map>=")) ) 
		{
			throw new DMLRuntimeException("Unknown opcode in MatrixBVectorRelationalSPInstruction: " + toString());
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{	
		//common binary matrix-broadcast vector process instruction
		super.processMatrixBVectorBinaryInstruction(ec, _vtype);
	}
}
