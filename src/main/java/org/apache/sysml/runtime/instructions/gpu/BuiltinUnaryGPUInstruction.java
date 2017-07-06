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


package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;

public abstract class BuiltinUnaryGPUInstruction  extends GPUInstruction {


	int _arity;
	CPOperand _input;
	CPOperand _output;
	
	public BuiltinUnaryGPUInstruction(Operator op, CPOperand in, CPOperand out, int _arity, String opcode, String istr )
	{
		super(op, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.BuiltinUnary;
		this._arity = _arity;
		_input = in;
    _output = out;
	}

	public int getArity() {
		return _arity;
	}
	
	public static BuiltinUnaryGPUInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = null;
		ValueFunction func = null;
		
		//print or stop or cumulative aggregates
		if( parts.length==4 ) 
		{
			opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			func = Builtin.getBuiltinFnObject(opcode);
			
			throw new DMLRuntimeException("The instruction is not supported on GPU:" + str);
//			if( Arrays.asList(new String[]{"ucumk+","ucum*","ucummin","ucummax"}).contains(opcode) )
//				return new MatrixBuiltinCPInstruction(new UnaryOperator(func,Integer.parseInt(parts[3])), in, out, opcode, str); 
//			else
//				return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, opcode, str);
		}
		else //2+1, general case
		{
			InstructionUtils.checkNumFields(str, 2);
			opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			func = Builtin.getBuiltinFnObject(opcode);
			
			if(in.getDataType() == DataType.SCALAR)
				throw new DMLRuntimeException("The instruction is not supported on GPU:" + str);
//				return new ScalarBuiltinCPInstruction(new SimpleOperator(func), in, out, opcode, str);
			else if(in.getDataType() == DataType.MATRIX)
				return new MatrixBuiltinGPUInstruction(new UnaryOperator(func), in, out, opcode, str);
		}
		
		return null;
	}
}
