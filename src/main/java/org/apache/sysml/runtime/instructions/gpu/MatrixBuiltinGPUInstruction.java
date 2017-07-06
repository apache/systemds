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

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

public class MatrixBuiltinGPUInstruction extends BuiltinUnaryGPUInstruction {
	
	public MatrixBuiltinGPUInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr){
		super(op, in, out, 1, opcode, instr);
		_gputype = GPUINSTRUCTION_TYPE.BuiltinUnary;
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();
		
		String opcode = getOpcode();
		MatrixObject mat = getMatrixInputForGPUInstruction(ec, _input.getName());
		ec.setMetaData(_output.getName(), mat.getNumRows(), mat.getNumColumns());

		switch(opcode) {
			case "sel+":
				LibMatrixCUDA.relu(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "exp":
				LibMatrixCUDA.exp(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "sqrt":
				LibMatrixCUDA.sqrt(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "log":
				LibMatrixCUDA.log(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "round":
				LibMatrixCUDA.round(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "floor":
				LibMatrixCUDA.floor(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "ceil":
				LibMatrixCUDA.ceil(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "abs":
				LibMatrixCUDA.abs(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "sin":
				LibMatrixCUDA.sin(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "cos":
				LibMatrixCUDA.cos(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "tan":
				LibMatrixCUDA.tan(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "asin":
				LibMatrixCUDA.asin(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "acos":
				LibMatrixCUDA.acos(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "atan":
				LibMatrixCUDA.atan(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "sign":
				LibMatrixCUDA.sign(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			default:
				throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
		}
		ec.releaseMatrixInputForGPUInstruction(_input.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}