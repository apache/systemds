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

package org.apache.sysds.runtime.instructions.gpu;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.data.LibMatrixCuDNN;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.GPUStatistics;
import org.apache.sysds.utils.stats.Timing;

public class MatrixBuiltinGPUInstruction extends BuiltinUnaryGPUInstruction {
	private static final Log LOG = LogFactory.getLog(MatrixBuiltinGPUInstruction.class.getName());
	
	protected MatrixBuiltinGPUInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(op, in, out, 1, opcode, instr);
		_gputype = GPUINSTRUCTION_TYPE.BuiltinUnary;
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();

		String opcode = getOpcode();
		MatrixObject mat = getMatrixInputForGPUInstruction(ec, _input1.getName());
		if(opcode != "ucumk+*")
			ec.setMetaData(_output.getName(), mat.getNumRows(), mat.getNumColumns());

		Timing time = new Timing(true);
		switch(opcode) {
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
			case "sinh":
				LibMatrixCUDA.sinh(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "cosh":
				LibMatrixCUDA.cosh(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "tanh":
				LibMatrixCUDA.tanh(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "asin":
				LibMatrixCUDA.asin(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "acos":
				LibMatrixCUDA.acos(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "atan":
				LibMatrixCUDA.atan(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "sign":
				LibMatrixCUDA.sign(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "sigmoid":
				LibMatrixCUDA.sigmoid(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "softmax":
				LibMatrixCuDNN.softmax(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName()); break;
			case "ucumk+":
				LibMatrixCUDA.cumulativeScan(ec, ec.getGPUContext(0), getExtendedOpcode(), "cumulative_sum", mat,
						_output.getName());
				break;
			case "ucum*":
				LibMatrixCUDA.cumulativeScan(ec, ec.getGPUContext(0), getExtendedOpcode(), "cumulative_prod", mat,
						_output.getName());
				break;
			case "ucumk+*":
				ec.setMetaData(_output.getName(), mat.getNumRows(), 1);
				LibMatrixCUDA.cumulativeSumProduct(ec, ec.getGPUContext(0), getExtendedOpcode(), "cumulative_sum_prod",
						mat, _output.getName());
				break;
			case "ucummin":
				LibMatrixCUDA.cumulativeScan(ec, ec.getGPUContext(0), getExtendedOpcode(), "cumulative_min", mat,
						_output.getName());
				break;
			case "ucummax":
				LibMatrixCUDA.cumulativeScan(ec, ec.getGPUContext(0), getExtendedOpcode(), "cumulative_max", mat,
						_output.getName());
				break;
			default:
				throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
		}

		if(LOG.isTraceEnabled())
		{
			double duration = time.stop();
			LOG.trace("processInstruction() " + getExtendedOpcode() + " executed in " + duration + "ms.");
		}

		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}