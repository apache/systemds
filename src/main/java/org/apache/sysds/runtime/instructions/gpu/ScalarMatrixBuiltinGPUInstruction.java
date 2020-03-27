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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.data.LibMatrixCuDNN;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.GPUStatistics;

public class ScalarMatrixBuiltinGPUInstruction extends BuiltinBinaryGPUInstruction {

	protected ScalarMatrixBuiltinGPUInstruction(Operator op, CPOperand input1, CPOperand input2, CPOperand output,
			String opcode, String istr, int _arity) {
		super(op, input1, input2, output, opcode, istr, _arity);
		_gputype = GPUINSTRUCTION_TYPE.BuiltinUnary;
	}

  @Override
  public void processInstruction(ExecutionContext ec) {
    GPUStatistics.incrementNoOfExecutedGPUInst();

    String opcode = getOpcode();
    CPOperand mat = ( input1.getDataType() == DataType.MATRIX ) ? input1 : input2;
	CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
	MatrixObject in1 = getMatrixInputForGPUInstruction(ec, mat.getName());
	ScalarObject constant = ec.getScalarInput(scalar);
    
    if(opcode.equals("max")) {
    	ec.setMetaData(output.getName(), in1.getNumRows(), in1.getNumColumns());
    	double constVal = constant.getDoubleValue();
    	if(constVal == 0)
    		LibMatrixCuDNN.relu(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, output.getName());
    	else
    		LibMatrixCUDA.matrixScalarOp(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, output.getName(), false, 
    				InstructionUtils.parseScalarBinaryOperator(opcode, false, constVal));
    } else if(opcode.equals("min")) {
    	ec.setMetaData(output.getName(), in1.getNumRows(), in1.getNumColumns());
    	double constVal = constant.getDoubleValue();
    	LibMatrixCUDA.matrixScalarOp(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, output.getName(), false, 
    				InstructionUtils.parseScalarBinaryOperator(opcode, false, constVal));
    } else {
      throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
    }
    ec.releaseMatrixInputForGPUInstruction(mat.getName());
    ec.releaseMatrixOutputForGPUInstruction(output.getName());
  }

}
