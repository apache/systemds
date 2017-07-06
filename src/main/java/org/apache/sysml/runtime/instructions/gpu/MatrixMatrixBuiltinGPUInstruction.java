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


public class MatrixMatrixBuiltinGPUInstruction extends BuiltinBinaryGPUInstruction {

  public MatrixMatrixBuiltinGPUInstruction(Operator op, CPOperand input1, CPOperand input2, CPOperand output, String opcode, String istr, int _arity) {
    super(op, input1, input2, output, opcode, istr, _arity);
    _gputype = GPUINSTRUCTION_TYPE.BuiltinUnary;

  }

  @Override
  public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
    GPUStatistics.incrementNoOfExecutedGPUInst();

    String opcode = getOpcode();
    MatrixObject mat1 = getMatrixInputForGPUInstruction(ec, input1.getName());
    MatrixObject mat2 = getMatrixInputForGPUInstruction(ec, input2.getName());

    if(opcode.equals("solve")) {
      ec.setMetaData(output.getName(), mat1.getNumColumns(), 1);
      LibMatrixCUDA.solve(ec, ec.getGPUContext(0), getExtendedOpcode(), mat1, mat2, output.getName());

    } else {
      throw new DMLRuntimeException("Unsupported GPU operator:" + opcode);
    }
    ec.releaseMatrixInputForGPUInstruction(input1.getName());
    ec.releaseMatrixInputForGPUInstruction(input2.getName());
    ec.releaseMatrixOutputForGPUInstruction(output.getName());
  }

}
