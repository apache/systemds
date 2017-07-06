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
import org.apache.sysml.runtime.functionobjects.IndexFunction;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

/**
 * Implements aggregate unary instructions for CUDA
 */
public class AggregateUnaryGPUInstruction extends GPUInstruction {
  private CPOperand _input1 = null;
  private CPOperand _output = null;

  public AggregateUnaryGPUInstruction(Operator op, CPOperand in1, CPOperand out,
                                       String opcode, String istr)
  {
    super(op, opcode, istr);
    _gputype = GPUINSTRUCTION_TYPE.AggregateUnary;
    _input1 = in1;
    _output = out;
  }

  public static AggregateUnaryGPUInstruction parseInstruction(String str )
          throws DMLRuntimeException
  {
    String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
    String opcode = parts[0];
    CPOperand in1 = new CPOperand(parts[1]);
    CPOperand out = new CPOperand(parts[2]);

    // This follows logic similar to AggregateUnaryCPInstruction.
    // nrow, ncol & length should either read or refresh metadata
    Operator aggop = null;
    if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")) {
      throw new DMLRuntimeException("nrow, ncol & length should not be compiled as GPU instructions!");
    } else {
      aggop = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
    }
    return new AggregateUnaryGPUInstruction(aggop, in1, out, opcode, str);
  }

  @Override
  public void processInstruction(ExecutionContext ec)
          throws DMLRuntimeException
  {
    GPUStatistics.incrementNoOfExecutedGPUInst();

    String opcode = getOpcode();

    // nrow, ncol & length should either read or refresh metadata
    if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") || opcode.equalsIgnoreCase("length")) {
      throw new DMLRuntimeException("nrow, ncol & length should not be compiled as GPU instructions!");
    }

    //get inputs
    MatrixObject in1 = getMatrixInputForGPUInstruction(ec, _input1.getName());

    int rlen = (int)in1.getNumRows();
    int clen = (int)in1.getNumColumns();

    IndexFunction indexFunction = ((AggregateUnaryOperator) _optr).indexFn;
    if (indexFunction instanceof ReduceRow){  // COL{SUM, MAX...}
      ec.setMetaData(_output.getName(), 1, clen);
    } else if (indexFunction instanceof ReduceCol) { // ROW{SUM, MAX,...}
      ec.setMetaData(_output.getName(), rlen, 1);
    }

    LibMatrixCUDA.unaryAggregate(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, _output.getName(), (AggregateUnaryOperator)_optr);

    //release inputs/outputs
    ec.releaseMatrixInputForGPUInstruction(_input1.getName());

    // If the unary aggregate is a row reduction or a column reduction, it results in a vector
    // which needs to be released. Otherwise a scala is produced and it is copied back to the host
    // and set in the execution context by invoking the setScalarOutput
    if (indexFunction instanceof ReduceRow || indexFunction instanceof ReduceCol) {
      ec.releaseMatrixOutputForGPUInstruction(_output.getName());
    }
  }

}
