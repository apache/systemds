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

import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class BuiltinBinaryGPUInstruction extends GPUInstruction {

  @SuppressWarnings("unused")
  private int _arity;

  CPOperand output;
  CPOperand input1, input2;


  public BuiltinBinaryGPUInstruction(Operator op, CPOperand input1, CPOperand input2, CPOperand output, String opcode, String istr, int _arity) {
    super(op, opcode, istr);
    this._arity = _arity;
    this.output = output;
    this.input1 = input1;
    this.input2 = input2;
  }

  public static BuiltinBinaryGPUInstruction parseInstruction(String str) throws DMLRuntimeException {
    CPOperand in1 = new CPOperand("", Expression.ValueType.UNKNOWN, Expression.DataType.UNKNOWN);
    CPOperand in2 = new CPOperand("", Expression.ValueType.UNKNOWN, Expression.DataType.UNKNOWN);
    CPOperand out = new CPOperand("", Expression.ValueType.UNKNOWN, Expression.DataType.UNKNOWN);

    String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
    InstructionUtils.checkNumFields ( parts, 3 );

    String opcode = parts[0];
    in1.split(parts[1]);
    in2.split(parts[2]);
    out.split(parts[3]);

    // check for valid data type of output
    if((in1.getDataType() == Expression.DataType.MATRIX || in2.getDataType() == Expression.DataType.MATRIX) && out.getDataType() != Expression.DataType.MATRIX)
      throw new DMLRuntimeException("Element-wise matrix operations between variables " + in1.getName() +
              " and " + in2.getName() + " must produce a matrix, which " + out.getName() + " is not");

    // Determine appropriate Function Object based on opcode
    ValueFunction func = Builtin.getBuiltinFnObject(opcode);

    // Only for "solve"
    if ( in1.getDataType() == Expression.DataType.SCALAR && in2.getDataType() == Expression.DataType.SCALAR )
      throw new DMLRuntimeException("GPU : Unsupported GPU builtin operations on 2 scalars");
    else if ( in1.getDataType() == Expression.DataType.MATRIX && in2.getDataType() == Expression.DataType.MATRIX )
      return new MatrixMatrixBuiltinGPUInstruction(new BinaryOperator(func), in1, in2, out, opcode, str, 2);
    else
      throw new DMLRuntimeException("GPU : Unsupported GPU builtin operations on a matrix and a scalar");


  }

}
