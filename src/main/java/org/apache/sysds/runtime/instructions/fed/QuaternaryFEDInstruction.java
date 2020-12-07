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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.fed.QuaternaryWCeMMFEDInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

public abstract class QuaternaryFEDInstruction extends ComputationFEDInstruction
{
  protected CPOperand _input4 = null;
  // TODO: use this variables when adding cache
  // protected boolean _cacheU = false;
  // protected boolean _cacheV = false;

  protected QuaternaryFEDInstruction(FEDInstruction.FEDType type, Operator operator,
    CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str)
  {
    // TODO: there are only 3 inputs in ComputationFEDInstruction. Add one?
    super(type, operator, in1, in2, in3, out, opcode, instruction_str);
    _input4 = in4;
    // TODO: assign the following variables when adding cache
    // _cacheU = cacheU;
    // _cacheV = cacheV;
  }

  public static QuaternaryFEDInstruction parseInstruction(String str)
  {
    String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
    String opcode = parts[0];

    // TODO: can we assume the same opcodes as for quaternarySP?
    if(!InstructionUtils.isDistQuaternaryOpcode(opcode))
    {
      throw new DMLRuntimeException("QuaternaryFED.parseInstruction(): Unknown opcode " + opcode);
    }

    int add_input_4 = (opcode.endsWith("wcemm")) ? 1 : 0;

    CPOperand in1 = new CPOperand(parts[1]);
    CPOperand in2 = new CPOperand(parts[2]);
    CPOperand in3 = new CPOperand(parts[3]);
    CPOperand out = new CPOperand(parts[4] + add_input_4);

    InstructionUtils.checkNumFields(parts, 5 + add_input_4);

    // TODO: can the output data type be checked here?

    // TODO: parse Operator here
    // Operator operator =

    // TODO: add cacheU and cacheV and other paramters in instruction?

    if(opcode.endsWith("wcemm"))
    {
      CPOperand in4 = new CPOperand(parts[4]);
      checkDataTypes(in1, in2, in3, in4);

      final WCeMMType wcemm_type = WCeMMType.valueOf(parts[6]);
      QuaternaryOperator quaternary_operator = (wcemm_type.hasFourInputs() ? new QuaternaryOperator(wcemm_type, Double.parseDouble(in4.getName())) : new QuaternaryOperator(wcemm_type));

      return new QuaternaryWCeMMFEDInstruction(quaternary_operator, in1, in2, in3, in4, out, opcode, str);
    }

    assert false: "Not implemented yet!\n";

    return null;
  }

  protected static void checkDataTypes(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4)
  {
    if(in1.getDataType() != DataType.MATRIX || in2.getDataType() != DataType.MATRIX || in3.getDataType() != DataType.MATRIX || in4.getDataType() != DataType.MATRIX)
    {
      throw new DMLRuntimeException("Federated quaternary operations supported with matrices only yet");
    }
  }

}
