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

package org.apache.sysds.lops;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class Tee extends Lop {

    public static final String OPCODE = "tee";
    /**
     * Constructor to be invoked by base class.
     *
     * @param input1  lop type
     * @param dt data type of the output
     * @param vt value type of the output
     */
    public Tee(Lop input1, DataType dt, ValueType vt) {
        super(Lop.Type.Tee, dt, vt);
        this.addInput(input1);
        input1.addOutput(this);
        lps.setProperties(inputs, Types.ExecType.OOC);
    }

    @Override
    public String toString() {
        return "Operation = Tee";
    }

    @Override
    public String getInstructions(String input1, String outputs) {

        String output2 = outputs + "_copy";
        System.out.println("DEBUG: Tee.getInstructions() called with:");
        System.out.println("  input1: " + input1);
        System.out.println("  output1: " + outputs);
        System.out.println("  This Tee node: " + this);
//        System.out.println("  Stack trace:");
//        Thread.dumpStack();

        // Return a temporary instruction to see if this fixes the empty string
        System.out.println("OOC" + OPERAND_DELIMITOR + "tee" + OPERAND_DELIMITOR + input1 + OPERAND_DELIMITOR + outputs + OPERAND_DELIMITOR + "TEMP_OUTPUT2");
        // This method generates the instruction string: OOC°tee°input°output1°output2...
        String ret = InstructionUtils.concatOperands(
                getExecType().name(), OPCODE,
                getInputs().get(0).prepInputOperand(input1),
                prepOutputOperand(outputs),
                prepInputOperand(output2)
        );
        System.out.println("DEBUG: Tee.getInstructions() returned: " + ret);

        return ret;
    }
}
