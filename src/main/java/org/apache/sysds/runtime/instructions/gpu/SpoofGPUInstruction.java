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

import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.lineage.LineageTraceable;


import java.util.ArrayList;

public class SpoofGPUInstruction extends GPUInstruction implements LineageTraceable {
    private final SpoofOperator _op;
    private final CPOperand[] _in;

    public final CPOperand output;
//	public final CPOperand input1, input2, input3;

    private SpoofGPUInstruction(SpoofOperator op, CPOperand[] in, CPOperand out, String opcode, String istr) {
        super(null, opcode, istr);
        _op = op;
        _in = in;
        output = out;
    }

    public static SpoofGPUInstruction parseInstruction(String str) {
        String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

        ArrayList<CPOperand> inlist = new ArrayList<>();
        Class<?> cla = CodegenUtils.getClass(parts[1]);
        SpoofOperator op = CodegenUtils.createInstance(cla);
        String opcode =  parts[0] + op.getSpoofType();

        for( int i=2; i<parts.length-2; i++ )
            inlist.add(new CPOperand(parts[i]));
        CPOperand out = new CPOperand(parts[parts.length-2]);
        int k = Integer.parseInt(parts[parts.length-1]);

        return new SpoofGPUInstruction(op, inlist.toArray(new CPOperand[0]), out, opcode, str);
    }

    @Override
    public void processInstruction(ExecutionContext ec) {

    }

    @Override
    public LineageItem[] getLineageItems(ExecutionContext ec) {
        return new LineageItem[]{new LineageItem(output.getName(),
                getOpcode(), LineageItemUtils.getLineage(ec, _in))};
    }

    //transpose-self matrix multiply
    public static native boolean execCellWise(double[] mat, double[] side_input, double[] scalars, int m, int n, long grix, int rix, int cix);

}

