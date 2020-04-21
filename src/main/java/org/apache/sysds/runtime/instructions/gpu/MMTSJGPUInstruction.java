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

import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.GPUStatistics;

public class MMTSJGPUInstruction extends GPUInstruction {
	private MMTSJType _type = null;
	CPOperand _input;
	CPOperand _output;

	/**
	 * MMTSJGPUInstruction constructor.
	 * 
	 * @param op
	 *            operator
	 * @param in1
	 *            input
	 * @param type
	 *            left/right, left-&gt; A' %*% A, right-&gt; A %*% A'
	 * @param out
	 *            output
	 * @param opcode
	 *            the opcode
	 * @param istr
	 *            ?
	 */
	private MMTSJGPUInstruction(Operator op, CPOperand in1, MMTSJType type, CPOperand out, String opcode, String istr) {
		super(op, opcode, istr);
		_gputype = GPUINSTRUCTION_TYPE.MMTSJ;
		_type = type;
		_input = in1;
		_output = out;
	}

        public static MMTSJGPUInstruction parseInstruction ( String str )
        {
                String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
                InstructionUtils.checkNumFields ( parts, 3 );

                String opcode = parts[0];
                CPOperand in1 = new CPOperand(parts[1]);
                CPOperand out = new CPOperand(parts[2]);
                MMTSJType titype = MMTSJType.valueOf(parts[3]);

                if(!opcode.equalsIgnoreCase("tsmm"))
                        throw new DMLRuntimeException("Unknown opcode while parsing an MMTSJGPUInstruction: " + str);
                else
                        return new MMTSJGPUInstruction(new Operator(true), in1, titype, out, opcode, str);
        }

        @Override
        public void processInstruction(ExecutionContext ec) {
                GPUStatistics.incrementNoOfExecutedGPUInst();
                MatrixObject mat = getMatrixInputForGPUInstruction(ec, _input.getName());
                boolean isLeftTransposed = ( _type == MMTSJType.LEFT);
                int rlen = (int) (isLeftTransposed? mat.getNumColumns() : mat.getNumRows());
                int clen = rlen;
                //execute operations 
                ec.setMetaData(_output.getName(), rlen, clen);
                LibMatrixCUDA.matmultTSMM(ec, ec.getGPUContext(0), getExtendedOpcode(), mat, _output.getName(), isLeftTransposed);
                ec.releaseMatrixInputForGPUInstruction(_input.getName());
                ec.releaseMatrixOutputForGPUInstruction(_output.getName());
        }

        public MMTSJType getMMTSJType() {
                return _type;
        }
}
