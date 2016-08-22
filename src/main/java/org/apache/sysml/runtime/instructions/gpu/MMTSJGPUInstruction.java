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

/*
 * Parses and processes the MMTSJ GPU Instruction
 * @function	GPUInstruction(...)
 * @function	parseInstruction(...)
 * @function	processInstruction(...)
 * @function	getMMTSJType(...)
 */
import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.Statistics;

public class MMTSJGPUInstruction extends GPUInstruction
{

        private MMTSJType _type = null;
        
        CPOperand _input;
        CPOperand _output;

        /**
         * @param op	operator
         * @param in1	input
         * @param type	left/right, left-> A' %*% A, right-> A %*% A'
         * @param out	output
         * @param opcode
         * @param istr
         */
        public MMTSJGPUInstruction(Operator op, CPOperand in1, MMTSJType type, CPOperand out,  String opcode, String istr)
        {
                super(op, opcode, istr);
                _gputype = GPUINSTRUCTION_TYPE.MMTSJ;
                _type = type;
                _input = in1;
                _output = out;
        }

        /**
         * parse MMTSJ GPU instruction
         * @param str
         * @return
         * @throws DMLRuntimeException
         */
        public static MMTSJGPUInstruction parseInstruction ( String str )
        	throws DMLRuntimeException
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

        /**
         * process MMTSJ GPU instruction 
         * @param ec	execution context
         * @throws DMLRuntimeException
         */
        @Override
        public void processInstruction(ExecutionContext ec)
                throws DMLRuntimeException
        {
                Statistics.incrementNoOfExecutedGPUInst();

                //get input
                MatrixObject mat = ec.getMatrixInputForGPUInstruction(_input.getName());
               
                boolean isLeftTransposed = ( _type == MMTSJType.LEFT);

                int rlen = (int) (isLeftTransposed? mat.getNumColumns() : mat.getNumRows());
                int clen = rlen;

                //execute operations 
                ec.setMetaData(_output.getName(), rlen, clen);
                MatrixObject out = ec.getDenseMatrixOutputForGPUInstruction(_output.getName());
                LibMatrixCUDA.matmultTSMM(mat, out, isLeftTransposed);
                
                ec.releaseMatrixInputForGPUInstruction(_input.getName());
                ec.releaseMatrixOutputForGPUInstruction(_output.getName());
        }

        /**
         * returns left/right depending on the type of MMTSJ instruction
         * @return _type
         */
        public MMTSJType getMMTSJType()
        {
                return _type;
        }
}