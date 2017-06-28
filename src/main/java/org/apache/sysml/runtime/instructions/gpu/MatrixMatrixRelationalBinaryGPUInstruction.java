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
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

public class MatrixMatrixRelationalBinaryGPUInstruction extends RelationalBinaryGPUInstruction {

	public MatrixMatrixRelationalBinaryGPUInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();

		MatrixObject in1 = getMatrixInputForGPUInstruction(ec, _input1.getName());
		MatrixObject in2 = getMatrixInputForGPUInstruction(ec, _input2.getName());

		long rlen1 = in1.getNumRows();
		long clen1 = in1.getNumColumns();
		long rlen2 = in2.getNumRows();
		long clen2 = in2.getNumColumns();

		// Assume ordinary binary op
		long rlen = rlen1;
		long clen = clen1;

		// Outer binary op ( [100,1] + [1,100] or [100,100] + [100,1]
		if (rlen1 != rlen2 || clen1 != clen2){
			rlen = rlen1 > rlen2 ? rlen1 : rlen2;
			clen = clen1 > clen2 ? clen1 : clen2;
		}

		ec.setMetaData(_output.getName(), (int)rlen, (int)clen);

		BinaryOperator bop = (BinaryOperator) _optr;
		LibMatrixCUDA.matrixMatrixRelational(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, in2, _output.getName(), bop);

		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixInputForGPUInstruction(_input2.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}
