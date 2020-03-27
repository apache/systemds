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
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.GPUStatistics;

public class ScalarMatrixRelationalBinaryGPUInstruction extends RelationalBinaryGPUInstruction {

	protected ScalarMatrixRelationalBinaryGPUInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();

		CPOperand mat = ( _input1.getDataType() == DataType.MATRIX ) ? _input1 : _input2;
		CPOperand scalar = ( _input1.getDataType() == DataType.MATRIX ) ? _input2 : _input1;
		MatrixObject in1 = getMatrixInputForGPUInstruction(ec, mat.getName());
		ScalarObject constant = ec.getScalarInput(scalar);

		int rlen = (int) in1.getNumRows();
		int clen = (int) in1.getNumColumns();
		ec.setMetaData(_output.getName(), rlen, clen);

		ScalarOperator sc_op = (ScalarOperator) _optr;
		sc_op = sc_op.setConstant(constant.getDoubleValue());

		LibMatrixCUDA.matrixScalarRelational(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, _output.getName(), sc_op);

		ec.releaseMatrixInputForGPUInstruction(mat.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}
}
