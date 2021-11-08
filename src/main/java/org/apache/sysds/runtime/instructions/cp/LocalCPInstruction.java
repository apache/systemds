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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * A simple instruction that take whatever input it is and return it as a local matrix. This forces any distributed
 * matrices to become local.
 */
public class LocalCPInstruction extends ComputationCPInstruction {
	// private static final Log LOG = LogFactory.getLog(LocalCPInstruction.class.getName());

	private LocalCPInstruction(CPOperand in, CPOperand out, String instruction) {
		super(CPType.Local, null, in, null, null, out, "LOCAL", instruction);
	}

	public static LocalCPInstruction parseInstruction(String str) {
		InstructionUtils.checkNumFields(str, 2);
		final String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		final CPOperand in = new CPOperand(parts[1]);
		final CPOperand out = new CPOperand(parts[2]);
		return new LocalCPInstruction(in, out, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		final MatrixBlock in = ec.getMatrixInput(input1.getName());
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), in);
	}
}
