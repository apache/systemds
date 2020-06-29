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
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class BinaryFrameMatrixCPInstruction extends BinaryCPInstruction {
	protected BinaryFrameMatrixCPInstruction(Operator op, CPOperand in1,
			CPOperand in2, CPOperand out, String opcode, String istr) {
		super(CPInstruction.CPType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// Read input frame
		FrameBlock inBlock1 = ec.getFrameInput(input1.getName());
		// the vector with valid column lengths
		MatrixBlock featurelength = ec.getMatrixInput(input2.getName());
		// identify columns with invalid lengths
		FrameBlock out = inBlock1.invalidByLength(featurelength);
		// Release the memory occupied by inputs
		ec.releaseFrameInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		// Attach result frame with output
		ec.setFrameOutput(output.getName(),out);
	}
}
