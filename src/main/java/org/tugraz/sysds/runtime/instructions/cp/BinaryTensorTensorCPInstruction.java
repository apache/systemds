/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;

public class BinaryTensorTensorCPInstruction extends BinaryCPInstruction {

	protected BinaryTensorTensorCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(CPType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// Read input tensors
		TensorBlock inBlock1 = ec.getTensorInput(input1.getName());
		TensorBlock inBlock2 = ec.getTensorInput(input2.getName());

		// Perform computation using input tensors, and produce the result tensor
		BinaryOperator bop = (BinaryOperator) _optr;
		TensorBlock retBlock = inBlock1.binaryOperations(bop, inBlock2, null);
		
		// Release the memory occupied by input matrices
		ec.releaseTensorInput(input1.getName(), input2.getName());
		
		// TODO Ensure right dense/sparse output representation (guarded by released input memory)

		// Attach result matrix with MatrixObject associated with output_name
		ec.setTensorOutput(output.getName(), retBlock);
	}
}