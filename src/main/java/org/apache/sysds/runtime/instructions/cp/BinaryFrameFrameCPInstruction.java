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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;

public class BinaryFrameFrameCPInstruction extends BinaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(BinaryFrameFrameCPInstruction.class.getName());

	protected BinaryFrameFrameCPInstruction(MultiThreadedOperator op, CPOperand in1,
			CPOperand in2, CPOperand out, String opcode, String istr) {
		super(CPType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// get input frames
		FrameBlock inBlock1 = ec.getFrameInput(input1.getName());
		FrameBlock inBlock2 = ec.getFrameInput(input2.getName());
		
		if(getOpcode().equals(Opcodes.DROPINVALIDTYPE.toString())) {
			// Perform computation using input frames, and produce the result frame
			FrameBlock retBlock = inBlock1.dropInvalidType(inBlock2);
			// Attach result frame with FrameBlock associated with output_name
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if(getOpcode().equals(Opcodes.VALUESWAP.toString())) {
			// Perform computation using input frames, and produce the result frame
			FrameBlock retBlock = inBlock1.valueSwap(inBlock2);
			// Attach result frame with FrameBlock associated with output_name
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if(getOpcode().equals(Opcodes.FREPLICATE.toString())) {
			// Perform computation using input frames, and produce the result frame
			FrameBlock retBlock = inBlock1.frameRowReplication(inBlock2);
			// Attach result frame with FrameBlock associated with output_name
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if(getOpcode().equals(Opcodes.APPLYSCHEMA.toString())) {
			final int k = ((MultiThreadedOperator)_optr).getNumThreads();
			final FrameBlock out = FrameLibApplySchema.applySchema(inBlock1, inBlock2, k);
			ec.setFrameOutput(output.getName(), out);
		}
		else {
			// Execute binary operations
			BinaryOperator dop = (BinaryOperator) _optr;
			FrameBlock outBlock = inBlock1.binaryOperations(dop, inBlock2, null);
			// Attach result frame with FrameBlock associated with output_name
			ec.setFrameOutput(output.getName(), outBlock);
		}
		
		// Release the memory occupied by input frames
		ec.releaseFrameInput(input1.getName());
		ec.releaseFrameInput(input2.getName());
	}
}
