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
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public class TernaryFrameScalarCPInstruction extends TernaryCPInstruction
{
	protected TernaryFrameScalarCPInstruction(TernaryOperator op, CPOperand in1,
			CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr) {
		super(op, in1, in2, in3, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec)  {
		// get input frames
		FrameBlock inBlock = ec.getFrameInput(input1.getName());
		ScalarObject margin = ec.getScalarInput(input3);
		String stringExpression = ec.getScalarInput(input2).getStringValue();
		//compute results
		FrameBlock outBlock = inBlock.map(stringExpression, margin.getLongValue());
		// Attach result frame with FrameBlock associated with output_name
		ec.setFrameOutput(output.getName(), outBlock);
		// Release the memory occupied by input frames
		ec.releaseFrameInput(input1.getName());
	}
}

