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
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.operators.MultiThreadedOperator;

public class UnaryFrameCPInstruction extends UnaryCPInstruction {

	protected UnaryFrameCPInstruction(MultiThreadedOperator op, CPOperand in, CPOperand out, String opcode,
		String instr) {
		super(CPType.Unary, op, in, out, opcode, instr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(getOpcode().equals(Opcodes.TYPEOF.toString())) {
			FrameBlock inBlock = ec.getFrameInput(input1.getName());
			FrameBlock retBlock = inBlock.getSchemaTypeOf();
			ec.releaseFrameInput(input1.getName());
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if(getOpcode().equals(Opcodes.DETECTSCHEMA.toString())) {
			FrameBlock inBlock = ec.getFrameInput(input1.getName());
			FrameBlock retBlock = inBlock.detectSchema(((MultiThreadedOperator) _optr).getNumThreads());
			ec.releaseFrameInput(input1.getName());
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if(getOpcode().equals(Opcodes.COLNAMES.toString())) {
			FrameBlock inBlock = ec.getFrameInput(input1.getName());
			FrameBlock retBlock = inBlock.getColumnNamesAsFrame();
			ec.releaseFrameInput(input1.getName());
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else if (getOpcode().equals(Opcodes.GETCOLNAMES.toString())) {
			FrameBlock inBlock = ec.getFrameInput(input1.getName());
			FrameBlock retBlock = inBlock.getColNames();
			ec.releaseFrameInput(input1.getName());
			ec.setFrameOutput(output.getName(), retBlock);
		}
		else
			throw new DMLScriptException("Opcode '" + getOpcode() + "' is not a valid UnaryFrameCPInstruction");
	}
}
