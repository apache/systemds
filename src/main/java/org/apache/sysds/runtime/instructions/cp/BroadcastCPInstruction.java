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

import java.util.concurrent.Executors;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.CommonThreadPool;

public class BroadcastCPInstruction extends UnaryCPInstruction {
	private BroadcastCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(CPType.Broadcast, op, in, out, opcode, istr);
	}
	
	public static BroadcastCPInstruction parseInstruction (String str) {
		InstructionUtils.checkNumFields(str, 2);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		return new BroadcastCPInstruction(null, in, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		ec.setVariable(output.getName(), ec.getMatrixObject(input1));

		if (CommonThreadPool.triggerRemoteOPsPool == null)
			CommonThreadPool.triggerRemoteOPsPool = Executors.newCachedThreadPool();
		CommonThreadPool.triggerRemoteOPsPool.submit(new TriggerBroadcastTask(ec, ec.getMatrixObject(output)));
	}
}
