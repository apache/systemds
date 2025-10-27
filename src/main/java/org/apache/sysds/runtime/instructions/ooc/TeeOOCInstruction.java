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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

public class TeeOOCInstruction extends ComputationOOCInstruction {

	protected TeeOOCInstruction(OOCType type, CPOperand in1, CPOperand out, String opcode, String istr) {
		super(type, null, in1, out, opcode, istr);
	}

	public static TeeOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		return new TeeOOCInstruction(OOCType.Tee, in1, out, opcode, str);
	}

	public void processInstruction( ExecutionContext ec ) {
		//get input stream
		MatrixObject min = ec.getMatrixObject(input1);
		OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();

		//get output and create new resettable stream
		MatrixObject mo = ec.getMatrixObject(output);
		mo.setStreamHandle(new CachingStream(qIn));
		mo.setMetaData(min.getMetaData());
	}
}
