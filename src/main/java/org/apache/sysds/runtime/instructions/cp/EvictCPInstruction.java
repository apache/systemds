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
import org.apache.sysds.runtime.lineage.LineageGPUCacheEviction;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class EvictCPInstruction extends UnaryCPInstruction
{
	private EvictCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(CPType.EvictLineageCache, op, in, out, opcode, istr);
	}

	public static EvictCPInstruction parseInstruction(String str) {
		InstructionUtils.checkNumFields(str, 3);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		return new EvictCPInstruction(null, in, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// Evict fraction of cached objects
		ScalarObject fr = ec.getScalarInput(input1);
		double evictFrac = ((double) fr.getLongValue()) / 100;
		LineageGPUCacheEviction.removeAllEntries(evictFrac);
	}
}
