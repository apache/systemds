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

package org.apache.sysml.runtime.instructions;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.flink.CSVReblockFLInstruction;
import org.apache.sysml.runtime.instructions.flink.FLInstruction;
import org.apache.sysml.runtime.instructions.flink.FLInstruction.FLINSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.flink.MapmmFLInstruction;
import org.apache.sysml.runtime.instructions.flink.ReblockFLInstruction;
import org.apache.sysml.runtime.instructions.flink.TsmmFLInstruction;
import org.apache.sysml.runtime.instructions.flink.WriteFLInstruction;

import java.util.HashMap;

public class FLInstructionParser extends InstructionParser {
	public static final HashMap<String, FLINSTRUCTION_TYPE> String2FLInstructionType;

	static {
		String2FLInstructionType = new HashMap<String, FLINSTRUCTION_TYPE>();

		//binary aggregate operators (matrix multiplication operators)
		String2FLInstructionType.put("mapmm", FLINSTRUCTION_TYPE.MAPMM);
		String2FLInstructionType.put("tsmm", FLINSTRUCTION_TYPE.TSMM);

		// REBLOCK Instruction Opcodes
		String2FLInstructionType.put("rblk", FLINSTRUCTION_TYPE.Reblock);
		String2FLInstructionType.put("csvrblk", FLINSTRUCTION_TYPE.CSVReblock);

		String2FLInstructionType.put("write", FLINSTRUCTION_TYPE.Write);
	}

	public static FLInstruction parseSingleInstruction(String str)
			throws DMLRuntimeException {
		if (str == null || str.isEmpty())
			return null;

		FLINSTRUCTION_TYPE cptype = InstructionUtils.getFLType(str);
		if (cptype == null)
			throw new DMLRuntimeException("Invalid FL Instruction Type: " + str);
		FLInstruction flinst = parseSingleInstruction(cptype, str);
		if (flinst == null)
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return flinst;
	}

	public static FLInstruction parseSingleInstruction(FLINSTRUCTION_TYPE fltype, String str)
			throws DMLRuntimeException {
		if (str == null || str.isEmpty())
			return null;

		String[] parts = null;
		switch (fltype) {
			// matrix multiplication instructions
			case MAPMM:
				return MapmmFLInstruction.parseInstruction(str);
			case TSMM:
				return TsmmFLInstruction.parseInstruction(str);


			case Reblock:
				return ReblockFLInstruction.parseInstruction(str);
			case CSVReblock:
				return CSVReblockFLInstruction.parseInstruction(str);
			case Write:
				return WriteFLInstruction.parseInstruction(str);

			case INVALID:
			default:
				throw new DMLRuntimeException("Invalid FL Instruction Type: " + fltype);
		}
	}
}
