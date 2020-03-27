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

package org.apache.sysds.runtime.instructions;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.fed.AggregateBinaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FEDType;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;

import java.util.HashMap;

public class FEDInstructionParser extends InstructionParser
{
	public static final HashMap<String, FEDType> String2FEDInstructionType;
	static {
		String2FEDInstructionType = new HashMap<>();
		String2FEDInstructionType.put("fedinit", FEDType.Init);
		String2FEDInstructionType.put("ba+*",    FEDType.AggregateBinary);
	}

	public static FEDInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		FEDType fedtype = InstructionUtils.getFEDType(str);
		if ( fedtype == null )
			throw new DMLRuntimeException("Unable derive fedtype for instruction: " + str);
		FEDInstruction cpinst = parseSingleInstruction(fedtype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static FEDInstruction parseSingleInstruction ( FEDType fedtype, String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		switch(fedtype) {
			case Init:
				return InitFEDInstruction.parseInstruction(str);
			case AggregateBinary:
				return AggregateBinaryFEDInstruction.parseInstruction(str);
			default:
				throw new DMLRuntimeException("Invalid FEDERATED Instruction Type: " + fedtype );
		}
	}
}
