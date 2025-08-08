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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.InstructionType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.*;

public class OOCInstructionParser extends InstructionParser {
	protected static final Log LOG = LogFactory.getLog(OOCInstructionParser.class.getName());

	public static OOCInstruction parseSingleInstruction(String str) {
		if(str == null || str.isEmpty())
			return null;
		InstructionType ooctype = InstructionUtils.getOOCType(str);
		if(ooctype == null)
			throw new DMLRuntimeException("Unable derive ooctype for instruction: " + str);
		OOCInstruction oocinst = parseSingleInstruction(ooctype, str);
		if(oocinst == null)
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return oocinst;
	}

	public static OOCInstruction parseSingleInstruction(InstructionType ooctype, String str) {
		if(str == null || str.isEmpty())
			return null;
		switch(ooctype) {
			case Reblock:
				return ReblockOOCInstruction.parseInstruction(str);
			case AggregateUnary:
				return AggregateUnaryOOCInstruction.parseInstruction(str);
			case Unary:
				return UnaryOOCInstruction.parseInstruction(str);
			case Binary:
				return BinaryOOCInstruction.parseInstruction(str);
			case AggregateBinary:
			case MAPMM:
				return MatrixVectorBinaryOOCInstruction.parseInstruction(str);
			
			default:
				throw new DMLRuntimeException("Invalid OOC Instruction Type: " + ooctype);
		}
	}
}
