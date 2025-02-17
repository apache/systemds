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

package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class Compression extends Lop {
	public static final String OPCODE = "compress";

	private final int _singletonLookupKey;
	private final int _numThreads;

	public enum CompressConfig {
		TRUE, FALSE, COST, AUTO, WORKLOAD;

		public boolean isEnabled() {
			return this != FALSE;
		}

		public boolean isWorkload(){
			return this == WORKLOAD;
		}
	}

	public Compression(Lop input, DataType dt, ValueType vt, ExecType et, int singletonLookupKey, int numThreads) {
		super(Lop.Type.Checkpoint, dt, vt);
		addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
		_singletonLookupKey = singletonLookupKey;
		_numThreads = numThreads;
	}

	@Override
	public String toString() {
		return OPCODE;
	}

	@Override
	public String getInstructions(String input1, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(OPCODE);
		sb.append(OPERAND_DELIMITOR);
		if(getInputs().get(0) instanceof FunctionCallCP &&
			((FunctionCallCP)getInputs().get(0)).getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString()) ){
			sb.append(getInputs().get(0).getOutputs().get(0).getOutputParameters().getLabel());
		}
		else{
			sb.append(getInputs().get(0).prepInputOperand(input1));
		}
		sb.append(OPERAND_DELIMITOR);
		if(getInputs().get(0) instanceof FunctionCallCP && 
			((FunctionCallCP)getInputs().get(0)).getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString()) ){
			sb.append(getInputs().get(0).getOutputs().get(0).getOutputParameters().getLabel());
		}
		else{
			sb.append(prepOutputOperand(output));
		}
		if(_singletonLookupKey != 0){
			sb.append(OPERAND_DELIMITOR);
			sb.append(_singletonLookupKey);
		}

		if(getExecType().equals(ExecType.CP) || getExecType().equals(ExecType.FED)){
			sb.append(OPERAND_DELIMITOR);
			sb.append(_numThreads);
		}
		
		
		return sb.toString();
	}
}
