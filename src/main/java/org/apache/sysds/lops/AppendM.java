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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


public class AppendM extends Lop
{
	public static final String OPCODE = Opcodes.MAPPEND.toString();
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}
	
	private boolean _cbind = true;
	
	public AppendM(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, boolean cbind, boolean partitioned, ExecType et) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, input3, dt, vt, et);
		
		_cbind = cbind;
	}
	
	public void init(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) 
	{
		addInput(input1);
		input1.addOutput(this);
		addInput(input2);
		input2.addOutput(this);
		addInput(input3);
		input3.addOutput(this);
		lps.setProperties(inputs, ExecType.SPARK);
	}

	@Override
	public Lop getBroadcastInput() {
		if (getExecType() != ExecType.SPARK)
			return null;
		return getInputs().get(1); //frame or matrix
	}
	
	@Override
	public String toString() {
		return "Operation = AppendM"; 
	}

	//called when append executes in SP
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			getInputs().get(2).prepScalarInputOperand(getExecType()),
			prepOutputOperand(output),
			String.valueOf(_cbind));
	}
}
