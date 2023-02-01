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

 
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform cross product operation
 */
public class MMRJ extends Lop 
{
	/**
	 * Constructor to perform a cross product operation.
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public MMRJ(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) 
	{
		//handle inputs and outputs
		super(Lop.Type.MMRJ, dt, vt);
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return "Operation = MMRJ";
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		boolean toCache = getOutputParameters().getLinCacheMarking();
		return InstructionUtils.concatOperands(
			getExecType().name(),
			"rmm",
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output));
	}
}
