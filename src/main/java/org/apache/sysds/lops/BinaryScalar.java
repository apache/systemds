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


import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;

/**
 * Lop to perform binary scalar operations. Both inputs must be scalars.
 * Example i = j + k, i = i + 1. 
 */
public class BinaryScalar extends Lop 
{
	private final OpOp2 operation;
	
	/**
	 * Constructor to perform a scalar operation
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 */
	public BinaryScalar(Lop input1, Lop input2, OpOp2 op, DataType dt, ValueType vt) {
		super(Lop.Type.BinaryCP, dt, vt);
		operation = op;
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		lps.setProperties(inputs, ExecType.CP);
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	public OpOp2 getOperationType() {
		return operation;
	}
	
	@Override
	public Lop.SimpleInstType getSimpleInstructionType() {
		return SimpleInstType.Scalar;
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(), operation.toString(),
			getInputs().get(0).prepScalarInputOperand(getExecType()),
			getInputs().get(1).prepScalarInputOperand(getExecType()),
			prepOutputOperand(output));
	}
}
