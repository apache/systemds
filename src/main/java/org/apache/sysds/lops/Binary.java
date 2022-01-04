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

import java.util.ArrayList;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform binary operation. Both inputs must be matrices or vectors. 
 * Example - A = B + C, where B and C are matrices or vectors.
 */

public class Binary extends Lop 
{
	private OpOp2 operation;
	private final int _numThreads;
	
	/**
	 * Constructor to perform a binary operation.
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et exec type
	 */
	
	public Binary(Lop input1, Lop input2, OpOp2 op, DataType dt, ValueType vt, ExecType et) {
		this(input1, input2, op, dt, vt, et, 1);
	}
	
	public Binary(Lop input1, Lop input2, OpOp2 op, DataType dt, ValueType vt, ExecType et, int k) {
		super(Lop.Type.Binary, dt, vt);
		init(input1, input2, op, dt, vt, et);
		_numThreads = k;
	}
	
	private void init(Lop input1, Lop input2, OpOp2 op, DataType dt, ValueType vt, ExecType et)  {
		operation = op;
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return " Operation: " + operation;
	}
	
	@Override
	public Lop getBroadcastInput() {
		if (getExecType() != ExecType.SPARK)
			return null;

		ArrayList<Lop> inputs = getInputs();
		if (inputs.get(0).getDataType() == DataType.FRAME && inputs.get(1).getDataType() == DataType.MATRIX)
			return inputs.get(1);
		else
			return null;
	}
	
	public OpOp2 getOperationType() {
		return operation;
	}

	private String getOpcode() {
		return operation.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		String ret = InstructionUtils.concatOperands(
			getExecType().name(), getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output));

		if ( getExecType() == ExecType.CP )
			ret = InstructionUtils.concatOperands(ret, String.valueOf(_numThreads));
		else if( getExecType() == ExecType.FED )
			ret = InstructionUtils.concatOperands(ret, String.valueOf(_numThreads), _fedOutput.name());

		return ret;
	}
}
