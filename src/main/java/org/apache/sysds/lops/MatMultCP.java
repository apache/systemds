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

public class MatMultCP extends Lop {
	private int numThreads = -1;
	private boolean isLeftTransposed; // Used for GPU matmult operation
	private boolean isRightTransposed;
	private boolean useTranspose;

	public MatMultCP(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		this(input1, input2, dt, vt, et, 1);
	}

	public MatMultCP(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et, int k) {
		super(Lop.Type.MatMultCP, dt, vt);
		init(input1, input2, dt, vt, et);
		numThreads = k;
	}

	public MatMultCP(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et, boolean isLeftTransposed,
		boolean isRightTransposed) {
		super(Lop.Type.Binary, dt, vt);
		init(input1, input2, dt, vt, et);
		this.isLeftTransposed = isLeftTransposed;
		this.isRightTransposed = isRightTransposed;
		this.useTranspose = true;
	}

	public MatMultCP(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et, int k, boolean isLeftTransposed,
		boolean isRightTransposed) {
		this(input1, input2, dt, vt, et, k);
		this.isLeftTransposed = isLeftTransposed;
		this.isRightTransposed = isRightTransposed;
		this.useTranspose = true;
	}

	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return " Operation: ba+*";
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		String ret = null;
		if(!useTranspose) {
			ret = InstructionUtils.concatOperands(
				getExecType().name(), Opcodes.MMULT.toString(),
				getInputs().get(0).prepInputOperand(input1),
				getInputs().get(1).prepInputOperand(input2),
				prepOutputOperand(output),
				String.valueOf(numThreads));
		}
		else { // GPU or compressed
			ret = InstructionUtils.concatOperands(
				getExecType().name(), Opcodes.MMULT.toString(),
				getInputs().get(0).prepInputOperand(input1),
				getInputs().get(1).prepInputOperand(input2),
				prepOutputOperand(output),
				String.valueOf(numThreads),
				String.valueOf(isLeftTransposed),
				String.valueOf(isRightTransposed));
		}
		
		if ( getExecType() == ExecType.FED )
			ret = InstructionUtils.concatOperands(ret, _fedOutput.name());
		
		return ret;
	}
}
