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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

/**
 * Lop to perform Sum of a matrix with another matrix multiplied by Scalar.
 */
public class Ternary extends Lop 
{
	private final OpOp3 _op;
	private final int _numThreads;
	
	public Ternary(OpOp3 op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et, int numThreads) {
		super(Lop.Type.Ternary, dt, vt);
		_op = op;
		_numThreads = numThreads;
		init(input1, input2, input3, et);
	}

	private void init(Lop input1, Lop input2, Lop input3, ExecType et) {
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		lps.setProperties( inputs, et);
	}
	
	@Override
	public String toString() {
		return "Operation = t("+_op.toString()+")";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output)  {
		String ret = InstructionUtils.concatOperands(
			getExecType().name(), _op.toString(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			getInputs().get(2).prepInputOperand(input3),
			prepOutputOperand(output));
		
		if( getDataType().isMatrix() ) {
			if( getExecType() == ExecType.CP )
				ret = InstructionUtils.concatOperands(ret, String.valueOf(_numThreads));
			else if( getExecType() == ExecType.FED )
				ret = InstructionUtils.concatOperands(ret, String.valueOf(_numThreads), _fedOutput.name());
		}
		
		return ret;
	}
}
