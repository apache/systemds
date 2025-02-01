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


/**
 * Lop to perform cross product operation
 */
public class CentralMoment extends Lop 
{
	private final int _numThreads;
	
	public CentralMoment(Lop input1, Lop input2, DataType dt, ValueType vt, int numThreads, ExecType et) {
		this(input1, input2, null, dt, vt, numThreads, et);
	}

	public CentralMoment(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, int numThreads, ExecType et) {
		super(Lop.Type.CentralMoment, dt, vt);
		init(input1, input2, input3, et);
		_numThreads = numThreads;
	}
	
	/**
	 * Constructor to perform central moment.
	 * input1 <- data (weighted or unweighted)
	 * input2 <- order (integer: 0, 2, 3, or 4)
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param input3 low-level operator 3
	 * @param et execution type
	 */
	private void init(Lop input1, Lop input2, Lop input3, ExecType et) {
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		// when executing in CP, this lop takes an optional 3rd input (Weights)
		if ( input3 != null ) {
			addInput(input3);
			input3.addOutput(this);
		}
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return "Operation = CentralMoment";
	}

	/**
	 * Function to generate CP centralMoment instruction for weighted operation.
	 * 
	 * input1: data
	 * input2: weights
	 * input3: order
	 */
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		if( input3 == null ) {
			InstructionUtils.concatOperands(sb,
				getExecType().toString(), Opcodes.CM.toString(),
				getInputs().get(0).prepInputOperand(input1),
				getInputs().get(1).prepScalarInputOperand(getExecType()),
				prepOutputOperand(output));
		}
		else {
			InstructionUtils.concatOperands(sb,
				getExecType().toString(), Opcodes.CM.toString(),
				getInputs().get(0).prepInputOperand(input1),
				getInputs().get(1).prepInputOperand(input2),
				getInputs().get(2).prepScalarInputOperand(getExecType()),
				prepOutputOperand(output));
		}
		if( getExecType() == ExecType.CP || getExecType() == ExecType.FED ) {
			sb.append(OPERAND_DELIMITOR);
			sb.append(_numThreads);
			if ( getExecType() == ExecType.FED ){
				sb.append(OPERAND_DELIMITOR);
				sb.append(_fedOutput);
			}
		}
		return sb.toString();
	}
	
	/**
	 * Function to generate CP centralMoment instruction for unweighted operation.
	 * 
	 * input1: data
	 * input2: order (not used, and order is derived internally!)
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) {
		return getInstructions(input1, input2, null, output);
	}
}
