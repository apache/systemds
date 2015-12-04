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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class ComputationCPInstruction extends CPInstruction 
{
	

	public CPOperand output;
	public CPOperand input1, input2, input3;
	
	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr ) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = null;
		output = out;
	}

	public ComputationCPInstruction ( Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr ) {
		super(op, opcode, istr);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		output = out;
	}

	public String getOutputVariableName() {
		return output.getName();
	}
	
	/**
	 * 
	 * @param in1
	 * @param out
	 * @return
	 */
	protected boolean checkGuardedRepresentationChange( MatrixBlock in1, MatrixBlock out ) {
		return checkGuardedRepresentationChange(in1, null, out);
	}
	
	/**
	 * 
	 * @param in1
	 * @param in2
	 * @param out
	 * @return
	 */
	protected boolean checkGuardedRepresentationChange( MatrixBlock in1, MatrixBlock in2, MatrixBlock out )
	{
		double memDense = OptimizerUtils.estimateSize(out.getNumRows(), out.getNumColumns());
		double memIn1 = (in1 != null) ? in1.getSizeInMemory() : 0;
		double memIn2 = (in2 != null) ? in2.getSizeInMemory() : 0;
		return ( memDense < memIn1 + memIn2 );	
	}
}
