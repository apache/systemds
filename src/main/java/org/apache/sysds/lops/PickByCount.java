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


public class PickByCount extends Lop 
{
	public static final String OPCODE = Opcodes.QPICK.toString();
	
	public enum OperationTypes {
		VALUEPICK, 
		RANGEPICK, 
		IQM, 
		MEDIAN
	}
	
	private OperationTypes operation;
	private boolean inMemoryInput = false;
	
	public PickByCount(Lop input1, Lop input2, DataType dt, ValueType vt, OperationTypes op, ExecType et) {
		super(Lop.Type.PickValues, dt, vt);
		init(input1, input2, op, et);
	}

	public PickByCount(Lop input1, Lop input2, DataType dt, ValueType vt, OperationTypes op, ExecType et, boolean inMemoryInput) {
		super(Lop.Type.PickValues, dt, vt);
		this.inMemoryInput = inMemoryInput;
		init(input1, input2, op, et);
	}

	
	private void init(Lop input1, Lop input2, OperationTypes op, ExecType et) {
		addInput(input1);
		input1.addOutput(this);
		
		if ( input2 != null ) {
			addInput(input2);
			input2.addOutput(this);
		}
		
		operation = op;
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	public OperationTypes getOperationType() {
		return operation;
	}

	/*
	 * This version of getInstructions() must be called only for valuepick (CP), IQM (CP)
	 * 
	 * Example instances:
	 * valuepick:::temp2:STRING:::0.25:DOUBLE:::Var1:DOUBLE
	 * valuepick:::temp2:STRING:::Var1:DOUBLE:::Var2:DOUBLE
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		if(operation != OperationTypes.MEDIAN) {
			if ( getInputs().get(1).getDataType() == DataType.SCALAR ) 
				sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
			else {
				sb.append( getInputs().get(1).prepInputOperand(input2));
			}
			sb.append( OPERAND_DELIMITOR );
		}
		
		sb.append( this.prepOutputOperand(output));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(operation);
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(inMemoryInput);

		if ( getExecType() == ExecType.FED ){
			sb.append( OPERAND_DELIMITOR );
			sb.append(_fedOutput.name());
		}
		
		return sb.toString();
	}
	
	/**
	 * This version of getInstructions() is called for IQM, executing in CP
	 * 
	 * Example instances:
	 *   iqm:::input:::output
	 */
	@Override
	public String getInstructions(String input, String output) {
		String ret = InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input),
			prepOutputOperand(output),
			operation.name(),
			String.valueOf(inMemoryInput));
		if ( getExecType() == ExecType.FED )
			ret = InstructionUtils.concatOperands(ret, _fedOutput.name());
		return ret;
	}
}
