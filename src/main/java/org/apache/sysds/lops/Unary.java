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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform following operations: with one operand -- NOT(A), ABS(A),
 * SQRT(A), LOG(A) with two operands where one of them is a scalar -- H=H*i,
 * H=H*5, EXP(A,2), LOG(A,2)
 * 
 */

public class Unary extends Lop 
{
	private OpOp1 operation;
	private Lop valInput;
	
	//cp-specific parameters
	private int _numThreads = 1;
	private boolean _inplace = false;


	/**
	 * Constructor to perform a unary operation with 2 inputs
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public Unary(Lop input1, Lop input2, OpOp1 op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, input2, op, dt, vt, et);
	}

	private void init(Lop input1, Lop input2, OpOp1 op, DataType dt, ValueType vt, ExecType et) {
		operation = op;

		if (input1.getDataType() == DataType.MATRIX)
			valInput = input2;
		else
			valInput = input1;

		addInput(input1);
		input1.addOutput(this);
		addInput(input2);
		input2.addOutput(this);
		lps.setProperties(inputs, et);
	}

	/**
	 * Constructor to perform a unary operation with 1 input.
	 * 
	 * @param input1 low-level operator 1
	 * @param op operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 * @param numThreads number of threads
	 * @param inplace inplace behavior
	 */
	public Unary(Lop input1, OpOp1 op, DataType dt, ValueType vt, ExecType et, int numThreads, boolean inplace) {
		super(Lop.Type.UNARY, dt, vt);
		init(input1, op, dt, vt, et);
		_numThreads = numThreads;
		_inplace = inplace;
	}

	private void init(Lop input1, OpOp1 op, DataType dt, ValueType vt, ExecType et) {
		//sanity check
		if ( (op == OpOp1.INVERSE || op == OpOp1.CHOLESKY) && et == ExecType.SPARK )
			throw new LopsException("Invalid exection type "+et.toString()+" for operation "+op.toString());
		
		operation = op;
		valInput = null;
		addInput(input1);
		input1.addOutput(this);
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		if (valInput != null)
			return "Operation: " + operation + " " + "Label: "
					+ valInput.getOutputParameters().getLabel()
					+ " input types " + this.getInputs().get(0).toString()
					+ " " + this.getInputs().get(1).toString();
		else
			return "Operation: " + operation + " " + "Label: N/A";
	}

	private String getOpcode() {
		return operation.toString();
	}
	
	public static boolean isMultiThreadedOp(OpOp1 op) {
		return op==OpOp1.CUMSUM
			|| op==OpOp1.CUMPROD
			|| op==OpOp1.CUMMIN
			|| op==OpOp1.CUMMAX
			|| op==OpOp1.CUMSUMPROD
			|| op==OpOp1.EXP
			|| op==OpOp1.LOG
			|| op==OpOp1.SIGMOID;
	}
	
	@Override
	public String getInstructions(String input1, String output) {
		//sanity check number of operands
		if( getInputs().size() != 1 ) {
			throw new LopsException(printErrorLocation() + "Invalid number of operands ("
					+ getInputs().size() + ") for an Unary opration: " + operation);		
		}
		
		// Unary operators with one input
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
		
		//num threads for cumulative cp ops
		if( getExecType() == ExecType.CP && isMultiThreadedOp(operation) ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
			sb.append( OPERAND_DELIMITOR );
			sb.append( _inplace );
		}
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		
		sb.append( OPERAND_DELIMITOR );
		if ( getInputs().get(0).getDataType() == DataType.SCALAR )
			sb.append( getInputs().get(0).prepScalarInputOperand(getExecType()));
		else
			sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append( OPERAND_DELIMITOR );
		if ( getInputs().get(1).getDataType() == DataType.SCALAR )
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		else 
			sb.append( getInputs().get(1).prepInputOperand(input2));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
}
