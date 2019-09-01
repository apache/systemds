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

package org.tugraz.sysds.lops;

import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;

public class DnnTransform extends Lop
{
	public enum OperationTypes {
		MAX_POOL, MAX_POOL_BACKWARD, AVG_POOL, AVG_POOL_BACKWARD,
		RELU_MAX_POOLING, RELU_MAX_POOLING_BACKWARD, RELU_BACKWARD,
		CONV2D, CONV2D_BACKWARD_FILTER, CONV2D_BACKWARD_DATA,
		BIAS_ADD, CONV2D_BIAS_ADD, BIAS_MULTIPLY, CHANNEL_SUMS, BATCH_NORM2D_TEST, 
		UPDATE_NESTEROV_X
	}
	
	private OperationTypes operation;
	private double intermediateMemBudget;
	private final int numThreads;
	
	/**
	 * Constructor when we have one input.
	 * 
	 * @param input low-level operator
	 * @param op convolution transform operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 * @param k number of threads
	 * @param intermediateMemBudget intermediate memory budget
	 */
	public DnnTransform(Lop input, DnnTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k, double intermediateMemBudget) 
	{
		super(Lop.Type.Transform, dt, vt);
		init(input, op, dt, vt, et);
		numThreads = k;
		this.intermediateMemBudget = intermediateMemBudget;
	}
	
	public DnnTransform(Lop input1, Lop input2, DnnTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) 
	{
		super(Lop.Type.Transform, dt, vt);
		init(input1, op, dt, vt, et);
		numThreads = k;
		this.addInput(input2);
		input2.addOutput(this);
		setLevel();
	}
	
	public DnnTransform(Lop input1, Lop input2, Lop input3, DnnTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) 
	{
		super(Lop.Type.Transform, dt, vt);
		init(input1, op, dt, vt, et);
		numThreads = k;
		this.addInput(input2);
		input2.addOutput(this);
		this.addInput(input3);
		input3.addOutput(this);
		setLevel();
	}

	private void init (Lop input, DnnTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		operation = op;
		this.addInput(input);
		input.addOutput(this);
		lps.setProperties( inputs, et);
	}
	
	public void updateLopProperties() {
		lps.setLevel(inputs);
	}

	@Override
	public String toString() {

		return " Operation: " + operation;
	}

	/**
	 * method to get operation type
	 * @return operation type
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}

	private String getOpcode() {
		switch(operation) {
				
		case MAX_POOL:
			return "maxpooling";
			
		case RELU_MAX_POOLING:
			return "relu_maxpooling";
			
		case RELU_MAX_POOLING_BACKWARD:
			return "relu_maxpooling_backward";
			
		case RELU_BACKWARD:
			return "relu_backward";
			
		case MAX_POOL_BACKWARD:
			return "maxpooling_backward";
		
		case AVG_POOL:
			return "avgpooling";
			
		case AVG_POOL_BACKWARD:
			return "avgpooling_backward";
		
		case CONV2D:
			return "conv2d";
		
		case CONV2D_BIAS_ADD:
			return "conv2d_bias_add";
		
		case BIAS_ADD:
			return "bias_add";
		
		case BIAS_MULTIPLY:
			return "bias_multiply";
			
		case CONV2D_BACKWARD_FILTER:
			return "conv2d_backward_filter";
			
		case CONV2D_BACKWARD_DATA:
			return "conv2d_backward_data";
			
		case CHANNEL_SUMS:
			return "channel_sums";
		
		case UPDATE_NESTEROV_X:
			return "update_nesterov_x";
			
		case BATCH_NORM2D_TEST:
			return "batch_norm2d_test";
			
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input, String bias, String output) {
		if(operation == OperationTypes.BIAS_ADD || operation == OperationTypes.BIAS_MULTIPLY || operation == OperationTypes.RELU_BACKWARD) {
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			
			sb.append( OPERAND_DELIMITOR );
			sb.append( getOpcode() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(0).prepInputOperand(input));
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(0).prepInputOperand(bias));
			//output
			sb.append( OPERAND_DELIMITOR );
			sb.append( this.prepOutputOperand(output));
			
			//append degree of parallelism
			if( getExecType()==ExecType.CP ) {
				sb.append( OPERAND_DELIMITOR );
				sb.append( numThreads );
			}
			
			sb.append( OPERAND_DELIMITOR );
			sb.append( intermediateMemBudget );
			return sb.toString();
		}
		else {
			throw new LopsException("The operation is not supported with two operands:" + operation.name());
		}
	}
	
	@Override
	public String getInstructions(String input, String C, String HW, String output) {
		if(operation != OperationTypes.CHANNEL_SUMS)
			throw new LopsException("The operation is not supported with three operands:" + operation.name());
		
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input),
			getInputs().get(1).prepInputOperand(C),
			getInputs().get(2).prepInputOperand(HW),
			prepOutputOperand(output));
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) {
		if(operation != OperationTypes.UPDATE_NESTEROV_X)
			throw new LopsException("The operation is not supported with three operands:" + operation.name());
			
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			getInputs().get(2).prepInputOperand(input3),
			getInputs().get(3).prepInputOperand(input4),
			prepOutputOperand(output));
	}
	
	@Override
	public String getInstructions(String[] inputs, String output) {
		StringBuilder sb = new StringBuilder();
		appendOpcode(sb);
		
		for( int i=0; i<inputs.length-12; i++ ) {
			if( i > 0 )
				sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(i).prepInputOperand(inputs[i]));
		}
		appendOperands(inputs.length-12, inputs.length, output, sb);
		
		return sb.toString();
	}
	
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String output) {
		if(operation != OperationTypes.BATCH_NORM2D_TEST)
			throw new LopsException("The operation is not supported with six operands:" + operation.name());
		
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			getInputs().get(2).prepInputOperand(input3),
			getInputs().get(3).prepInputOperand(input4),
			getInputs().get(4).prepInputOperand(input5),
			getInputs().get(5).prepInputOperand(input6),
			prepOutputOperand(output));
	}

	public void appendOpcode(StringBuilder sb) {
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
	}
	
	public void appendOperands(int startInputIndex, int endInputIndex, String output, StringBuilder sb) {
		for( int i=startInputIndex; i < endInputIndex; i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( ltmp.prepScalarInputOperand(getExecType()));
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		//append degree of parallelism
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( numThreads );
		}
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( intermediateMemBudget );
	}
}
