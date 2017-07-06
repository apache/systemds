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

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

public class ConvolutionTransform extends Lop
{

	
	public enum OperationTypes {
		MAX_POOLING, MAX_POOLING_BACKWARD, RELU_MAX_POOLING, RELU_BACKWARD, RELU_MAX_POOLING_BACKWARD,
		DIRECT_CONV2D, DIRECT_CONV2D_BACKWARD_FILTER, DIRECT_CONV2D_BACKWARD_DATA,
		BIAS_ADD, DIRECT_CONV2D_BIAS_ADD, BIAS_MULTIPLY
	};
	
	private OperationTypes operation = null;
	private int numThreads = -1;
	
	/**
	 * Constructor when we have one input.
	 * 
	 * @param input low-level operator
	 * @param op convolution transform operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 * @param k number of threads
	 */
	public ConvolutionTransform(Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
		numThreads = k;
	}
	
	public ConvolutionTransform(Lop input1, Lop input2, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input1, op, dt, vt, et);
		numThreads = k;
		this.addInput(input2);
		input2.addOutput(this);
		setLevel();
	}

	private void init (Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		operation = op;
 
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
			throw new RuntimeException("The execution type is not supported: " + et.name());
		}
		else //CP/SPARK
		{
			// <code>breaksAlignment</code> is not meaningful when <code>Transform</code> executes in CP. 
			breaksAlignment = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
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
				
		case MAX_POOLING:
			return "maxpooling";
			
		case RELU_MAX_POOLING:
			return "relu_maxpooling";
			
		case RELU_MAX_POOLING_BACKWARD:
			return "relu_maxpooling_backward";
			
		case RELU_BACKWARD:
			return "relu_backward";
			
		case MAX_POOLING_BACKWARD:
			return "maxpooling_backward";
		
		case DIRECT_CONV2D:
			return "conv2d";
		
		case DIRECT_CONV2D_BIAS_ADD:
			return "conv2d_bias_add";
		
		case BIAS_ADD:
			return "bias_add";
		
		case BIAS_MULTIPLY:
			return "bias_multiply";
			
		case DIRECT_CONV2D_BACKWARD_FILTER:
			return "conv2d_backward_filter";
			
		case DIRECT_CONV2D_BACKWARD_DATA:
			return "conv2d_backward_data";
			
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	public String getInstructions(String input, String bias, String output) throws LopsException {
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
			return sb.toString();
		}
		else {
			throw new LopsException("The operation is not supported with two operands:" + operation.name());
		}
	}
	
	@Override
	public String getInstructions(String[] inputs, String output) throws LopsException {
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
	}

}
