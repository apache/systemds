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
		IM2COL,
		RESHAPE_COL,
		ROTATE180,
		COL2IM,
		MAX_POOLING,
		MAX_POOLING_BACKWARD,
		DIRECT_CONV2D, DIRECT_CONV2D_BACKWARD_FILTER, DIRECT_CONV2D_BACKWARD_DATA
	};
	
	private OperationTypes operation = null;
	private int numThreads = -1;
	
	/**
	 * Constructor when we have one input.
	 * @param input
	 * @param op
	 */

	public ConvolutionTransform(Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
		numThreads = k;
	}
	
	public ConvolutionTransform(Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, ExecType.MR);
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
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}

	private String getOpcode() {
		switch(operation) {
			
		case IM2COL:
			return "im2col";
			
		case RESHAPE_COL:
			return "reshape_col";
		
		case ROTATE180:
			return "rotate180";
		
		case COL2IM:
			return "col2im";
			
		case MAX_POOLING:
			return "maxpooling";
			
		case MAX_POOLING_BACKWARD:
			return "maxpooling_backward";
		
		case DIRECT_CONV2D:
			return "conv2d";
			
		case DIRECT_CONV2D_BACKWARD_FILTER:
			return "conv2d_backward_filter";
			
		case DIRECT_CONV2D_BACKWARD_DATA:
			return "conv2d_backward_data";
			
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	//CP instructions
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
	public String getInstructions(String input, String stride1, String stride2, String padding1, String padding2, 
			String input_shape1, String input_shape2, String input_shape3, String input_shape4,
			String filter_shape1, String filter_shape2, String filter_shape3, String filter_shape4,
			String output) throws LopsException {
		//only used for im2col and col2im
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		
		//rows, cols, byrow
		String[] inputX = new String[]{stride1, stride2, padding1, padding2, 
			 input_shape1, input_shape2, input_shape3, input_shape4,
			 filter_shape1, filter_shape2, filter_shape3, filter_shape4};
		for( int i=1; i<=(inputX.length); i++ ) {
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
		
		return sb.toString();
	}
	
	public String getInstructions(String input, String dout, String stride1, String stride2, String padding1, String padding2, 
			String input_shape1, String input_shape2, String input_shape3, String input_shape4,
			String filter_shape1, String filter_shape2, String filter_shape3, String filter_shape4,
			String output) throws LopsException {
		//only used for im2col and col2im
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(dout));
		
		String[] inputX = new String[]{input, dout, stride1, stride2, padding1, padding2, 
			 input_shape1, input_shape2, input_shape3, input_shape4,
			 filter_shape1, filter_shape2, filter_shape3, filter_shape4};
		for( int i=2; i < inputX.length; i++ ) {
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
		
		return sb.toString();
	}

}