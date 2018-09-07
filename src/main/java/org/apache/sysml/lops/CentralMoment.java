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
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform cross product operation
 */
public class CentralMoment extends Lop 
{
	
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
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
			definesMRJob = true;
			lps.addCompatibility(JobType.CM_COV);
			lps.setProperties(inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else { //CP/SPARK
			// when executing in CP, this lop takes an optional 3rd input (Weights)
			if ( input3 != null ) {
				this.addInput(input3);
				input3.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	public CentralMoment(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		this(input1, input2, null, dt, vt, et);
	}

	public CentralMoment(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.CentralMoment, dt, vt);
		init(input1, input2, input3, et);
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
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( "cm" );
		sb.append( OPERAND_DELIMITOR );
		
		// Input data
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		
		// Weights
		if( input3 != null ) {
			sb.append( getInputs().get(1).prepInputOperand(input2) );
			sb.append( OPERAND_DELIMITOR );
		}
		
		// Order
		sb.append( getInputs().get((input3!=null)?2:1)
				.prepScalarInputOperand(getExecType()) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOutputOperand(output));
		
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
	
	/**
	 * Function to generate MR central moment instruction, invoked from
	 * <code>Dag.java:getAggAndOtherInstructions()</code>. This function
	 * is used for both weighted and unweighted operations.
	 * 
	 * input_index: data (in case of weighted: data combined via combinebinary)
	 * output_index: produced output
	 * 
	 * The order for central moment is derived internally in the function.
	 */
	@Override
	public String getInstructions(int input_index, int output_index) {
		return getInstructions(String.valueOf(input_index), "", String.valueOf(output_index));
	}
}