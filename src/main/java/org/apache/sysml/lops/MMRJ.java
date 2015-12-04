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
public class MMRJ extends Lop 
{

		
	/**
	 * Constructor to perform a cross product operation.
	 * @param input
	 * @param op
	 */

	public MMRJ(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) 
	{
		//handle inputs and outputs
		super(Lop.Type.MMRJ, dt, vt);		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		//set basic lop properties based on exec type
		if( et == ExecType.MR )
		{
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = true;
			lps.addCompatibility(JobType.MMRJ);
			this.lps.setProperties( inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
		}
		else //if( et == ExecType.SPARK )
		{
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	@Override
	public String toString() {
	
		return "Operation = MMRJ";
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "rmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "rmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
}