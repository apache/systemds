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
 * TODO Additional compiler enhancements:
 * 1) Partial Shuffle Elimination - Any full or aligned blocks could be directly output from the mappers
 *    to the result index. We only need to shuffle, sort and aggregate partial blocks. However, this requires
 *    piggybacking changes, i.e., (1) operations with multiple result indexes, and (2) multiple operations 
 *    with the same result index. 
 * 2) Group Elimination for Append Chains - If we have chains of rappend each intermediate is shuffled and 
 *    aggregated. This is unnecessary if all offsets are known in advance. We could directly pack all rappends
 *    in one GMR map-phase followed by one group and subsequent aggregate. However, this requires an n-ary 
 *    rappend or group (with multiple inputs).
 * 
 */
public class AppendG extends Lop
{
	public static final String OPCODE = "gappend";
	
	private boolean _cbind = true;
	
	public AppendG(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, boolean cbind, ExecType et) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, input3, input4, dt, vt, et);
		
		_cbind = cbind;
	}
	
	public void init(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, ExecType et) 
	{
		addInput(input1);
		input1.addOutput(this);

		addInput(input2);
		input2.addOutput(this);
		
		addInput(input3);
		input3.addOutput(this);
		
		addInput(input4);
		input4.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if( et == ExecType.MR )
		{
			lps.addCompatibility(JobType.GMR);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //SP
		{
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	@Override
	public String toString() {
		return " AppendG: ";
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) 
		throws LopsException
	{
		return getInstructions(
				String.valueOf(input_index1),
				String.valueOf(input_index2),
				String.valueOf(input_index3),
				String.valueOf(input_index4),
				String.valueOf(output_index) );
	}
	
	//called when append executes in SP
	public String getInstructions(String input_index1, String input_index2, String input_index3, String input_index4, String output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( OPCODE );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(3).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output_index) );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( _cbind );
		
		return sb.toString();
	}
}
