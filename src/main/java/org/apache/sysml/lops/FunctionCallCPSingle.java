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


import java.util.ArrayList;

import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

public class FunctionCallCPSingle extends Lop
{
	private String _fnamespace;
	private String _fname;
	
	public FunctionCallCPSingle(ArrayList<Lop> inputs, String fnamespace, String fname, ExecType et)
	{
		super(Lop.Type.FunctionCallCPSingle, DataType.UNKNOWN, ValueType.UNKNOWN);
		//note: data scalar in order to prevent generation of redundant createvar, rmvar
		
		_fnamespace = fnamespace;
		_fname = fname;
		
		//wire inputs
		for( Lop in : inputs ) {
			addInput( in );
			in.addOutput( this );
		}
		
		//lop properties: always in CP
		boolean breaksAlignment = false; 
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	
	@Override
	public String toString() {
		return "function call: " + DMLProgram.constructFunctionKey(_fnamespace, _fname);
	}
	
	@Override
	public String getInstructions(String input1, String output) {
		return getInstructions(new String[]{input1}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		return getInstructions(new String[]{input1, input2}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		return getInstructions(new String[]{input1, input2, input3}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) {
		return getInstructions(new String[]{input1, input2, input3, input4}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String output) {
		return getInstructions(new String[]{input1, input2, input3, input4, input5}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String output) {
		return getInstructions(new String[]{input1, input2, input3, input4, input5, input6}, new String[]{output});	
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String input7, String output) {
		return getInstructions(new String[]{input1, input2, input3, input4, input5, input6, input7}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String[] inputs, String output)
	{
		StringBuilder inst = new StringBuilder();
		inst.append(getExecType());
		
		inst.append(Lop.OPERAND_DELIMITOR); 
		inst.append(FunctionOp.OPSTRING);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fnamespace);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fname);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(inputs.length);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append("1"); //single output
		
		for(int i=0; i<inputs.length; i++) {
			inst.append(Lop.OPERAND_DELIMITOR);
			inst.append( getInputs().get(i).prepInputOperand(inputs[i]) );
		}
		
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(output);
		
		return inst.toString();
	}
}
