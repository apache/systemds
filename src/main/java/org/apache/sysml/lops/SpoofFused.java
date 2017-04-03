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

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

public class SpoofFused extends Lop
{
	private final Class<?> _class;
	private final int _numThreads;
	
	public SpoofFused( ArrayList<Lop> inputs, DataType dt, ValueType vt, Class<?> cla, int k, ExecType etype) {
		super(Type.SpoofFused, dt, vt);
		_class = cla;
		_numThreads = k;
		
		for( Lop lop : inputs ) {
			addInput(lop);
			lop.addOutput(this);
		}
		
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties( inputs, etype, ExecLocation.ControlProgram, false, false, false );
	}

	@Override
	public String toString() {
		return "spoof("+_class.getSimpleName()+")";
	}

	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		return getInstructions(new String[]{input1}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2, input3}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2, input3, input4}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2, input3, input4, input5}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2, input3, input4, input5, input6}, new String[]{output});	
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String input7, String output) throws LopsException {
		return getInstructions(new String[]{input1, input2, input3, input4, input5, input6, input7}, new String[]{output});
	}
	
	@Override
	public String getInstructions(String[] inputs, String output) throws LopsException {
		return getInstructions(inputs, new String[]{output});
	}
	
	@Override
	public String getInstructions(String[] inputs, String[] outputs) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "spoof" );

		sb.append( OPERAND_DELIMITOR );
		sb.append( _class.getName() );
		
		for(int i=0; i < inputs.length; i++) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(i).prepInputOperand(inputs[i]));
		}
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(outputs[0]) );
	
		sb.append( OPERAND_DELIMITOR );
		sb.append( _numThreads );
		
		return sb.toString();
	}
}
