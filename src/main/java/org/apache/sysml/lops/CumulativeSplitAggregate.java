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

public class CumulativeSplitAggregate extends Lop 
{
	private double _initValue = 0;
	
	public CumulativeSplitAggregate(Lop input, DataType dt, ValueType vt, double init)
		throws LopsException 
	{
		super(Lop.Type.CumulativeSplitAggregate, dt, vt);
		_initValue = init;
		init(input, dt, vt, ExecType.MR);
	}
	
	private void init(Lop input, DataType dt, ValueType vt, ExecType et) {
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
	}

	public String toString() {
		return "CumulativeSplitAggregate";
	}
	
	private String getOpcode() {
		return "ucumsplit";
	}
	
	@Override
	public String getInstructions(int input_index, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( String.valueOf(_initValue) );

		return sb.toString();
	}
}
