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

import org.apache.sysml.lops.Aggregate.OperationTypes;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;

public class CumulativePartialAggregate extends Lop 
{
	
	private OperationTypes _op;
	
	public CumulativePartialAggregate(Lop input, DataType dt, ValueType vt, OperationTypes op, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.CumulativePartialAggregate, dt, vt);
		
		//sanity check for supported aggregates
		if( !(op == OperationTypes.KahanSum || op == OperationTypes.Product ||
			  op == OperationTypes.Min || op == OperationTypes.Max) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
		_op = op;
		
		init(input, dt, vt, et);
	}
	
	private void init(Lop input, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input);
		input.addOutput(this);

		if( et == ExecType.MR )
		{
			//setup MR parameters
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = false;
			
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		}
		else //Spark/CP
		{
			//setup Spark parameters 
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	@Override
	public String toString() {
		return "CumulativePartialAggregate";
	}
	
	private String getOpcode() 
	{
		switch( _op ) {
			case KahanSum: 	return "ucumack+";
			case Product: 	return "ucumac*";
			case Min:		return "ucumacmin";
			case Max: 		return "ucumacmax";
			default: 		return null;
		}
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) {
		return getInstructions(String.valueOf(input_index), String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );

		return sb.toString();
	}
}
