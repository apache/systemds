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


/**
 * 
 * 
 */
public class CumulativeOffsetBinary extends Lop 
{
	
	private OperationTypes _op;
	private double _initValue = 0;
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, OperationTypes op, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
	
		init(data, offsets, dt, vt, et);
	}
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, double init, OperationTypes op, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
		
		//in case of Spark, CumulativeOffset includes CumulativeSplit and hence needs the init value
		_initValue = init;
		
		init(data, offsets, dt, vt, et);
	}
	
	/**
	 * 
	 * @param input
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		if( et == ExecType.MR )
		{
			//setup MR parameters
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = false;
			
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.setProperties(inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob);
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

	public String toString() {
		return "CumulativeOffsetBinary";
	}

	/**
	 * 
	 * @param op
	 * @throws LopsException
	 */
	private void checkSupportedOperations(OperationTypes op) 
		throws LopsException
	{
		//sanity check for supported aggregates
		if( !(op == OperationTypes.KahanSum || op == OperationTypes.Product ||
		      op == OperationTypes.Min || op == OperationTypes.Max) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
	}
	
	private String getOpcode() {
		switch( _op ) {
			case KahanSum: 	return "bcumoffk+";
			case Product: 	return "bcumoff*";
			case Min:		return "bcumoffmin";
			case Max: 		return "bcumoffmax";
			default: 		return null;
		}
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index) );

		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output) );
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _initValue );	
		}
		
		return sb.toString();
	}
}
