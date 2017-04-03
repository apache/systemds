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
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;


/**
 * Lop to perform data partitioning.
 */
public class DataPartition extends Lop 
{
	
	public static final String OPCODE = "partition"; 
	
	private PDataPartitionFormat _pformat = null;
	
	public DataPartition(Lop input, DataType dt, ValueType vt, ExecType et, PDataPartitionFormat pformat) 
		throws LopsException 
	{
		super(Lop.Type.DataPartition, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		_pformat = pformat;
		
		//setup lop properties
		ExecLocation eloc = (et==ExecType.MR)? ExecLocation.MapAndReduce : ExecLocation.ControlProgram;
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = (et==ExecType.MR);
		lps.addCompatibility(JobType.DATA_PARTITION);
		lps.setProperties( inputs, et, eloc, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "DataPartition";
	}

	@Override
	public String getInstructions(String input_index, String output_index) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _pformat.toString() );
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) {
		return getInstructions(String.valueOf(input_index), String.valueOf(output_index));
	}
}