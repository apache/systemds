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


/**
 * TODO Currently this lop only support the right hand side in distributed cache. This
 *  should be generalized (incl hop operator selection) to left/right cache types.
 *  
 * 
 */
public class UAggOuterChain extends Lop 
{
	
	public static final String OPCODE = "uaggouterchain";

	//outer operation
	private Aggregate.OperationTypes _uaggOp         = null;
	private PartialAggregate.DirectionTypes _uaggDir = null;
	//inner operation
	private Binary.OperationTypes _binOp             = null;	
		
	
	/**
	 * Constructor to setup a unaryagg outer chain
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param uaop aggregate operation type
	 * @param uadir partial aggregate direction type
	 * @param bop binary operation type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 * @throws LopsException if LopsException occurs
	 */
	public UAggOuterChain(Lop input1, Lop input2, Aggregate.OperationTypes uaop, PartialAggregate.DirectionTypes uadir, Binary.OperationTypes bop, DataType dt, ValueType vt, ExecType et) 
		throws LopsException 
	{
		super(Lop.Type.UaggOuterChain, dt, vt);		
		addInput(input1);
		addInput(input2);
		input1.addOutput(this); 
		input2.addOutput(this); 
		
		//setup operator types
		_uaggOp = uaop;
		_uaggDir = uadir;
		_binOp = bop;
		
		//setup MR parameters 
		if( et == ExecType.MR )
		{
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			lps.addCompatibility(JobType.CSV_REBLOCK);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //SPARK
		{
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	

	public String toString() {
		return "Operation = UaggOuterChain";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) {
		return getInstructions(String.valueOf(input_index1), 
				String.valueOf(input_index2), String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		StringBuilder sb = new StringBuilder();
		
		//exec type
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		//inst op code
		sb.append(OPCODE);
		sb.append(Lop.OPERAND_DELIMITOR);

		//outer operation op code
		sb.append(PartialAggregate.getOpcode(_uaggOp, _uaggDir));		
		sb.append(Lop.OPERAND_DELIMITOR);

		//inner operation op code
		sb.append(Binary.getOpcode(_binOp));
		sb.append(Lop.OPERAND_DELIMITOR);
				
		//inputs and outputs
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input2));
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output));
				
		return sb.toString();
	}
	
	
	@Override
	public boolean usesDistributedCache() {
		return true;
	}
	
	@Override
	public int[] distributedCacheInputIndex() {
		return new int[]{2};
	}
}
