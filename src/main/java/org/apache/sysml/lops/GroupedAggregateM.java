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

import java.util.HashMap;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform mr map-side grouped aggregates 
 * (restriction: sum, w/o weights, ngroups), groups broadcasted
 * 
 */
public class GroupedAggregateM extends Lop 
{	
	public static final String OPCODE = "mapgroupedagg";

	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}
	
	private HashMap<String, Lop> _inputParams;
	private CacheType _cacheType = null;
		
	public GroupedAggregateM(HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, boolean partitioned, ExecType et) {		
		super(Lop.Type.GroupedAggM, dt, vt);
		init(inputParameterLops, dt, vt, et);
		_inputParams = inputParameterLops;
		_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
	}

	/**
	 * 
	 * @param inputParameterLops
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) 
	{
		addInput(inputParameterLops.get(Statement.GAGG_TARGET));
		inputParameterLops.get(Statement.GAGG_TARGET).addOutput(this);
		addInput(inputParameterLops.get(Statement.GAGG_GROUPS));
		inputParameterLops.get(Statement.GAGG_GROUPS).addOutput(this);
		
		if( et == ExecType.MR )
		{
			//setup MR parameters
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //SPARK
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
		return "Operation = MapGroupedAggregate";
	}
	
	@Override
	public String getInstructions(int input1, int input2, int output) 
	{
		return getInstructions(
			String.valueOf(input1),
			String.valueOf(input2),
			String.valueOf(output) );
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) 
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
	
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
	
		sb.append( OPERAND_DELIMITOR );
		sb.append( _inputParams.get(Statement.GAGG_NUM_GROUPS)
				.prepScalarInputOperand(getExecType()) );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( _cacheType.toString() );
	
		return sb.toString();
	}
	
	@Override
	public boolean usesDistributedCache() 
	{
		if( getExecType()==ExecType.MR )
			return true;
		else
			return false;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{
		if( getExecType()==ExecType.MR )
			return new int[]{2};
		else
			return new int[]{-1};
	}
}