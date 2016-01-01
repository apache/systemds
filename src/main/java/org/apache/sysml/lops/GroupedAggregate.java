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
import java.util.Map.Entry;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform grouped aggregates
 * 
 */
public class GroupedAggregate extends Lop 
{	
	private HashMap<String, Lop> _inputParams;
	private static final String opcode = "groupedagg";
	public static final String COMBINEDINPUT = "combinedinput";
	
	private boolean _weights = false;	
	
	//spark-specific parameters
	private boolean _broadcastGroups = false;
	
	//cp-specific parameters
	private int _numThreads = 1;

	/**
	 * Constructor to perform grouped aggregate.
	 * inputParameterLops <- parameters required to compute different aggregates (hashmap)
	 *   "combinedinput" -- actual data
	 *   "function" -- aggregate function
	 */
	
	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, boolean weights,
			DataType dt, ValueType vt) {		
		this(inputParameterLops, dt, vt, ExecType.MR);
		_weights = weights;
	}

	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.GroupedAgg, dt, vt);
		init(inputParameterLops, dt, vt, et);
	}
	
	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et, boolean broadcastGroups) {
		super(Lop.Type.GroupedAgg, dt, vt);
		init(inputParameterLops, dt, vt, et);
		_broadcastGroups = broadcastGroups;
	}
	
	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et, int k) {
		super(Lop.Type.GroupedAgg, dt, vt);
		init(inputParameterLops, dt, vt, et);
		_numThreads = k;
	}
	
	/**
	 * 
	 * @param inputParameterLops
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) {
		if ( et == ExecType.MR ) {
			/*
			 * Inputs to ParameterizedBuiltinOp can be in an arbitrary order. However,
			 * piggybacking (Dag.java:getAggAndOtherInstructions()) expects the first 
			 * input to be the data (named as "combinedinput") on which the aggregate 
			 * needs to be computed. Make sure that "combinedinput" is the first input
			 * to GroupedAggregate lop. 
			 */
			this.addInput(inputParameterLops.get(COMBINEDINPUT));
			inputParameterLops.get(COMBINEDINPUT).addOutput(this);
			
			// process remaining parameters
			for ( Entry<String, Lop> e : inputParameterLops.entrySet() ) {
				String k = e.getKey();
				Lop lop = e.getValue();
				if ( !k.equalsIgnoreCase(COMBINEDINPUT) ) {
					this.addInput(lop);
					lop.addOutput(this);
				}
			}
			
			_inputParams = inputParameterLops;
			
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = true;
			lps.addCompatibility(JobType.GROUPED_AGG);
			this.lps.setProperties(inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			
			// First, add inputs corresponding to "target" and "groups"
			this.addInput(inputParameterLops.get(Statement.GAGG_TARGET));
			inputParameterLops.get(Statement.GAGG_TARGET).addOutput(this);
			this.addInput(inputParameterLops.get(Statement.GAGG_GROUPS));
			inputParameterLops.get(Statement.GAGG_GROUPS).addOutput(this);
			
			// process remaining parameters
			for ( Entry<String, Lop> e : inputParameterLops.entrySet() ) {
				String k = e.getKey();
				Lop lop = e.getValue();
				if ( !k.equalsIgnoreCase(Statement.GAGG_TARGET) && !k.equalsIgnoreCase(Statement.GAGG_GROUPS) ) {
					this.addInput(lop);
					lop.addOutput(this);
				}
			}
			_inputParams = inputParameterLops;
			
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	@Override
	public String toString() {

		return "Operation = GroupedAggregate";
	}

	/**
	 * Function to generate CP Grouped Aggregate Instructions.
	 * 
	 */
	@Override
	public String getInstructions(String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( opcode );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		if ( _inputParams.get(Statement.GAGG_TARGET) == null || _inputParams.get(Statement.GAGG_GROUPS) == null || _inputParams.get("fn") == null ) 
			throw new LopsException(this.printErrorLocation() + "Invalid parameters to groupedAggregate -- \"target\", \"groups\", \"fn\" must be provided");
		
		String targetVar = _inputParams.get(Statement.GAGG_TARGET).getOutputParameters().getLabel();
		String groupsVar = _inputParams.get(Statement.GAGG_GROUPS).getOutputParameters().getLabel();
		
		sb.append( Statement.GAGG_TARGET );
		sb.append( Lop.NAME_VALUE_SEPARATOR );
		sb.append( targetVar );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( Statement.GAGG_GROUPS );
		sb.append( Lop.NAME_VALUE_SEPARATOR );
		sb.append( groupsVar );
		
		if ( _inputParams.get(Statement.GAGG_WEIGHTS) != null )
		{
			sb.append( Lop.OPERAND_DELIMITOR );
			sb.append( Statement.GAGG_WEIGHTS );
			sb.append( Lop.NAME_VALUE_SEPARATOR );
			sb.append( _inputParams.get(Statement.GAGG_WEIGHTS).getOutputParameters().getLabel() );
		}
		
		// Process all other name=value parameters, which are scalars
		String name, valueString;
		Lop value;
		for(Entry<String, Lop>  e : _inputParams.entrySet()) {
			name = e.getKey();
			if ( !name.equalsIgnoreCase(Statement.GAGG_TARGET) && !name.equalsIgnoreCase(Statement.GAGG_GROUPS) && !name.equalsIgnoreCase(Statement.GAGG_WEIGHTS) ) {
				value =  e.getValue();
				valueString = value.prepScalarLabel();
				
				sb.append( OPERAND_DELIMITOR );
				sb.append( name );
				sb.append( Lop.NAME_VALUE_SEPARATOR );
				sb.append( valueString );
			}
		}
		
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( "k" );
			sb.append( Lop.NAME_VALUE_SEPARATOR );
			sb.append( _numThreads );	
		}
		else if( getExecType()==ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( "broadcast" );
			sb.append( Lop.NAME_VALUE_SEPARATOR );
			sb.append( _broadcastGroups );	
		}
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( opcode );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index));
				
		// get the aggregate function
		sb.append( OPERAND_DELIMITOR );
		Lop funcLop = _inputParams.get(Statement.GAGG_FN); 
		sb.append( funcLop.prepScalarInputOperand(getExecType()));
		
		// get the "optional" parameters
		if ( _inputParams.get(Statement.GAGG_FN_CM_ORDER) != null ) {
			sb.append( OPERAND_DELIMITOR );
			Lop orderLop = _inputParams.get(Statement.GAGG_FN_CM_ORDER); 
			sb.append( orderLop.prepScalarInputOperand(getExecType()));
		}
		
		// add output_index to instruction
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output_index) );
	
		sb.append( OPERAND_DELIMITOR );
		sb.append( _weights );
			
		return sb.toString();
	}
	

}