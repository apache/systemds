/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to perform grouped aggregates
 * 
 */
public class GroupedAggregate extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashMap<String, Lop> _inputParams;
	private static final String opcode = "groupedagg";
	public static final String COMBINEDINPUT = "combinedinput";
	
	/**
	 * Constructor to perform grouped aggregate.
	 * inputParameterLops <- parameters required to compute different aggregates (hashmap)
	 *   "combinedinput" -- actual data
	 *   "function" -- aggregate function
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
	
	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt) {
		this(inputParameterLops, dt, vt, ExecType.MR);
	}

	public GroupedAggregate(
			HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.GroupedAgg, dt, vt);
		init(inputParameterLops, dt, vt, et);
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
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	/*@Override
	public String getInstructions(String input1, String input2, String output) 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "groupedagg" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).getValueType() );
		sb.append( input2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).getValueType() );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		
		return sb.toString();
	}*/
	
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
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}
	

}