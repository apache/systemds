package com.ibm.bi.dml.lops;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Lop to perform grouped aggregates
 * 
 */
public class GroupedAggregate extends Lops {

	private HashMap<String, Lops> _inputParams;
	
	/**
	 * Constructor to perform grouped aggregate.
	 * inputParameterLops <- parameters required to compute different aggregates (hashmap)
	 *   "combinedinput" -- actual data
	 *   "function" -- aggregate function
	 */

	private void init(HashMap<String, Lops> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) {
		if ( et == ExecType.MR ) {
			/*
			 * Inputs to ParameterizedBuiltinOp can be in an arbitrary order. However,
			 * piggybacking (Dag.java:getAggAndOtherInstructions()) expects the first 
			 * input to be the data (named as "combinedinput") on which the aggregate 
			 * needs to be computed. Make sure that "combinedinput" is the first input
			 * to GroupedAggregate lop. 
			 */
			this.addInput(inputParameterLops.get("combinedinput"));
			inputParameterLops.get("combinedinput").addOutput(this);
			
			// process remaining parameters
			for ( String k : inputParameterLops.keySet()) {
				if ( !k.equalsIgnoreCase("combinedinput") ) {
					this.addInput(inputParameterLops.get(k));
					inputParameterLops.get(k).addOutput(this);
				}
			}
			
			_inputParams = inputParameterLops;
			
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = true;
			lps.addCompatibility(JobType.GROUPED_AGG);
			this.lps.setProperties(et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			
			// First, add inputs corresponding to "target" and "groups"
			this.addInput(inputParameterLops.get("target"));
			inputParameterLops.get("target").addOutput(this);
			this.addInput(inputParameterLops.get("groups"));
			inputParameterLops.get("groups").addOutput(this);
			
			// process remaining parameters
			for ( String k : inputParameterLops.keySet()) {
				if ( !k.equalsIgnoreCase("target") && !k.equalsIgnoreCase("groups") ) {
					this.addInput(inputParameterLops.get(k));
					inputParameterLops.get(k).addOutput(this);
				}
			}
			_inputParams = inputParameterLops;
			
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	public GroupedAggregate(
			HashMap<String, Lops> inputParameterLops, 
			DataType dt, ValueType vt) {
		this(inputParameterLops, dt, vt, ExecType.MR);
	}

	public GroupedAggregate(
			HashMap<String, Lops> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) {
		super(Lops.Type.GroupedAgg, dt, vt);
		init(inputParameterLops, dt, vt, et);
	}

	@Override
	public String toString() {

		return "Operation = GroupedAggregate";
	}

	@Override
	// This version of getInstructions() is invoked when groupedAgg is executed in CP
	public String getInstructions(String output) throws LopsException {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "groupedagg";
		
		if ( _inputParams.get("target") == null || _inputParams.get("groups") == null || _inputParams.get("fn") == null ) 
			throw new LopsException("Invalid parameters to groupedAggregate -- \"target\", \"groups\", \"fn\" must be provided");
		
		String targetVar = _inputParams.get("target").getOutputParameters().getLabel();
		String groupsVar = _inputParams.get("groups").getOutputParameters().getLabel();
		
		inst += Lops.OPERAND_DELIMITOR + "target" + Lops.NAME_VALUE_SEPARATOR + targetVar;
		inst += Lops.OPERAND_DELIMITOR + "groups" + Lops.NAME_VALUE_SEPARATOR + groupsVar;
		if ( _inputParams.get("weights") != null )
			inst += Lops.OPERAND_DELIMITOR + "weights" + Lops.NAME_VALUE_SEPARATOR + _inputParams.get("weights").getOutputParameters().getLabel();
		
		// Process all the other parameters, which are scalars
		String name, valueString;
		Lops value;
		for(Entry<String, Lops>  e : _inputParams.entrySet()) {
			name = e.getKey();
			if ( !name.equalsIgnoreCase("target") && !name.equalsIgnoreCase("groups") && !name.equalsIgnoreCase("weights") ) {
				value =  e.getValue();
				
				if ( value.getExecLocation() == ExecLocation.Data && 
						((Data)value).isLiteral() ) {
					valueString = value.getOutputParameters().getLabel();
				}
				else {
					valueString = "##" + value.getOutputParameters().getLabel() + "##";
				}
				inst += OPERAND_DELIMITOR + name + Lops.NAME_VALUE_SEPARATOR + valueString;
			}
		}
		
		inst += OPERAND_DELIMITOR + output + DATATYPE_PREFIX + this.get_dataType() + VALUETYPE_PREFIX + this.get_valueType();
		return inst;
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "groupedagg" + OPERAND_DELIMITOR
		+ input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType()
		+ input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType();
		
		inst += output + DATATYPE_PREFIX + this.get_dataType() + VALUETYPE_PREFIX + this.get_valueType();
		return inst;
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) {

		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		// value type for "order" is INT
		inst += "groupedagg" + OPERAND_DELIMITOR + input_index + VALUETYPE_PREFIX
				+ this.getInputs().get(0).get_valueType();
				
		// get the aggregate function
		Lops funcLop = _inputParams.get("fn"); 
		//OperationTypes op = OperationTypes.INVALID;
		String func = null;
		if ( funcLop.getExecLocation() == ExecLocation.Data && 
				((Data)funcLop).isLiteral() ) {
			func = funcLop.getOutputParameters().getLabel();
		}
		else {
			func = "##" + funcLop.getOutputParameters().getLabel() + "##";
		}
		inst += OPERAND_DELIMITOR + func + VALUETYPE_PREFIX + funcLop.get_valueType();
		
		
		// get the "optional" parameters
		String order = null;
		if ( _inputParams.get("order") != null ) {
			Lops orderLop = _inputParams.get("order"); 
			if ( orderLop.getExecLocation() == ExecLocation.Data && 
					((Data)orderLop).isLiteral() ) {
				order = orderLop.getOutputParameters().getLabel();
			}
			else {
				order = "##" + orderLop.getOutputParameters().getLabel() + "##";
			}
			inst += OPERAND_DELIMITOR + order + VALUETYPE_PREFIX + orderLop.get_valueType();
		}
		
		// add output_index to instruction
		inst += OPERAND_DELIMITOR + output_index + VALUETYPE_PREFIX + get_valueType();
		
		return inst;
	}
	

}