package dml.lops;

import java.util.HashMap;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

/**
 * Lop to perform grouped aggregates
 * 
 * @author statiko
 */
public class GroupedAggregate extends Lops {

	private HashMap<String, Lops> _inputParams;
	
	/**
	 * Constructor to perform grouped aggregate.
	 * inputParameterLops <- parameters required to compute different aggregates (hashmap)
	 *   "combinedinput" -- actual data
	 *   "function" -- aggregate function
	 */

	public GroupedAggregate(
			HashMap<String, Lops> inputParameterLops, 
			DataType dt, ValueType vt) {
		super(Lops.Type.GroupedAgg, dt, vt);

		/*
		 * First input should be the data on which aggregate has to be computed
		 * This is required due to Dag.java:getAggAndOtherInstructions()
		 * -- in that function, getInstructions() is invoked with inputIndices.get(0)
		 * -- therefore, first input should point to the data! 
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
		
		/*
		 * This lop can be executed only in GROUPED_AGG job.
		 */

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.GROUPED_AGG);
		this.lps.setProperties(ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {

		return "Operation = GroupedAggregate";
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