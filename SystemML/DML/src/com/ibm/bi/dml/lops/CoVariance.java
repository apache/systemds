package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;

/**
 * Lop to compute covariance between two 1D matrices
 * 
 * @author statiko
 */
public class CoVariance extends Lops {

	private void init(Lops input1, Lops input2, Lops input3, ExecType et) throws LopsException {
		this.addInput(input1);
		input1.addOutput(this);

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.CM_COV);
			this.lps.setProperties(et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			definesMRJob = false;
			if ( input2 == null ) {
				throw new LopsException("Invalid inputs to covariance lop.");
			}
			this.addInput(input2);
			input2.addOutput(this);
			if ( input3 != null ) {
				this.addInput(input3);
				input3.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	/**
	 * Constructor to perform covariance.
	 * input1 <- data 
	 * (prior to this lop, input vectors need to attached together using CombineBinary or CombineTertiary) 
	 * @throws LopsException 
	 */

	public CoVariance(Lops input1, DataType dt, ValueType vt) throws LopsException {
		this(input1, dt, vt, ExecType.MR);
	}

	public CoVariance(Lops input1, DataType dt, ValueType vt, ExecType et) throws LopsException {
		super(Lops.Type.CoVariance, dt, vt);
		init(input1, null, null, et);
	}
	
	public CoVariance(Lops input1, Lops input2, DataType dt, ValueType vt, ExecType et) throws LopsException {
		this(input1, input2, null, dt, vt, et);
	}
	
	public CoVariance(Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt, ExecType et) throws LopsException {
		super(Lops.Type.CoVariance, dt, vt);
		init(input1, input2, input3, et);
	}

	@Override
	public String toString() {

		return "Operation = coVariance";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "cov" + OPERAND_DELIMITOR + 
			input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
			output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType();
		return inst;
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "cov" + OPERAND_DELIMITOR + 
			input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
			input3 + DATATYPE_PREFIX + getInputs().get(2).get_dataType() + VALUETYPE_PREFIX + getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
			output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType();
		return inst;
	}

	@Override
	public String getInstructions(int input_index, int output_index) {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "cov" + OPERAND_DELIMITOR + 
				input_index + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				output_index + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType();
		return inst;
	}

}