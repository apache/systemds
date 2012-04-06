package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

/**
 * Lop to perform cross product operation
 * 
 * @author statiko
 */
public class CentralMoment extends Lops {

	/**
	 * Constructor to perform central moment.
	 * input1 <- data (weighted or unweighted)
	 * input2 <- order (integer: 0, 2, 3, or 4)
	 */

	public CentralMoment(
			Lops input1,
			Lops input2,
			DataType dt, ValueType vt) {
		super(Lops.Type.CentralMoment, dt, vt);
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		/*
		 * This lop can be executed only in CM job.
		 */

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.CM_COV);
		this.lps.setProperties(ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {

		return "Operation = CentralMoment";
	}

	@Override
	public String getInstructions(int input_index, int output_index) {
		String opString = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		opString += "cm";

		// get label for scalar input -- the "order" for central moment.
		String order = this.getInputs().get(1).getOutputParameters().getLabel();
		String valueString = "";

		/*
		 * if it is a literal, copy val, else surround with the label with
		 * ## symbols. these will be replaced at runtime.
		 */
		if(this.getInputs().get(1).getExecLocation() == ExecLocation.Data && 
				((Data)this.getInputs().get(1)).isLiteral())
			valueString = "" + order;
		else
			valueString = "##" + order + "##";
		
		String inst = new String("");
		// value type for "order" is INT
		inst += opString + OPERAND_DELIMITOR + input_index + VALUETYPE_PREFIX
				+ this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR
				+ valueString + VALUETYPE_PREFIX + ValueType.INT + OPERAND_DELIMITOR
				+ output_index + VALUETYPE_PREFIX + this.get_valueType();
		return inst;
	}

}