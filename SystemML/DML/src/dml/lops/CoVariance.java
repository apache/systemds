package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

/**
 * Lop to compute covariance between two 1D matrices
 * 
 * @author statiko
 */
public class CoVariance extends Lops {

			/**
			 * Constructor to perform covariance.
			 * input1 <- data 
			 * (prior to this lop, input vectors need to attached together using CombineBinary or CombineTertiary) 
			 */

	public CoVariance(Lops input1, DataType dt, ValueType vt) {
		super(Lops.Type.CoVariance, dt, vt);
		this.addInput(input1);
		input1.addOutput(this);

		/*
		 * This lop can be executed only in CM job.
		 */

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.CM_COV);
		this.lps.setProperties(ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {

		return "Operation = coVariance";
	}

	@Override
	public String getInstructions(int input_index, int output_index) {
		String inst = new String("");
		inst += "cov" + OPERAND_DELIMITOR + 
				input_index + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				output_index + VALUETYPE_PREFIX + this.get_valueType();
		return inst;
	}

}