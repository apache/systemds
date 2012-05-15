package com.ibm.bi.dml.lops;

import java.util.HashSet;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineTertiary extends Lops {

	public enum OperationTypes {
		PreCovWeighted, PreGroupedAggWeighted
	}; // PreCovUnweighted,PreGroupedAggWeighted will be CombineBinary

	OperationTypes operation;

	/**
	 * @param input
	 *            - input lop
	 * @param op
	 *            - operation type
	 */

	public CombineTertiary( OperationTypes op, Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt) {
		super(Lops.Type.CombineTertiary, dt, vt);
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		this.addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);

		/*
		 * This lop can ONLY be executed as a STANDALONE job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.COMBINE);
		this.lps.setProperties(ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	/**
	 * for debugging purposes.
	 */

	public String toString() {
		return "combinetertiary";
	}

	@Override
	public String getInstructions(int input_index1, int input_index2,
			int input_index3, int output_index) throws LopsException {
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);

		inst += "combinetertiary" + OPERAND_DELIMITOR + 
				input_index1 + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				input_index2 + VALUETYPE_PREFIX + this.getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
				input_index3 + VALUETYPE_PREFIX + this.getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
				output_index + VALUETYPE_PREFIX + this.get_valueType();

		return inst;
	}

	public OperationTypes getOperation() {
		return operation;
	}

	public static CombineTertiary constructCombineLop( OperationTypes op, Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt) {

		HashSet<Lops> set1 = new HashSet<Lops>();
		set1.addAll(input1.getOutputs());

		// find intersection of input1.getOutputs() and input2.getOutputs()
		set1.retainAll(input2.getOutputs());

		// find intersection of the above result and input3.getOutputs()
		set1.retainAll(input3.getOutputs());
		
		for (Lops lop : set1) {
			if (lop.type == Lops.Type.CombineTertiary) {
				CombineTertiary combine = (CombineTertiary) lop;
				if (combine.operation == op)
					return (CombineTertiary) lop;
			}
		}

		CombineTertiary comn = new CombineTertiary(op, input1, input2, input3, dt, vt);
		return comn;
	}

}
