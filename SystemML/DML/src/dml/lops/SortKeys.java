package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;

public class SortKeys extends Lops 
{

	public enum OperationTypes { WithWeights, WithNoWeights };
	OperationTypes operation;
	
	public SortKeys(Lops input, OperationTypes op, DataType dt, ValueType vt) {
		super(Lops.Type.SortKeys, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		operation = op;
		
		
		/*
		 *  This lop can be executed in only in FULL Sort.
		 *  location is MapAndReduce.
		 */
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = true;

		lps.addCompatibility(JobType.SORT);
		this.lps.setProperties( ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String toString() {
		return "Operation: SortKeys (" + operation + ")";
	}

	@Override
	public String getInstructions(int input_index, int output_index)
	{
		String opString = new String("");
		opString += "sortkeys";
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input_index + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
		return inst;
	}
	
	public static SortKeys constructSortByValueLop(Lops input1, OperationTypes op, 
			DataType dt, ValueType vt) {
		
		for (Lops lop  : input1.getOutputs()) {
			if ( lop.type == Lops.Type.SortKeys ) {
				return (SortKeys)lop;
			}
		}
		
		return new SortKeys(input1, op, dt, vt);
	}


}
