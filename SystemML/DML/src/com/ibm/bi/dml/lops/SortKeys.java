package com.ibm.bi.dml.lops;

import java.util.HashSet;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

public class SortKeys extends Lops 
{

	public enum OperationTypes { WithWeights, WithNoWeights };
	OperationTypes operation;
	
	private void init(Lops input1, Lops input2, OperationTypes op, ExecType et) {
		this.addInput(input1);
		input1.addOutput(this);
		
		operation = op;
		
		if ( et == ExecType.MR ) {
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = true;
			
			lps.addCompatibility(JobType.SORT);
			this.lps.setProperties( et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			// SortKeys can accept a optional second input only when executing in CP
			// Example: sorting with weights inside CP
			if ( input2 != null ) {
				this.addInput(input2);
				input2.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, false, false, false);
		}
	}
	
	public SortKeys(Lops input, OperationTypes op, DataType dt, ValueType vt) {
		super(Lops.Type.SortKeys, dt, vt);		
		init(input, null, op, ExecType.MR);
	}

	public SortKeys(Lops input, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lops.Type.SortKeys, dt, vt);		
		init(input, null, op, et);
	}

	public SortKeys(Lops input1, Lops input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lops.Type.SortKeys, dt, vt);		
		init(input1, input2, op, et);
	}

	@Override
	public String toString() {
		return "Operation: SortKeys (" + operation + ")";
	}

	@Override
	public String getInstructions(int input_index, int output_index)
	{
		return getInstructions(""+input_index, ""+output_index);
	}
	
	@Override
	public String getInstructions(String input, String output)
	{
		String opString = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		opString += "sort";
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + this.get_valueType() ;
		return inst;
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		String inst = getExecType() + Lops.OPERAND_DELIMITOR 
						+ "sort" + Lops.OPERAND_DELIMITOR
						+ input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + Lops.OPERAND_DELIMITOR
						+ input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + Lops.OPERAND_DELIMITOR
						+ output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType();
		return inst;
	}
	
	// This method is invoked in two cases:
	// 1) SortKeys (both weighted and unweighted) executes in MR
	// 2) Unweighted SortKeys executes in CP
	public static SortKeys constructSortByValueLop(Lops input1, OperationTypes op, 
			DataType dt, ValueType vt, ExecType et) {
		
		for (Lops lop  : input1.getOutputs()) {
			if ( lop.type == Lops.Type.SortKeys ) {
				return (SortKeys)lop;
			}
		}
		
		return new SortKeys(input1, op, dt, vt, et);
	}

	// This method is invoked ONLY for the case of Weighted SortKeys executing in CP
	public static SortKeys constructSortByValueLop(Lops input1, Lops input2, OperationTypes op, 
			DataType dt, ValueType vt, ExecType et) {
		
		HashSet<Lops> set1 = new HashSet<Lops>();
		set1.addAll(input1.getOutputs());
		// find intersection of input1.getOutputs() and input2.getOutputs();
		set1.retainAll(input2.getOutputs());
		
		for (Lops lop  : set1) {
			if ( lop.type == Lops.Type.SortKeys ) {
				return (SortKeys)lop;
			}
		}
		
		return new SortKeys(input1, input2, op, dt, vt, et);
	}


}
