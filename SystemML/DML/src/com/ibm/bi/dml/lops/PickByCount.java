package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


public class PickByCount extends Lops {
	
	public enum OperationTypes {VALUEPICK, RANGEPICK, IQM};	
	OperationTypes operation;
	boolean inMemoryInput = false;
	
	
	private void init(Lops input1, Lops input2, OperationTypes op, ExecType et) {
		this.addInput(input1);
		input1.addOutput(this);
		
		if ( input2 != null ) {
			this.addInput(input2);
			input2.addOutput(this);
		}
		
		operation = op;
		
		/*
		 * This lop can be executed only in RecordReader of GMR job.
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			this.lps.setProperties( et, ExecLocation.RecordReader, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	/*
	 * valuepick: first input is always a matrix, second input can either be a scalar or a matrix
	 * rangepick: first input is always a matrix, second input is always a scalar
	 */
	public PickByCount(Lops input1, Lops input2, DataType dt, ValueType vt, OperationTypes op) {
		this(input1, input2, dt, vt, op, ExecType.MR);
	}
	
	public PickByCount(Lops input1, Lops input2, DataType dt, ValueType vt, OperationTypes op, ExecType et) {
		super(Lops.Type.PickValues, dt, vt);
		init(input1, input2, op, et);
	}

	public PickByCount(Lops input1, Lops input2, DataType dt, ValueType vt, OperationTypes op, ExecType et, boolean inMemoryInput) {
		super(Lops.Type.PickValues, dt, vt);
		this.inMemoryInput = inMemoryInput;
		init(input1, input2, op, et);
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	public OperationTypes getOperationType() {
		return operation;
	}

	/*
	 * This version of getInstruction() must be called only for valuepick (MR) and rangepick
	 * 
	 * Example instances:
	 * valupick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE
	 * rangepick:::0:DOUBLE:::0.25:DOUBLE:::1:DOUBLE
	 * rangepick:::0:DOUBLE:::Var3:DOUBLE:::1:DOUBLE
	 */
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException
	{
		
		if ( operation == OperationTypes.RANGEPICK ) {
			// check the scalar input
			if ( this.getInputs().get(1).get_dataType() == DataType.SCALAR ) {
				String valueLabel = this.getInputs().get(1).getOutputParameters().getLabel();
				String valueString = "";

				/**
				 * if it is a literal, copy val, else surround with the label with
				 * ## symbols. these will be replaced at runtime.
				 */
				if(this.getInputs().get(1).getExecLocation() == ExecLocation.Data && 
						((Data)this.getInputs().get(1)).isLiteral())
					valueString = "" + valueLabel;
				else
					valueString = "##" + valueLabel + "##";
				return getInstructions(""+input_index1, valueString, ""+output_index);
			}
		}
		return getInstructions(""+input_index1, ""+input_index2, ""+output_index);

	}

	/*
	 * This version of getInstructions() must be called only for valuepick (CP), IQM (CP)
	 * 
	 * Example instances:
	 * valuepick:::temp2:STRING:::0.25:DOUBLE:::Var1:DOUBLE
	 * valuepick:::temp2:STRING:::Var1:DOUBLE:::Var2:DOUBLE
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException
	{
		
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		String opString = new String ("");
		if ( operation == OperationTypes.VALUEPICK)
			opString = (inMemoryInput ? "inmem-valuepick" : "valuepick");
		else if ( operation == OperationTypes.RANGEPICK ) {
			//opString = "rangepick";
			opString = (inMemoryInput ? "inmem-rangepick" : "rangepick");
			if ( this.getInputs().get(1).get_dataType() != DataType.SCALAR  )
				throw new LopsException(this.printErrorLocation() + "In PickByCount Lop, Unexpected input datatype " + this.getInputs().get(1).get_dataType() + " for rangepick: expecting a SCALAR.");
		}
		else if ( operation == OperationTypes.IQM ) {
			if ( !inMemoryInput ) {
				throw new LopsException(this.printErrorLocation() + "Pick.IQM in can only execute in Control Program on in-memory matrices.");
			}
			opString = "inmem-iqm";
		}
		else
			throw new LopsException(this.printErrorLocation() + "Invalid operation specified for PickByCount: " + operation);
		
		inst += opString + OPERAND_DELIMITOR
					+ input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR
					+ input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR
					+ output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType();

		return inst;
	}
	
	/**
	 * This version of getInstructions() is called for IQM, executing in CP
	 * 
	 * Example instances:
	 *   iqm:::input:::output
	 */
	@Override
	public String getInstructions(String input, String output) throws LopsException {
		String inst = "";
		inst = getExecType() + Lops.OPERAND_DELIMITOR 
				+ "inmem-iqm" + Lops.OPERAND_DELIMITOR
				+ input + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR
				+ output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;
		return inst;
	}
}
