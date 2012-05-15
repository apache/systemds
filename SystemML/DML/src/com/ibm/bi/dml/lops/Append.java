package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


public class Append extends Lops{

	public Append(Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lops.Type.Append, dt, vt);
		init(input1, input2, input3, dt, vt, et);
	}
	
	public Append(Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Append, dt, vt);		
		init(input1, input2, input3, dt, vt, ExecType.MR);
	}
	
	public void init(Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);
		
		this.addInput(input3);
		input3.addOutput(this);
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			//confirm this
			lps.addCompatibility(JobType.REBLOCK_TEXT);
			lps.addCompatibility(JobType.REBLOCK_BINARY);
			this.lps.setProperties( et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	@Override
	public String toString() {

		return " Append: ";
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException
	{
		return getInstructions(input_index1+"",
							   input_index2+"",
							   input_index3+"",
							   output_index+"");	
	}
	
	//....in CP
	public String getInstructions(String input_index1, String input_index2, String input_index3, String output_index) throws LopsException
	{
		String offsetString;
		String offsetLabel = this.getInputs().get(2).getOutputParameters().getLabel();
	
		if (getExecType() == ExecType.MR ) {
			if (this.getInputs().get(2).getExecLocation() == ExecLocation.Data
				&& ((Data) this.getInputs().get(2)).isLiteral())
				offsetString = "" + offsetLabel;
			else
				offsetString = "##" + offsetLabel + "##";
		}
		else {
			offsetString = "" + offsetLabel;
		}
		
		return this.lps.execType 
					 + OPERAND_DELIMITOR + "append" 
					 + OPERAND_DELIMITOR + input_index1 + DATATYPE_PREFIX + this.getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() 
					 + OPERAND_DELIMITOR + input_index2 + DATATYPE_PREFIX + this.getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + this.getInputs().get(1).get_valueType() 
					 + OPERAND_DELIMITOR + output_index + DATATYPE_PREFIX + this.get_dataType() + VALUETYPE_PREFIX + this.get_valueType() 
					 + OPERAND_DELIMITOR + offsetString;	
	}
}
