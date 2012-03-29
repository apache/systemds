package dml.lops;

import dml.utils.LopsException;
import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;

public class Append extends Lops{

	public Append(Lops input1, Lops input2, Lops input3, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Append, dt, vt);		
		
		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);
		
		this.addInput(input3);
		input3.addOutput(this);
		
		//confirm this
		lps.addCompatibility(JobType.REBLOCK_TEXT);
		lps.addCompatibility(JobType.REBLOCK_BINARY);
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		this.lps.setProperties( ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}
	
	@Override
	public String toString() {

		return " Append: ";
	}

	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException
	{
		String offsetString;
		
		String offsetLabel = this.getInputs().get(2).getOutputParameters().getLabel();
		
		if (this.getInputs().get(2).getExecLocation() == ExecLocation.Data
			&& ((Data) this.getInputs().get(2)).isLiteral())
			offsetString = "" + offsetLabel;
		else
			offsetString = "##" + offsetLabel + "##";
		
		return "append" + OPERAND_DELIMITOR + 
				input_index1 + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				input_index2 + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				output_index + VALUETYPE_PREFIX + this.get_valueType() + OPERAND_DELIMITOR +
				offsetString;	
	}	
}
