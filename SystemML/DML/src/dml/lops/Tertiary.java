package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.utils.LopsException;

/**
 * Lop to perform tertiary operation. All inputs must be matrices or vectors. 
 * For example, this lop is used in evaluating A = ctable(B,C,W)
 * 
 * Currently, this lop is used only in case of CTABLE functionality.
 */

public class Tertiary extends Lops 
{
	public enum OperationTypes { CTABLE_TRANSFORM, CTABLE_TRANSFORM_SCALAR_WEIGHT, CTABLE_TRANSFORM_HISTOGRAM, CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM };	
	OperationTypes operation;
	

	
	/**
	 * Constructor to perform a binary operation.
	 * @param input
	 * @param op
	 */

	public Tertiary(Lops input1, Lops input2, Lops input3, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Tertiary, dt, vt);	
		operation = op;
		this.addInput(input1);
		this.addInput((Lops)input2);
		this.addInput((Lops)input3);
		input1.addOutput(this);
		((Lops)input2).addOutput(this);
		((Lops)input3).addOutput(this);
		
		/*
		 *  This lop can be executed in GMR, RAND, REBLOCK jobs
		 */
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.RAND);
		lps.addCompatibility(JobType.REBLOCK_BINARY);
		lps.addCompatibility(JobType.REBLOCK_TEXT);
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		this.lps.setProperties( ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return " Operation: " + operation;

	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException
	{
		String inst = "";
		switch(operation) {
		/* Arithmetic */
		case CTABLE_TRANSFORM:
			// F = ctable(A,B,W)
			inst += "ctabletransform" + OPERAND_DELIMITOR + 
					input_index1 + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
					input_index2 + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
					input_index3 + VALUETYPE_PREFIX + getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
			        output_index + VALUETYPE_PREFIX + get_valueType() ;
			
			break;
		
		case CTABLE_TRANSFORM_SCALAR_WEIGHT:
			// F = ctable(A,B) or F = ctable(A,B,1)
			// third input must be a scalar, and hence input_index3 == -1
			if ( input_index3 != -1 ) {
				throw new LopsException("Unexpected input while computing the instructions for op: " + operation);
			}
			
			// parse the third input (scalar)
			// if it is a literal, copy val, else surround with the label with
			// ## symbols. these will be replaced at runtime.
			
			int scalarIndex = 2; // index of the scalar input
			String valueLabel = null;
			if(this.getInputs().get(scalarIndex).getExecLocation() == ExecLocation.Data && 
					((Data)this.getInputs().get(scalarIndex)).isLiteral())
				valueLabel = getInputs().get(scalarIndex).getOutputParameters().getLabel();
			else
				valueLabel = "##" + getInputs().get(scalarIndex).getOutputParameters().getLabel() + "##";
			
			inst += "ctabletransformscalarweight" + OPERAND_DELIMITOR + 
			input_index1 + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			input_index2 + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
			valueLabel + VALUETYPE_PREFIX + getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
	        output_index + VALUETYPE_PREFIX + get_valueType() ;
			break;
			
		case CTABLE_TRANSFORM_HISTOGRAM:
			// F=ctable(A,1) or F = ctable(A,1,1)
			if ( input_index2 != -1 || input_index3 != -1)
				throw new LopsException("Unexpected input while computing the instructions for op: " + operation);
			
			// parse the scalar inputs (2nd and 3rd inputs)
			String scalar2=null;
			if(this.getInputs().get(1).getExecLocation() == ExecLocation.Data && 
					((Data)this.getInputs().get(1)).isLiteral())
				scalar2 = getInputs().get(1).getOutputParameters().getLabel();
			else
				scalar2 = "##" + getInputs().get(1).getOutputParameters().getLabel() + "##";
			
			String scalar3=null;
			if(this.getInputs().get(2).getExecLocation() == ExecLocation.Data && 
					((Data)this.getInputs().get(2)).isLiteral())
				scalar3 = getInputs().get(2).getOutputParameters().getLabel();
			else
				scalar3 = "##" + getInputs().get(2).getOutputParameters().getLabel() + "##";
			
			inst += "ctabletransformhistogram" + OPERAND_DELIMITOR + 
			input_index1 + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			scalar2 + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
			scalar3 + VALUETYPE_PREFIX + getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
	        output_index + VALUETYPE_PREFIX + get_valueType() ;
			
			break;
		
		case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM:
			// F=ctable(A,1,W)
			if ( input_index2 != -1 )
				throw new LopsException("Unexpected input while computing the instructions for op: " + operation);
			// parse the scalar inputs (2nd and 3rd inputs)
			String scalarInput2=null;
			if(this.getInputs().get(1).getExecLocation() == ExecLocation.Data && 
					((Data)this.getInputs().get(1)).isLiteral())
				scalarInput2 = getInputs().get(1).getOutputParameters().getLabel();
			else
				scalarInput2 = "##" + getInputs().get(1).getOutputParameters().getLabel() + "##";
			inst += "ctabletransformweightedhistogram" + OPERAND_DELIMITOR + 
					input_index1 + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
					scalarInput2 + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
					input_index3 + VALUETYPE_PREFIX + getInputs().get(2).get_valueType() + OPERAND_DELIMITOR + 
					output_index + VALUETYPE_PREFIX + get_valueType() ;
			
			break;
			
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Tertiary operation: " + operation);
		}
		
		return inst;
	}

 
}