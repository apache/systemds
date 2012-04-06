package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.JobType;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.utils.LopsException;

public class PickByCount <KIN extends KEY, VIN extends VAL, KOUT extends KEY, VOUT extends VAL> 
extends Lops 
{
	
	public enum OperationTypes {VALUEPICK, RANGEPICK};	
	OperationTypes operation;
	
	/*
	 * valuepick: first input is always a matrix, second input can either be a scalar or a matrix
	 * rangepick: first input is always a matrix, second input is always a scalar
	 */
	public PickByCount(Lops input1, Lops input2, DataType dt, ValueType vt, OperationTypes op) {
		super(Lops.Type.PickValues, dt, vt);
		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		operation = op;
		
		/*
		 * This lop can be executed only in RecordReader of GMR job.
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		ExecLocation location = ExecLocation.RecordReader;
		ExecType et = ExecType.MR;
		
		// TODO: this logic should happen in the hop layer
		
		if ( op == OperationTypes.VALUEPICK && input2.get_dataType() == DataType.SCALAR ) {
			// if the second input to PickByCount == SCALAR then this lop should get executed in control program
			location = ExecLocation.ControlProgram;
			et = ExecType.CP;
			lps.addCompatibility(JobType.INVALID);
		}
		else {
			lps.addCompatibility(JobType.GMR);
		}
		this.lps.setProperties( et, location, breaksAlignment, aligner, definesMRJob );
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
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		String opString = new String ("");
		ValueType vtype1 = this.getInputs().get(0).get_valueType();
		ValueType vtype2 = this.getInputs().get(1).get_valueType();
		
		switch(operation) {
		case VALUEPICK:	
			opString = "valuepick";
			inst += opString + OPERAND_DELIMITOR + 
					input_index1 + VALUETYPE_PREFIX + vtype1 + OPERAND_DELIMITOR + 
					input_index2 + VALUETYPE_PREFIX + vtype2 + OPERAND_DELIMITOR + 
			        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
			return inst;

		case RANGEPICK:
			// second input is a scalar
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
				
				opString = "rangepick";
				inst += opString + OPERAND_DELIMITOR + 
						input_index1 + VALUETYPE_PREFIX + vtype1 + OPERAND_DELIMITOR + 
						valueString + VALUETYPE_PREFIX + vtype2 + OPERAND_DELIMITOR + 
				        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
				return inst;
			}
			else {
				throw new LopsException("Unexpected input datatype " + this.getInputs().get(1).get_dataType() + " for rangepick: expecting a SCALAR.");
			}
			
		default:
			throw new LopsException("Invalid operation while creating a PickByCount instruction: " + operation);
		}

	}

	/*
	 * This version of getInstructions() must be called only for valuepick (CP)
	 * 
	 * Example instances:
	 * valuepickCP:::temp2:STRING:::0.25:DOUBLE:::Var1:DOUBLE
	 * valuepickCP:::temp2:STRING:::Var1:DOUBLE:::Var2:DOUBLE
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException
	{
		String opString = new String ("");
		ValueType vtype1 = this.getInputs().get(0).get_valueType();
		ValueType vtype2 = this.getInputs().get(1).get_valueType();
		
		switch(operation) {
		case VALUEPICK:	
			if ( this.getExecLocation() == ExecLocation.ControlProgram ) {
				opString = "valuepickCP";
				vtype1 = ValueType.STRING; // first input is a String (a filename)
			}
			else
				throw new LopsException("Instruction not defined for PickByCount opration: " + operation);
				
			break;
		
		default:
			throw new LopsException("Instruction not defined for PickByCount opration: " + operation);
		}
		
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += opString + OPERAND_DELIMITOR + 
				input1 + VALUETYPE_PREFIX + vtype1 + OPERAND_DELIMITOR + 
				input2 + VALUETYPE_PREFIX + vtype2 + OPERAND_DELIMITOR + 
		        output + VALUETYPE_PREFIX + this.get_valueType() ;
		return inst;

	}
}
