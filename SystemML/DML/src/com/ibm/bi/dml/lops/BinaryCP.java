package com.ibm.bi.dml.lops;


 
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;

 
 


/**
 * Lop to perform binary scalar operations. Both inputs must be scalars.
 * Example i = j + k, i = i + 1. 
 * @author aghoting
 */

public class BinaryCP extends Lops  
{
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, DIVIDE,
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, 
		LOG,POW,MAX,MIN,PRINT,
		IQSIZE,
		Over,
		MATMULT
	}
	
	OperationTypes operation;

	
	/**
	 * Constructor to perform a scalar operation
	 * @param input
	 * @param op
	 */

	public BinaryCP(Lops input1, Lops input2, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.BinaryCP, dt, vt);		
		operation = op;		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		/*
		 * This lop is executed in control program.
		 */
		boolean breaksAlignment = false; // this field does not carry any meaning for this lop
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {

		return "Operation: " + operation;

	}
	
	public OperationTypes getOperationType(){
		return operation;
	}

	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException
	{
		String opString = new String (getExecType() + Lops.OPERAND_DELIMITOR);
		
		ValueType vtype_input1 = this.getInputs().get(0).get_valueType();
		
		switch ( operation ) {
		
		/* Arithmetic */
		case ADD:
			opString += "+"; break;
		case SUBTRACT:
			opString += "-"; break;
		case MULTIPLY:
			opString += "*"; break;
		case DIVIDE:
			opString += "/"; break;
		case POW:	
			opString += "^"; break;
			
		/* Relational */
		case LESS_THAN:
			opString += "<"; break;
		case LESS_THAN_OR_EQUALS:
			opString += "<="; break;
		case GREATER_THAN:
			opString += ">"; break;
		case GREATER_THAN_OR_EQUALS:
			opString += ">="; break;
		case EQUALS:
			opString += "=="; break;
		case NOT_EQUALS:
			opString += "!="; break;
		
		/* Boolean */
		case AND:
			opString += "&&"; break;
		case OR:
			opString += "||"; break;
		
		/* Builtin Functions */
		case LOG:
			opString += "log"; break;
		case MIN:
			opString += "min"; break;
		case MAX:
			opString += "max"; break;
		
		case PRINT:
			opString += "print"; break;
			
		case IQSIZE:
			opString += "iqsize"; 
			vtype_input1 = ValueType.STRING; // first input is a filename
			break;
		
		case MATMULT:
			opString += "ba+*";
			
/*			String mminst = opString + OPERAND_DELIMITOR + 
			input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType()  + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType()  + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
	        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;

			return mminst;
*/
			break;
			
		default:
			throw new UnsupportedOperationException("Instruction is not defined for BinaryScalar operator: " + operation);
		}
		
		String inst = opString + OPERAND_DELIMITOR + 
				input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + vtype_input1 + OPERAND_DELIMITOR + 
				input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType() + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
		        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + this.get_valueType() ;

		return inst;
		
	}
	
	@Override
	public com.ibm.bi.dml.lops.Lops.SimpleInstType getSimpleInstructionType()
	{
		switch (operation){
 
		default:
			return SimpleInstType.Scalar;
		}
	}


}



