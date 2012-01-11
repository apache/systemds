package dml.lops;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.utils.LopsException;

/**
 * Lop to perform binary operation. Both inputs must be matrices or vectors. 
 * Example - A = B + C, where B and C are matrices or vectors.
 * @author aghoting
 */

public class Binary extends Lops 
{
	public enum OperationTypes {
		ADD, SUBTRACT, MULTIPLY, DIVIDE,
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, 
		MAX, MIN, NOTSUPPORTED};	
	OperationTypes operation;
	

	
	/**
	 * Constructor to perform a binary operation.
	 * @param input
	 * @param op
	 */

	public Binary(Lops input1, Lops input2, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.Binary, dt, vt);	
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.RAND);
		lps.addCompatibility(JobType.REBLOCK_BINARY);
		lps.addCompatibility(JobType.REBLOCK_TEXT);
		
		boolean breaksAlignment = false;
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
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException
	{
		String opString = new String("");
		switch(operation) {
		/* Arithmetic */
		case ADD:
			opString += "b+"; break;
		case SUBTRACT:
			opString += "b-"; break;
		case MULTIPLY:
			opString += "b*"; break;
		case DIVIDE:
			opString += "b/"; break;
		
		/* Relational */
		case LESS_THAN:
			opString += "b<"; break;
		case LESS_THAN_OR_EQUALS:
			opString += "b<="; break;
		case GREATER_THAN:
			opString += "b>"; break;
		case GREATER_THAN_OR_EQUALS:
			opString += "b>="; break;
		case EQUALS:
			opString += "b=="; break;
		case NOT_EQUALS:
			opString += "b!="; break;
		
			/* Boolean */
		case AND:
			opString += "b&&"; break;
		case OR:
			opString += "b||"; break;
		
		
		/* Builtin Functions */
		case MIN:
			opString += "bmin"; break;
		case MAX:
			opString += "bmax"; break;
			
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Binary operation: " + operation);
		}
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input_index1 + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
				input_index2 + VALUETYPE_PREFIX + this.getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
		
		return inst;
	}

 
}