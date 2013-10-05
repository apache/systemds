/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;


 
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;

 
 


/**
 * Lop to perform binary scalar operations. Both inputs must be scalars.
 * Example i = j + k, i = i + 1. 
 */

public class BinaryCP extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, DIVIDE, MODULUS,
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, 
		LOG,POW,MAX,MIN,PRINT,
		IQSIZE,
		Over,
		MATMULT, SEQINCR
	}
	
	OperationTypes operation;

	
	/**
	 * Constructor to perform a scalar operation
	 * @param input
	 * @param op
	 */

	public BinaryCP(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.BinaryCP, dt, vt);		
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
		this.lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
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
		String opString = new String (getExecType() + Lop.OPERAND_DELIMITOR);
		
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
		case MODULUS:
			opString += "%%"; break;	
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
			break;
			
		case SEQINCR:
			opString += "seqincr";
			break;
			
/*			String mminst = opString + OPERAND_DELIMITOR + 
			input1 + DATATYPE_PREFIX + getInputs().get(0).get_dataType()  + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
			input2 + DATATYPE_PREFIX + getInputs().get(1).get_dataType()  + VALUETYPE_PREFIX + getInputs().get(1).get_valueType() + OPERAND_DELIMITOR + 
	        output + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() ;

			return mminst;
*/
			
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for BinaryScalar operator: " + operation);
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append( opString );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( vtype_input1 );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).get_valueType());
		sb.append( OPERAND_DELIMITOR );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() ); 
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );

		return sb.toString();
		
	}
	
	@Override
	public Lop.SimpleInstType getSimpleInstructionType()
	{
		switch (operation){
 
		default:
			return SimpleInstType.Scalar;
		}
	}


}



