/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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

public class BinaryScalar extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public enum OperationTypes {
		ADD, SUBTRACT, SUBTRACTRIGHT, MULTIPLY, DIVIDE, MODULUS, INTDIV,
		LESS_THAN, LESS_THAN_OR_EQUALS, GREATER_THAN, GREATER_THAN_OR_EQUALS, EQUALS, NOT_EQUALS,
		AND, OR, 
		LOG,POW,MAX,MIN,PRINT,
		IQSIZE,
		Over,
		SEQINCR
	}
	
	OperationTypes operation;

	/**
	 * This overloaded constructor is used for setting exec type in case of spark backend
	 */
	public BinaryScalar(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.BinaryCP, dt, vt);		
		operation = op;		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		boolean breaksAlignment = false; // this field does not carry any meaning for this lop
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}
	
	/**
	 * Constructor to perform a scalar operation
	 * @param input
	 * @param op
	 */

	public BinaryScalar(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.BinaryCP, dt, vt);		
		operation = op;		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

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
		String opString = getOpcode( operation );
		
		
		
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( opString );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepScalarInputOperand(getExecType()) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOutputOperand(output));

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
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static String getOpcode( OperationTypes op )
	{
		switch ( op ) 
		{
			/* Arithmetic */
			case ADD:
				return "+";
			case SUBTRACT:
				return "-";
			case MULTIPLY:
				return "*";
			case DIVIDE:
				return "/";
			case MODULUS:
				return "%%";	
			case INTDIV:
				return "%/%";	
			case POW:	
				return "^";
				
			/* Relational */
			case LESS_THAN:
				return "<";
			case LESS_THAN_OR_EQUALS:
				return "<=";
			case GREATER_THAN:
				return ">";
			case GREATER_THAN_OR_EQUALS:
				return ">=";
			case EQUALS:
				return "==";
			case NOT_EQUALS:
				return "!=";
			
			/* Boolean */
			case AND:
				return "&&";
			case OR:
				return "||";
			
			/* Builtin Functions */
			case LOG:
				return "log";
			case MIN:
				return "min"; 
			case MAX:
				return "max"; 
			
			case PRINT:
				return "print";
				
			case IQSIZE:
				return "iqsize"; 
			
			case SEQINCR:
				return "seqincr";
				
			default:
				throw new UnsupportedOperationException("Instruction is not defined for BinaryScalar operator: " + op);
		}
	}
}
