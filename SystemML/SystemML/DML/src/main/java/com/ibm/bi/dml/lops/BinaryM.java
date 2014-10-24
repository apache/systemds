/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.Binary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to perform binary operation. Both inputs must be matrices or vectors. 
 * Example - A = B + C, where B and C are matrices or vectors.
 */

public class BinaryM extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}
	
	private OperationTypes _operation;
	private CacheType _cacheType = null;
	
	
	/**
	 * Constructor to perform a binary operation.
	 * @param input
	 * @param op
	 */

	public BinaryM(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, boolean partitioned ) {
		super(Lop.Type.Binary, dt, vt);
		
		_operation = op;
		_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );	
	}

	@Override
	public String toString() 
	{
		return " Operation: " + _operation;
	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return _operation;
	}

	private String getOpcode()
	{
		return getOpcode( _operation );
	}
	
	public static String getOpcode( OperationTypes op ) {
		switch(op) {
		/* Arithmetic */
		case ADD:
			return "map+";
		case SUBTRACT:
			return "map-";
		case MULTIPLY:
			return "map*";
		case DIVIDE:
			return "map/";
		case MODULUS:
			return "map%%";	
		case INTDIV:
			return "map%/%";		
		
		/* Relational */
		case LESS_THAN:
			return "map<";
		case LESS_THAN_OR_EQUALS:
			return "map<=";
		case GREATER_THAN:
			return "map>";
		case GREATER_THAN_OR_EQUALS:
			return "map>=";
		case EQUALS:
			return "map==";
		case NOT_EQUALS:
			return "map!=";
		
			/* Boolean */
		case AND:
			return "map&&";
		case OR:
			return "map||";
		
		
		/* Builtin Functions */
		case MIN:
			return "mapmin";
		case MAX:
			return "mapmax";
		case POW:
			return "map^";
			
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Binary operation: " + op);
		}
	}

	public static boolean isOpcode(String opcode)
	{
		return opcode.equals("map+") || opcode.equals("map-") ||
			   opcode.equals("map*") || opcode.equals("map/") ||	
			   opcode.equals("map%%") || opcode.equals("map%/%") ||	
			   opcode.equals("map<") || opcode.equals("map<=") ||	
			   opcode.equals("map>") || opcode.equals("map>=") ||	
			   opcode.equals("map==") || opcode.equals("map!=") ||	
			   opcode.equals("map&&") || opcode.equals("map||") ||	
			   opcode.equals("mapmin") || opcode.equals("mapmax") ||	
			   opcode.equals("map^");
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() ); 
		sb.append( OPERAND_DELIMITOR );
		
		sb.append ( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append ( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		sb.append( OPERAND_DELIMITOR );
		sb.append(_cacheType);
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException
	{
		return getInstructions(input_index1+"", input_index2+"", output_index+"");
	}
	
	@Override
	public boolean usesDistributedCache() 
	{
		return true;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{	
		// second input is from distributed cache
		return new int[]{2};
	}
}