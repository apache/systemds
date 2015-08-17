/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * 
 * 
 */
public class TernaryAggregate extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String OPCODE = "tak+*";
	
	//NOTE: currently only used for ta+*
	//private Aggregate.OperationTypes _aggOp = null;
	//private Binary.OperationTypes _binOp = null;
	
	/**
	 * @param et 
	 * @param input - input lop
	 * @param op - operation type
	 */
	public TernaryAggregate(Lop input1, Lop input2, Lop input3, Aggregate.OperationTypes aggOp, Binary.OperationTypes binOp, DataType dt, ValueType vt, ExecType et ) 
	{
		super(Lop.Type.TernaryAggregate, dt, vt);
		
		//_aggOp = aggOp;	
		//_binOp = binOp;
		
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}
	
	@Override
	public String toString()
	{
		return "Operation: "+OPCODE;		
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepInputOperand(input3));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		
		return sb.toString();
	}
}
