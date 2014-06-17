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


public class AppendR extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "rappend";
	
	public AppendR(Lop input1, Lop input2, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, dt, vt);
	}
	
	public void init(Lop input1, Lop input2, DataType dt, ValueType vt) 
	{
		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN); //currently required for correctness		
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
	}
	
	@Override
	public String toString() {

		return " AppendR: ";
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( this.lps.execType );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "rappend" );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index+"") );
		
		return sb.toString();	
	}
}