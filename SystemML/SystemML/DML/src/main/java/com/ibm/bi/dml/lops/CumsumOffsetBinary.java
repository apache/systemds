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
 * 
 * 
 */
public class CumsumOffsetBinary extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public CumsumOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt)
		throws LopsException 
	{
		super(Lop.Type.CumsumOffsetBinary, dt, vt);
		init(data, offsets, dt, vt, ExecType.MR);
	}
	
	/**
	 * 
	 * @param input
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		this.lps.setProperties(inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob);
	}

	public String toString() {
		return "Cumsum Offset Binary";
	}
	
	private String getOpcode() {
		return "bcumoffk+";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index) );

		return sb.toString();
	}
}
