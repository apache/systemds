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
 */
public class CumulativeSplitAggregate extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private double _initValue = 0;
	
	public CumulativeSplitAggregate(Lop input, DataType dt, ValueType vt, double init)
		throws LopsException 
	{
		super(Lop.Type.CumulativeSplitAggregate, dt, vt);
		_initValue = init;
		init(input, dt, vt, ExecType.MR);
	}
	
	/**
	 * 
	 * @param input
	 * @param dt
	 * @param vt
	 * @param et
	 */
	private void init(Lop input, DataType dt, ValueType vt, ExecType et) {
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
	}

	public String toString() {
		return "CumulativeSplitAggregate";
	}
	
	private String getOpcode() {
		return "ucumsplit";
	}
	
	@Override
	public String getInstructions(int input_index, int output_index)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( String.valueOf(_initValue) );

		return sb.toString();
	}
}
