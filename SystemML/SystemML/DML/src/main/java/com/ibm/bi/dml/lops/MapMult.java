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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class MapMult extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "mapmult";
	
	/**
	 * Constructor to setup a partial Matrix-Vector Multiplication
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */
	
	public MapMult(Lop input1, Lop input2, DataType dt, ValueType vt) throws LopsException {
		super(Lop.Type.MapMult, dt, vt);		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		/*
		 * This lop can be executed only in MMCJ job.
		 */
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}


	public String toString() {
		return "Operation = MVMult";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(OPCODE);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	public boolean usesDistributedCache() {
		return true;
	}
	
	public int distributedCacheInputIndex() {
		return 2;  // second input is from distributed cache
	}


}
