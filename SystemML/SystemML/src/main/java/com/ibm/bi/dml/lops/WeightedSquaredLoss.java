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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * 
 */
public class WeightedSquaredLoss extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final String OPCODE = "mapwsloss";
	public static final String OPCODE_CP = "wsloss";
	private int _numThreads = 1;

	public enum WeightsType {
		POST, 
		PRE, 
		NONE,
	}
	
	private WeightsType _weightsType = null;
	
	public WeightedSquaredLoss(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, WeightsType wt, ExecType et) 
		throws LopsException 
	{
		super(Lop.Type.WeightedSquaredLoss, dt, vt);		
		addInput(input1); //X
		addInput(input2); //U
		addInput(input3); //V
		addInput(input4); //W
		input1.addOutput(this); 
		input2.addOutput(this);
		input3.addOutput(this);
		input4.addOutput(this);
		
		//setup mapmult parameters
		_weightsType = wt;
		setupLopProperties(et);
	}
	
	/**
	 * 
	 * @param et
	 */
	private void setupLopProperties( ExecType et )
	{
		if( et == ExecType.MR )
		{
			//setup MR parameters 
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //Spark/CP
		{
			//setup Spark parameters 
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	public String toString() {
		return "Operation = WeightedSquaredLoss";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index)
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
		sb.append( getInputs().get(2).prepInputOperand(input_index3));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(3).prepInputOperand(input_index4));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output_index));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_weightsType);
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output)
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(OPCODE_CP);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input2));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(2).prepInputOperand(input3));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(3).prepInputOperand(input4));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_weightsType);
		
		//append degree of parallelism
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		
		return sb.toString();
	}
	
	@Override
	public boolean usesDistributedCache() 
	{
		if( getExecType()==ExecType.MR )
			return true;
		else
			return false;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{
		if( getExecType()==ExecType.MR )
			return new int[]{2,3};
		else
			return new int[]{-1};
	}
	
	public void setNumThreads(int k) {
		_numThreads = k;
	}
}
