/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * 
 */
public class WeightedSigmoidR extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static final String OPCODE = "redwsigmoid";
	
	private WSigmoidType _wsType = null;

	private boolean _cacheU = false;
	private boolean _cacheV = false;
	
	public WeightedSigmoidR(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, WSigmoidType wt, boolean cacheU, boolean cacheV, ExecType et) 
		throws LopsException 
	{
		super(Lop.Type.WeightedSigmoid, dt, vt);		
		addInput(input1); //X
		addInput(input2); //U
		addInput(input3); //V
		input1.addOutput(this); 
		input2.addOutput(this);
		input3.addOutput(this);
		
		//setup mapmult parameters
		_wsType = wt;
		_cacheU = cacheU;
		_cacheV = cacheV;
		setupLopProperties(et);
	}
	
	/**
	 * 
	 * @param et
	 * @throws LopsException 
	 */
	private void setupLopProperties( ExecType et ) 
		throws LopsException
	{
		if( et != ExecType.MR )
			throw new LopsException("Execution type other than MR (currently not supported for this lop): "+et);
		
		//setup MR parameters 
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		lps.setProperties( inputs, ExecType.MR, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
	}

	public String toString() {
		return "Operation = WeightedSigmoidR";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index)
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
		sb.append( this.prepOutputOperand(output_index));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_wsType);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_cacheU);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_cacheV);
		
		return sb.toString();
	}

	
	@Override
	public boolean usesDistributedCache() 
	{
		if( _cacheU || _cacheV )
			return true;
		else
			return false;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{
		if( !_cacheU && !_cacheV )
			return new int[]{-1};
		else if( _cacheU && !_cacheV )
			return new int[]{2};
		else if( !_cacheU && _cacheV )
			return new int[]{3};
		else
			return new int[]{2,3};
	}
}
