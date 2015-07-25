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
 * TODO Currently this lop only support the right hand side in distributed cache. This
 *  should be generalized (incl hop operator selection) to left/right cache types.
 *  
 * 
 */
public class UAggOuterChain extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "uaggouterchain";

	//outer operation
	private Aggregate.OperationTypes _uaggOp         = null;
	private PartialAggregate.DirectionTypes _uaggDir = null;
	//inner operation
	private Binary.OperationTypes _binOp             = null;	
		
	
	/**
	 * Constructor to setup a unaryagg outer chain
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */	
	public UAggOuterChain(Lop input1, Lop input2, Aggregate.OperationTypes uaop, PartialAggregate.DirectionTypes uadir, Binary.OperationTypes bop, DataType dt, ValueType vt, ExecType et) 
		throws LopsException 
	{
		super(Lop.Type.UaggOuterChain, dt, vt);		
		addInput(input1);
		addInput(input2);
		input1.addOutput(this); 
		input2.addOutput(this); 
		
		//setup operator types
		_uaggOp = uaop;
		_uaggDir = uadir;
		_binOp = bop;
		
		//setup MR parameters 
		if( et == ExecType.MR )
		{
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			lps.addCompatibility(JobType.CSV_REBLOCK);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else //SPARK
		{
			boolean breaksAlignment = false;
			boolean aligner = false;
			boolean definesMRJob = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	

	public String toString() {
		return "Operation = UaggOuterChain";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		
		//exec type
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		//inst op code
		sb.append(OPCODE);
		sb.append(Lop.OPERAND_DELIMITOR);

		//outer operation op code
		sb.append(PartialAggregate.getOpcode(_uaggOp, _uaggDir));		
		sb.append(Lop.OPERAND_DELIMITOR);

		//inner operation op code
		sb.append(Binary.getOpcode(_binOp));
		sb.append(Lop.OPERAND_DELIMITOR);
				
		//inputs and outputs
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output_index));
				
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String output)
	{
		throw new RuntimeException("not implemented yet.");
	}
	
	
	@Override
	public boolean usesDistributedCache() 
	{
		return true;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{
		return new int[]{2};
	}
}
