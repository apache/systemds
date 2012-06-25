package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Lop to perform reblock operation
 */
public class ReBlock extends Lops 
{
	
	/**
	 * Constructor to perform a reblock operation. 
	 * @param input
	 * @param op
	 */
	
	Long rows_per_block;
	Long cols_per_block;

	public ReBlock(Lops input, Long rows_per_block, Long cols_per_block, DataType dt, ValueType vt) throws LopsException 
	{
		super(Lops.Type.ReBlock, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		this.rows_per_block = rows_per_block;
		this.cols_per_block = cols_per_block;
		
		/*
		 * This lop can be executed only in REBLOCK job.
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		
		if ( getChildFormat(this) == Format.BINARY ) 
			lps.addCompatibility(JobType.REBLOCK_BINARY);
		else
			lps.addCompatibility(JobType.REBLOCK_TEXT);
		this.lps.setProperties( ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "Reblock - rows per block = " + rows_per_block + " cols per block  " + cols_per_block ;
	}

	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		String opString = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		opString += "rblk";
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input_index + DATATYPE_PREFIX + getInputs().get(0).get_dataType() + VALUETYPE_PREFIX + getInputs().get(0).get_valueType() + OPERAND_DELIMITOR +
		        output_index + DATATYPE_PREFIX + get_dataType() + VALUETYPE_PREFIX + get_valueType() + OPERAND_DELIMITOR +
				rows_per_block + OPERAND_DELIMITOR + cols_per_block;
		
		return inst;
	}
	
	// This function is replicated in Dag.java
	private Format getChildFormat(Lops node) throws LopsException {
		
		if(node.getOutputParameters().getFile_name() != null
				|| node.getOutputParameters().getLabel() != null)
		{
			return node.getOutputParameters().getFormat();
		}
		else
		{
			// Reblock lop should always have a single child
			if(node.getInputs().size() > 1)
				throw new LopsException("Should only have one child!");
			
			/*
			 * Return the format of the child node (i.e., input lop)
			 * No need of recursion here.. because
			 * 1) Reblock lop's input can either be DataLop or some intermediate computation
			 *    If it is Data then we just take its format (TEXT or BINARY)
			 *    If it is intermediate lop then it is always BINARY 
			 *      since we assume that all intermediate computations will be in Binary format
			 * 2) Note that Reblock job will never have any instructions in the mapper 
			 *    => the input lop (if it is other than Data) is always executed in a different job
			 */
			// return getChildFormat(node.getInputs().get(0));
			return node.getInputs().get(0).getOutputParameters().getFormat();		}
		
	}

 
 
}