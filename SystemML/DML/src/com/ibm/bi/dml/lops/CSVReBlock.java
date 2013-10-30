/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to convert CSV data into SystemML data format (TextCell, BinaryCell, or BinaryBlock)
 */
public class CSVReBlock extends Lop 
{
	public static final String OPCODE = "csvrblk"; 
	
	Long rows_per_block;
	Long cols_per_block;

	public CSVReBlock(Lop input, Long rows_per_block, Long cols_per_block, DataType dt, ValueType vt) throws LopsException 
	{
		super(Lop.Type.CSVReBlock, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		this.rows_per_block = rows_per_block;
		this.cols_per_block = cols_per_block;
		
		/*
		 * This lop can be executed only in CSVREBLOCK job.
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		
		lps.addCompatibility(JobType.CSV_REBLOCK);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "CSVReblock - rows per block = " + rows_per_block + " cols per block  " + cols_per_block ;
	}

	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		
		Lop input = getInputs().get(0);
		
		sb.append( input_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( input.get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( input.get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( rows_per_block );
		sb.append( OPERAND_DELIMITOR );
		sb.append( cols_per_block );
		
		Lop headerLop = ((Data)input).getNamedInputLop(Statement.DELIM_HAS_HEADER_ROW);
		Lop delimLop = ((Data)input).getNamedInputLop(Statement.DELIM_DELIMITER);
		Lop fillLop = ((Data)input).getNamedInputLop(Statement.DELIM_FILL); 
		Lop fillValueLop = ((Data)input).getNamedInputLop(Statement.DELIM_FILL_VALUE);
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)headerLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)delimLop).getStringValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillValueLop).getDoubleValue() );
		
		return sb.toString();
	}
 
}
