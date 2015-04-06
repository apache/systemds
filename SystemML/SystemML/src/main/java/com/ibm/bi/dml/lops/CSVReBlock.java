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
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Lop to convert CSV data into SystemML data format (TextCell, BinaryCell, or BinaryBlock)
 */
public class CSVReBlock extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "csvrblk"; 
	
	Long rows_per_block;
	Long cols_per_block;

	public CSVReBlock(Lop input, Long rows_per_block, Long cols_per_block, DataType dt, ValueType vt, ExecType et) throws LopsException 
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
		
		if(et == ExecType.MR) {
			this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
		}
		else if(et == ExecType.SPARK) {
			this.lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
		else {
			throw new LopsException("Incorrect execution type for CSVReblock:" + et);
		}
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
		
		sb.append( input.prepInputOperand(input_index) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output_index) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( rows_per_block );
		sb.append( OPERAND_DELIMITOR );
		sb.append( cols_per_block );
		sb.append( OPERAND_DELIMITOR );
		
		Lop headerLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
		Lop delimLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_DELIMITER);
		Lop fillLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_FILL); 
		Lop fillValueLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_FILL_VALUE);
		
		if (headerLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
					+ " must be a literal for a seq operation.");
		if (delimLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_DELIMITER
					+ " must be a literal for a seq operation.");
		if (fillLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL
					+ " must be a literal for a seq operation.");
		if (fillValueLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL_VALUE
					+ " must be a literal for a seq operation.");

		sb.append( ((Data)headerLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)delimLop).getStringValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillLop).getBooleanValue() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( ((Data)fillValueLop).getDoubleValue() );
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		if(getExecType() != ExecType.SPARK) {
			throw new LopsException("The method getInstructions(String,String) for CSVReblock should be called only for Spark execution type");
		}
		
		if (this.getInputs().size() == 1) {
			
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lop.OPERAND_DELIMITOR );
			sb.append( "csvrblk" );
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(0).prepInputOperand(input1));
			sb.append( OPERAND_DELIMITOR );
			sb.append( this.prepOutputOperand(output));
			sb.append( OPERAND_DELIMITOR );
			sb.append( rows_per_block );
			sb.append( OPERAND_DELIMITOR );
			sb.append( cols_per_block );
			sb.append( OPERAND_DELIMITOR );
			
			Lop input = getInputs().get(0);
			
			Lop headerLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW);
			Lop delimLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_DELIMITER);
			Lop fillLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_FILL); 
			Lop fillValueLop = ((Data)input).getNamedInputLop(DataExpression.DELIM_FILL_VALUE);
			
			if (headerLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
						+ " must be a literal for a seq operation.");
			if (delimLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_DELIMITER
						+ " must be a literal for a seq operation.");
			if (fillLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_FILL
						+ " must be a literal for a seq operation.");
			if (fillValueLop.isVariable())
				throw new LopsException(this.printErrorLocation()
						+ "Parameter " + DataExpression.DELIM_FILL_VALUE
						+ " must be a literal for a seq operation.");

			sb.append( ((Data)headerLop).getBooleanValue() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( ((Data)delimLop).getStringValue() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( ((Data)fillLop).getBooleanValue() );
			sb.append( OPERAND_DELIMITOR );
			sb.append( ((Data)fillValueLop).getDoubleValue() );
			
			
			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for CSVReblock operation");
		}
	}
 
}
