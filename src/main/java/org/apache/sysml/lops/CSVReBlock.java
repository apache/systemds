/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


/**
 * Lop to convert CSV data into SystemML data format (TextCell, BinaryCell, or BinaryBlock)
 */
public class CSVReBlock extends Lop 
{
	
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
	
	private String prepCSVProperties() throws LopsException {
		StringBuilder sb = new StringBuilder();

		Data dataInput = (Data)getInputs().get(0);
		
		Lop headerLop = dataInput.getNamedInputLop(DataExpression.DELIM_HAS_HEADER_ROW, 
			String.valueOf(DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW));
		Lop delimLop = dataInput.getNamedInputLop(DataExpression.DELIM_DELIMITER, 
			DataExpression.DEFAULT_DELIM_DELIMITER);
		Lop fillLop = dataInput.getNamedInputLop(DataExpression.DELIM_FILL, 
			String.valueOf(DataExpression.DEFAULT_DELIM_FILL)); 
		Lop fillValueLop = dataInput.getNamedInputLop(DataExpression.DELIM_FILL_VALUE, 
			String.valueOf(DataExpression.DEFAULT_DELIM_FILL_VALUE));
		
		if (headerLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_HAS_HEADER_ROW
					+ " must be a literal.");
		if (delimLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_DELIMITER
					+ " must be a literal.");
		if (fillLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL
					+ " must be a literal.");
		if (fillValueLop.isVariable())
			throw new LopsException(this.printErrorLocation()
					+ "Parameter " + DataExpression.DELIM_FILL_VALUE
					+ " must be a literal.");

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
	public String getInstructions(int input_index, int output_index) throws LopsException {
		return getInstructions(String.valueOf(input_index), String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		sb.append( OPERAND_DELIMITOR );
		sb.append( rows_per_block );
		sb.append( OPERAND_DELIMITOR );
		sb.append( cols_per_block );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepCSVProperties() );

		return sb.toString();
	}
}
