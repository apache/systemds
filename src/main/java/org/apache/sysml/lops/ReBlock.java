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
import org.apache.sysml.lops.OutputParameters.Format;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


/**
 * Lop to perform reblock operation
 */
public class ReBlock extends Lop 
{
	
	public static final String OPCODE = "rblk"; 
	
	private boolean _outputEmptyBlocks = true;
	
	/**
	 * Constructor to perform a reblock operation. 
	 * @param input
	 * @param op
	 */
	
	private Long _rows_per_block;
	private Long _cols_per_block;

	public ReBlock(Lop input, Long rows_per_block, Long cols_per_block, DataType dt, ValueType vt, boolean outputEmptyBlocks, ExecType et) throws LopsException
	{
		super(Lop.Type.ReBlock, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		_rows_per_block = rows_per_block;
		_cols_per_block = cols_per_block;
		
		_outputEmptyBlocks = outputEmptyBlocks;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		
		lps.addCompatibility(JobType.REBLOCK);
		
		if(et == ExecType.MR) 
			lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
		else if(et == ExecType.SPARK) 
			lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		else 
			throw new LopsException("Incorrect execution type for Reblock:" + et);
	}

	@Override
	public String toString() {
	
		return "Reblock - rows per block = " + _rows_per_block + " cols per block  " + _cols_per_block ;
	}

	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append ( this.prepOutputOperand(output_index));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( _rows_per_block );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( _cols_per_block );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(_outputEmptyBlocks);
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		if(getExecType() != ExecType.SPARK) {
			throw new LopsException("The method getInstructions(String,String) for Reblock should be called only for Spark execution type");
		}
		
		if (this.getInputs().size() == 1) {
			
			StringBuilder sb = new StringBuilder();
			sb.append( getExecType() );
			sb.append( Lop.OPERAND_DELIMITOR );
			sb.append( "rblk" );
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(0).prepInputOperand(input1));
			sb.append( OPERAND_DELIMITOR );
			sb.append( this.prepOutputOperand(output));
			sb.append( OPERAND_DELIMITOR );
			sb.append( _rows_per_block );
			sb.append( OPERAND_DELIMITOR );
			sb.append( _cols_per_block );
			sb.append( OPERAND_DELIMITOR );
			sb.append(_outputEmptyBlocks);
			
			return sb.toString();

		} else {
			throw new LopsException(this.printErrorLocation() + "Invalid number of operands ("
					+ this.getInputs().size() + ") for Reblock operation");
		}
	}
	
	// This function is replicated in Dag.java
	@SuppressWarnings("unused")
	private Format getChildFormat(Lop node) throws LopsException {
		
		if(node.getOutputParameters().getFile_name() != null
				|| node.getOutputParameters().getLabel() != null)
		{
			return node.getOutputParameters().getFormat();
		}
		else
		{
			// Reblock lop should always have a single child
			if(node.getInputs().size() > 1)
				throw new LopsException(this.printErrorLocation() + "Should only have one child! \n");
			
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