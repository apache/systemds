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
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to represent an aggregation.
 * It is used in rowsum, colsum, etc. 
 */

public class Aggregate extends Lop 
{

	
	/** Aggregate operation types **/
	
	public enum OperationTypes {
		Sum, Product, Min, Max, Trace, KahanSum, KahanSumSq, KahanTrace, Mean,MaxIndex, MinIndex
	}
	OperationTypes operation;
 
	private boolean isCorrectionUsed = false;
	private CorrectionLocationType correctionLocation = CorrectionLocationType.INVALID;

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	public Aggregate(Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt ) {
		super(Lop.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, ExecType.MR );
	}
	
	public Aggregate(Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		super(Lop.Type.Aggregate, dt, vt);
		init ( input, op, dt, vt, et );
	}
	
	private void init (Lop input, Aggregate.OperationTypes op, DataType dt, ValueType vt, ExecType et ) {
		operation = op;	
		this.addInput(input);
		input.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			this.lps.setProperties( inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	// this function must be invoked during hop-to-lop translation
	public void setupCorrectionLocation(CorrectionLocationType loc) {
		if (operation == OperationTypes.KahanSum || operation == OperationTypes.KahanSumSq
				|| operation == OperationTypes.KahanTrace || operation == OperationTypes.Mean) {
			isCorrectionUsed = true;
			correctionLocation = loc;
		}
	}
	
	/**
	 * for debugging purposes. 
	 */
	
	public String toString()
	{
		return "Operation: " + operation;		
	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}
	
	
	private String getOpcode() {
		switch(operation) {
		case Sum: 
		case Trace: 
			return "a+"; 
		case Mean: 
			return "amean"; 
		case Product: 
			return "a*"; 
		case Min: 
			return "amin"; 
		case Max: 
			return "amax"; 
		case MaxIndex:
			return "arimax";
		case MinIndex:
			return "arimin";
		case KahanSum:
		case KahanTrace: 
			return "ak+";
		case KahanSumSq:
			return "asqk+";
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Aggregate operation: " + operation);
		}
	}
	
	@Override
	public String getInstructions(String input1, String output) throws LopsException {
		String opcode = getOpcode(); 
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( opcode );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		boolean isCorrectionApplicable = false;
		
		String opcode = getOpcode(); 
		if (operation == OperationTypes.Mean || operation == OperationTypes.KahanSum
				|| operation == OperationTypes.KahanSumSq || operation == OperationTypes.KahanTrace)
			isCorrectionApplicable = true;
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( opcode );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index));
		sb.append( OPERAND_DELIMITOR );

		sb.append( this.prepOutputOperand(output_index));
		
		if ( isCorrectionApplicable )
		{
			// add correction information to the instruction
			sb.append( OPERAND_DELIMITOR );
			sb.append( isCorrectionUsed );
			sb.append( OPERAND_DELIMITOR );
			sb.append( correctionLocation );
		}
		
		return sb.toString();
	}

 
 
}
