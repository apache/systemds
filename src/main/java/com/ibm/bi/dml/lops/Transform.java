/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/*
 * Lop to perform transpose/vector to diag operations
 * This lop can change the keys and hence break alignment.
 */

public class Transform extends Lop
{

	
	public enum OperationTypes {
		Transpose,
		Diag,
		Reshape,
		Sort,
	};
	
	private OperationTypes operation = null;
	
	/**
	 * Constructor when we have one input.
	 * @param input
	 * @param op
	 */

	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
	}
	
	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, ExecType.MR);
	}

	private void init (Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		operation = op;
 
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
			/*
			 *  This lop CAN NOT be executed in PARTITION, SORT, STANDALONE
			 *  MMCJ: only in mapper.
			 */
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			lps.addCompatibility(JobType.CSV_REBLOCK);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			
			if( op == OperationTypes.Reshape )
				//reshape should be executed in map because we have potentially large intermediate data and want to exploit the combiner.
				this.lps.setProperties( inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
			else
				this.lps.setProperties( inputs, et, ExecLocation.MapOrReduce, breaksAlignment, aligner, definesMRJob );
		}
		else //CP/SPARK
		{
			// <code>breaksAlignment</code> is not meaningful when <code>Transform</code> executes in CP. 
			breaksAlignment = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	@Override
	public String toString() {

		return " Operation: " + operation;
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
		case Transpose:
			// Transpose a matrix
			return "r'";
		
		case Diag:
			// Transform a vector into a diagonal matrix
			return "rdiag";
		
		case Reshape:
			// Transform a vector into a diagonal matrix
			return "rshape";
		
		case Sort:
			// Transform a matrix into a sorted matrix 
			return "rsort";
		
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	//CP instructions
	
	@Override
	public String getInstructions(String input1, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) 
		throws LopsException 
	{
		//only used for reshape
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		//rows, cols, byrow
		String[] inputX = new String[]{input2,input3,input4};
		for( int i=1; i<=(inputX.length); i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( ltmp.prepScalarInputOperand(getExecType()));
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	//MR instructions

	@Override 
	public String getInstructions(int input_index, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}
	
	@Override 
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) 
		throws LopsException
	{
		//only used for reshape
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		
		//rows		
		Lop input2 = getInputs().get(1); 
		String rowsString = input2.prepScalarLabel(); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowsString );
		
		//cols
		Lop input3 = getInputs().get(2); 
		String colsString = input3.prepScalarLabel(); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( colsString );
		
		//byrow
		Lop input4 = getInputs().get(3); 
		String byrowString = input4.prepScalarLabel();
		if ( input4.getExecLocation() == ExecLocation.Data 
				&& !((Data)input4).isLiteral() || !(input4.getExecLocation() == ExecLocation.Data ) ){
			throw new LopsException(this.printErrorLocation() + "Parameter 'byRow' must be a literal for a matrix operation.");
		}
		sb.append( OPERAND_DELIMITOR );
		sb.append( byrowString );
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	public static Transform constructTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.Transform ) {
				return (Transform)lop;
			}
		}
		Transform retVal = new Transform(input1, op, dt, vt);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal;
	}

	public static Transform constructTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.Transform ) {
				return (Transform)lop;
			}
		}
		Transform retVal = new  Transform(input1, op, dt, vt, et);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal; 
	}

 
}