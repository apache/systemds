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

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


public class PickByCount extends Lop 
{
		
	public static final String OPCODE = "qpick";
	
	public enum OperationTypes {
		VALUEPICK, 
		RANGEPICK, 
		IQM, 
		MEDIAN
	};	
	
	private OperationTypes operation;
	private boolean inMemoryInput = false;

	
	/*
	 * valuepick: first input is always a matrix, second input can either be a scalar or a matrix
	 * rangepick: first input is always a matrix, second input is always a scalar
	 */
	public PickByCount(Lop input1, Lop input2, DataType dt, ValueType vt, OperationTypes op) {
		this(input1, input2, dt, vt, op, ExecType.MR);
	}
	
	public PickByCount(Lop input1, Lop input2, DataType dt, ValueType vt, OperationTypes op, ExecType et) {
		super(Lop.Type.PickValues, dt, vt);
		init(input1, input2, op, et);
	}

	public PickByCount(Lop input1, Lop input2, DataType dt, ValueType vt, OperationTypes op, ExecType et, boolean inMemoryInput) {
		super(Lop.Type.PickValues, dt, vt);
		this.inMemoryInput = inMemoryInput;
		init(input1, input2, op, et);
	}

	
	private void init(Lop input1, Lop input2, OperationTypes op, ExecType et) {
		this.addInput(input1);
		input1.addOutput(this);
		
		if ( input2 != null ) {
			this.addInput(input2);
			input2.addOutput(this);
		}
		
		operation = op;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			lps.setProperties( inputs, et, ExecLocation.RecordReader, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	@Override
	public String toString() {
		return "Operation: " + operation;
	}
	
	public OperationTypes getOperationType() {
		return operation;
	}

	/*
	 * This version of getInstruction() must be called only for valuepick (MR) and rangepick
	 * 
	 * Example instances:
	 * valupick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE
	 * rangepick:::0:DOUBLE:::0.25:DOUBLE:::1:DOUBLE
	 * rangepick:::0:DOUBLE:::Var3:DOUBLE:::1:DOUBLE
	 */
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) 
		throws LopsException
	{
		return getInstructions(""+input_index1, ""+input_index2, ""+output_index);

	}

	/*
	 * This version of getInstructions() must be called only for valuepick (CP), IQM (CP)
	 * 
	 * Example instances:
	 * valuepick:::temp2:STRING:::0.25:DOUBLE:::Var1:DOUBLE
	 * valuepick:::temp2:STRING:::Var1:DOUBLE:::Var2:DOUBLE
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) throws LopsException
	{		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		if(operation != OperationTypes.MEDIAN) {
			if ( getInputs().get(1).getDataType() == DataType.SCALAR ) 
				sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
			else {
				sb.append( getInputs().get(1).prepInputOperand(input2));
			}
			sb.append( OPERAND_DELIMITOR );
		}
		
		sb.append( this.prepOutputOperand(output));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(operation);
		
		sb.append( OPERAND_DELIMITOR );		
		sb.append(inMemoryInput);
		
		return sb.toString();
	}
	
	/**
	 * This version of getInstructions() is called for IQM, executing in CP
	 * 
	 * Example instances:
	 *   iqm:::input:::output
	 */
	@Override
	public String getInstructions(String input, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output) );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append(operation);
		
		sb.append( OPERAND_DELIMITOR );		
		sb.append(inMemoryInput);
		
		return sb.toString();
	}
}
