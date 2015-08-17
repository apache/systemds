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

import java.util.HashSet;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

public class SortKeys extends Lop 
{
	
	public static final String OPCODE = "qsort"; //quantile sort 
	
	public enum OperationTypes { 
		WithWeights, 
		WithoutWeights,
		Indexes,
	};
	
	private OperationTypes operation;
	private boolean descending = false;
	
	public OperationTypes getOpType() {
		return operation;
	}
	
	public SortKeys(Lop input, OperationTypes op, DataType dt, ValueType vt) {
		super(Lop.Type.SortKeys, dt, vt);		
		init(input, null, op, ExecType.MR);
	}

	public SortKeys(Lop input, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.SortKeys, dt, vt);		
		init(input, null, op, et);
	}
	
	public SortKeys(Lop input, boolean desc, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.SortKeys, dt, vt);		
		init(input, null, op, et);
		descending = desc;
	}

	public SortKeys(Lop input1, Lop input2, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.SortKeys, dt, vt);		
		init(input1, input2, op, et);
	}
	
	private void init(Lop input1, Lop input2, OperationTypes op, ExecType et) {
		this.addInput(input1);
		input1.addOutput(this);
		
		operation = op;
		
		if ( et == ExecType.MR ) {
			boolean breaksAlignment = true;
			boolean aligner = false;
			boolean definesMRJob = true;
			
			lps.addCompatibility(JobType.SORT);
			this.lps.setProperties( inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
			if(op != OperationTypes.Indexes)
				this.lps.setProducesIntermediateOutput(true);
		}
		else {
			// SortKeys can accept a optional second input only when executing in CP
			// Example: sorting with weights inside CP
			if ( input2 != null ) {
				this.addInput(input2);
				input2.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, false, false, false);
		}
	}


	@Override
	public String toString() {
		return "Operation: SortKeys (" + operation + ")";
	}

	@Override
	public String getInstructions(int input_index, int output_index)
	{
		return getInstructions(String.valueOf(input_index), String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input, String output)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		sb.append( OPERAND_DELIMITOR );
		sb.append ( this.prepOutputOperand(output));
		
		if( getExecType() == ExecType.MR ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( operation );
			sb.append( OPERAND_DELIMITOR );
			sb.append( descending );
		}
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	// This method is invoked in two cases:
	// 1) SortKeys (both weighted and unweighted) executes in MR
	// 2) Unweighted SortKeys executes in CP
	public static SortKeys constructSortByValueLop(Lop input1, OperationTypes op, 
			DataType dt, ValueType vt, ExecType et) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.SortKeys ) {
				return (SortKeys)lop;
			}
		}
		
		SortKeys retVal = new SortKeys(input1, op, dt, vt, et);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal;
	}

	// This method is invoked ONLY for the case of Weighted SortKeys executing in CP
	public static SortKeys constructSortByValueLop(Lop input1, Lop input2, OperationTypes op, 
			DataType dt, ValueType vt, ExecType et) {
		
		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());
		// find intersection of input1.getOutputs() and input2.getOutputs();
		set1.retainAll(input2.getOutputs());
		
		for (Lop lop  : set1) {
			if ( lop.type == Lop.Type.SortKeys ) {
				return (SortKeys)lop;
			}
		}
		
		SortKeys retVal = new SortKeys(input1, input2, op, dt, vt, et);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal;
	}


}
