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

import java.util.HashSet;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineBinary extends Lop 
{

	
	public enum OperationTypes {PreSort, PreCentralMoment, PreCovUnweighted, PreGroupedAggUnweighted}; // (PreCovWeighted,PreGroupedAggWeighted) will be CombineTertiary	
	OperationTypes operation;

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	
	public CombineBinary(OperationTypes op, Lop input1, Lop input2, DataType dt, ValueType vt) 
	{
		super(Lop.Type.CombineBinary, dt, vt);	
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		/*
		 *  This lop can ONLY be executed as a STANDALONE job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		lps.addCompatibility(JobType.COMBINE);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
		this.lps.setProducesIntermediateOutput(true);
	}
	
	public String toString()
	{
		return "combinebinary";		
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) 
		throws LopsException
	{
		// Determine whether or not the second input denotes weights vector.
		// CombineBinary can be used to combine (data,weights) vectors or (data1,data2) vectors  
		boolean isSecondInputIsWeight = true;
		if ( operation == OperationTypes.PreCovUnweighted || operation == OperationTypes.PreGroupedAggUnweighted ) {
			isSecondInputIsWeight = false;
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "combinebinary" );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOperand(String.valueOf(isSecondInputIsWeight), DataType.SCALAR, ValueType.BOOLEAN) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	public OperationTypes getOperation() { 
		return operation;
	}
	
	public static CombineBinary constructCombineLop(OperationTypes op, Lop input1, 
			Lop input2, DataType dt, ValueType vt) {
		
		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());
		
		// find intersection of input1.getOutputs() and input2.getOutputs();
		set1.retainAll(input2.getOutputs());
		
		for (Lop lop  : set1) {
			if ( lop.type == Lop.Type.CombineBinary ) {
				CombineBinary combine = (CombineBinary)lop;
				if ( combine.operation == op)
					return (CombineBinary)lop;
			}
		}
		
		CombineBinary comn = new CombineBinary(op, input1, input2, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return comn;
	}
 
}
