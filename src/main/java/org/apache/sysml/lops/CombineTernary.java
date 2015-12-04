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

import java.util.HashSet;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineTernary extends Lop 
{
	
	public enum OperationTypes {
		PreCovWeighted, PreGroupedAggWeighted
	}; // PreCovUnweighted,PreGroupedAggWeighted will be CombineBinary

	OperationTypes operation;

	public CombineTernary( OperationTypes op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) {
		super(Lop.Type.CombineTernary, dt, vt);
		operation = op;
		this.addInput(input1);
		this.addInput(input2);
		this.addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);

		/*
		 * This lop can ONLY be executed as a STANDALONE job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		this.lps.addCompatibility(JobType.COMBINE);
		this.lps.setProperties(inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		this.lps.setProducesIntermediateOutput(true);
	}

	public String toString() {
		return "combineternary";
	}

	@Override
	public String getInstructions(int input_index1, int input_index2,
			int input_index3, int output_index) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "combineternary" );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(2).prepInputOperand(input_index3));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepInputOperand(output_index));

		return sb.toString();
	}

	public OperationTypes getOperation() {
		return operation;
	}

	public static CombineTernary constructCombineLop( OperationTypes op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) {

		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());

		// find intersection of input1.getOutputs() and input2.getOutputs()
		set1.retainAll(input2.getOutputs());

		// find intersection of the above result and input3.getOutputs()
		set1.retainAll(input3.getOutputs());
		
		for (Lop lop : set1) {
			if (lop.type == Lop.Type.CombineTernary) {
				CombineTernary combine = (CombineTernary) lop;
				if (combine.operation == op)
					return (CombineTernary) lop;
			}
		}

		CombineTernary comn = new CombineTernary(op, input1, input2, input3, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return comn;
	}

}
