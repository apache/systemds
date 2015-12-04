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
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineUnary extends Lop
{

	

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	
	public CombineUnary(Lop input1, DataType dt, ValueType vt) 
	{
		super(Lop.Type.CombineUnary, dt, vt);	
		this.addInput(input1);
		input1.addOutput(this);
				
		/*
		 *  This lop gets piggybacked into, and can ONLY be executed in SORT_KEYS job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.SORT);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}
	
	public String toString()
	{
		return "combineunary";		
	}

	@Override
	public String getInstructions(int input_index1, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "combineunary" );
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output_index));
		
		return sb.toString();
	}

	public static CombineUnary constructCombineLop(Lop input1, 
			DataType dt, ValueType vt) {
		
		HashSet<Lop> set1 = new HashSet<Lop>();
		set1.addAll(input1.getOutputs());
			
		for (Lop lop  : set1) {
			if ( lop.type == Lop.Type.CombineUnary ) {
				return (CombineUnary)lop;
			}
		}
		
		CombineUnary comn = new CombineUnary(input1, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndLine());
		return comn;
	}
	
 
}
