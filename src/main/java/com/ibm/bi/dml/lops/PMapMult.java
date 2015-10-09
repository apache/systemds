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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class PMapMult extends Lop 
{	
	public static final String OPCODE = "pmapmm";

	/**
	 * 
	 * @param input1
	 * @param input2
	 * @param dt
	 * @param vt
	 * @param rightCache
	 * @param emptyBlocks
	 * @param aggregate
	 * @param et
	 * @throws LopsException
	 */
	public PMapMult(Lop input1, Lop input2, DataType dt, ValueType vt) 
		throws LopsException 
	{
		super(Lop.Type.MapMult, dt, vt);		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		//setup MR parameters 
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	public String toString() {
		return "Operation = PMapMM";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		//Spark instruction generation		
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(OPCODE);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( prepOutputOperand(output));
		
		return sb.toString();
	}
}
