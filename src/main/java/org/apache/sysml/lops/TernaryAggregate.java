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


/**
 * 
 * 
 */
public class TernaryAggregate extends Lop 
{
	
	private static final String OPCODE = "tak+*";
	
	//NOTE: currently only used for ta+*
	//private Aggregate.OperationTypes _aggOp = null;
	//private Binary.OperationTypes _binOp = null;
	
	//optional attribute for cp
	private int _numThreads = -1;
	
	public TernaryAggregate(Lop input1, Lop input2, Lop input3, Aggregate.OperationTypes aggOp, Binary.OperationTypes binOp, DataType dt, ValueType vt, ExecType et ) {
		this(input1, input2, input3, aggOp, binOp, dt, vt, et, 1);
	}
	
	/**
	 * @param et 
	 * @param input - input lop
	 * @param op - operation type
	 */
	public TernaryAggregate(Lop input1, Lop input2, Lop input3, Aggregate.OperationTypes aggOp, Binary.OperationTypes binOp, DataType dt, ValueType vt, ExecType et, int k ) 
	{
		super(Lop.Type.TernaryAggregate, dt, vt);
		
		//_aggOp = aggOp;	
		//_binOp = binOp;
		
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		_numThreads = k;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}
	
	@Override
	public String toString()
	{
		return "Operation: "+OPCODE;		
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepInputOperand(input3));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		
		if( getExecType() == ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );	
		}
		
		return sb.toString();
	}
}
