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
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


/**
 * Lop to perform Sum of a matrix with another matrix multiplied by Scalar.
 */
public class Ternary extends Lop 
{
	public enum OperationType {
		PLUS_MULT,
		MINUS_MULT,
		IFELSE,
	}
	
	private final OperationType _type;
		
	public Ternary(OperationType op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.Ternary, dt, vt);
		_type = op;
		init(input1, input2, input3, et);
	}

	private void init(Lop input1, Lop input2, Lop input3, ExecType et) {
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.CP ||  et == ExecType.SPARK || et == ExecType.GPU ){
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
		else if( et == ExecType.MR ) {
			lps.addCompatibility(JobType.GMR);
			lps.setProperties( inputs, et, ExecLocation.Reduce, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	@Override
	public String toString() {
		return "Operation = t("+_type.name().toLowerCase()+")";
	}
	
	public String getOpString() {
		switch( _type ) {
			case PLUS_MULT: return "+*";
			case MINUS_MULT: return "-*";
			case IFELSE: return "ifelse";
		}
		return null;
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) 
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpString() );
		
		//process three operands
		String[] inputs = new String[]{input1, input2, input3};
		for( int i=0; i<3; i++ ) {
			sb.append( OPERAND_DELIMITOR );
			if( getExecType()==ExecType.MR && getInputs().get(i).getDataType().isScalar() )
				sb.append( getInputs().get(i).prepScalarLabel() );
			else
				sb.append( getInputs().get(i).prepInputOperand(inputs[i]) );
		}
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input1, int input2, int input3, int output) {
		return getInstructions(String.valueOf(input1), String.valueOf(input2), 
				String.valueOf(input3), String.valueOf(output));
	}
}
