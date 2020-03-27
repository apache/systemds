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

package org.apache.sysds.lops;

 
import org.apache.sysds.lops.LopProperties.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform Sum of a matrix with another matrix multiplied by Scalar.
 */
public class Ternary extends Lop 
{
	private final OpOp3 _op;
		
	public Ternary(OpOp3 op, Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.Ternary, dt, vt);
		_op = op;
		init(input1, input2, input3, et);
	}

	private void init(Lop input1, Lop input2, Lop input3, ExecType et) {
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		lps.setProperties( inputs, et);
	}
	
	@Override
	public String toString() {
		return "Operation = t("+_op.toString()+")";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output)  {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _op.toString() );
		
		//process three operands and output
		String[] inputs = new String[]{input1, input2, input3};
		for( int i=0; i<3; i++ ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(i).prepInputOperand(inputs[i]) );
		}
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output) );
		
		return sb.toString();
	}
}
