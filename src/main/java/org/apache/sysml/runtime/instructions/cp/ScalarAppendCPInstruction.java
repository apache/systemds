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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.Operator;

public final class ScalarAppendCPInstruction extends AppendCPInstruction
{	
	public ScalarAppendCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr) {
		super(op, in1, in2, in3, out, type, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//Append type: STRING
		
		//get input strings (vars or literals)
		ScalarObject so1 = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
		ScalarObject so2 = ec.getScalarInput( input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		//pre-checks
		String val1 = so1.getStringValue();
		String val2 = so2.getStringValue();
		StringObject.checkMaxStringLength( val1.length()+val2.length() );
		
		//core execution
		String outString = val1 + "\n" + val2;			
		ScalarObject sores = new StringObject(outString);
		
		//set output
		ec.setScalarOutput(output.getName(), sores);		
	}
}
