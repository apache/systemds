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
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.ValueComparisonFunction;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class ScalarScalarRelationalCPInstruction extends RelationalBinaryCPInstruction
{	
	public ScalarScalarRelationalCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		ScalarObject so1 = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		ValueComparisonFunction vcomp = ((ValueComparisonFunction)((BinaryOperator)_optr).fn);
		boolean rval = false;
		
		//compute output value, incl implicit type promotion if necessary
		if( so1 instanceof StringObject || so2 instanceof StringObject )
			rval = vcomp.compare ( so1.getStringValue(), so2.getStringValue() );
		else if( so1 instanceof DoubleObject || so2 instanceof DoubleObject )
			rval = vcomp.compare( so1.getDoubleValue(), so2.getDoubleValue() );
		else if( so1 instanceof IntObject || so2 instanceof IntObject )
			rval = vcomp.compare( so1.getLongValue(), so2.getLongValue() );
		else //all boolean
			rval = vcomp.compare( so1.getBooleanValue(), so2.getBooleanValue() );
		
		//set boolean output value
		ec.setScalarOutput(output.getName(), new BooleanObject(rval));
	}
}
