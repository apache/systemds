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

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class ScalarScalarArithmeticCPInstruction extends ArithmeticBinaryCPInstruction
{	
	public ScalarScalarArithmeticCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException{
		ScalarObject so1 = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		BinaryOperator dop = (BinaryOperator) _optr;
		ScalarObject sores = null;

		//compute output value, incl implicit type promotion if necessary
		if( so1 instanceof StringObject || so2 instanceof StringObject ) {
			if( !getOpcode().equals("+") ) //not string concatenation
				throw new DMLRuntimeException("Arithmetic '"+getOpcode()+"' not supported over string inputs.");
			sores = new StringObject( dop.fn.execute(
				so1.getLanguageSpecificStringValue(), so2.getLanguageSpecificStringValue()) );
		}
		else if( so1 instanceof DoubleObject || so2 instanceof DoubleObject || output.getValueType()==ValueType.DOUBLE ) {
			sores = new DoubleObject( dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue()) );
		}
		else if( so1 instanceof IntObject || so2 instanceof IntObject ) {
			double tmp = dop.fn.execute(so1.getLongValue(), so2.getLongValue());
			if( tmp > Long.MAX_VALUE ) //cast to long if no overflow, otherwise controlled exception
				throw new DMLRuntimeException("Integer operation created numerical result overflow ("+tmp+" > "+Long.MAX_VALUE+").");
			sores = new IntObject((long) tmp);
		}
		else { //all boolean
			//NOTE: boolean-boolean arithmetic treated as double for consistency with R
			sores = new DoubleObject( dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue()) );
		}
		
		ec.setScalarOutput(output.getName(), sores);
	}
}
