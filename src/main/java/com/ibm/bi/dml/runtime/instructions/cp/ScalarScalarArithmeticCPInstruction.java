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

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.Divide;
import com.ibm.bi.dml.runtime.functionobjects.Power;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarArithmeticCPInstruction extends ArithmeticBinaryCPInstruction
{
	
	public ScalarScalarArithmeticCPInstruction(Operator op, 
								   CPOperand in1, 
								   CPOperand in2,
								   CPOperand out, 
								   String opcode,
								   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException{
		// 1) Obtain data objects associated with inputs 
		ScalarObject so1 = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		ScalarObject sores = null;
		
		
		// 2) Compute the result value & make an appropriate data object 
		BinaryOperator dop = (BinaryOperator) _optr;
		
		if ( input1.getValueType() == ValueType.STRING 
			 || input2.getValueType() == ValueType.STRING ) 
		{
			//pre-check (for robustness regarding too long strings)
			String val1 = so1.getStringValue();
			String val2 = so2.getStringValue();
			StringObject.checkMaxStringLength(val1.length() + val2.length());
			
			String rval = dop.fn.execute(val1, val2);
			sores = (ScalarObject) new StringObject(rval);
		}
		else if ( so1 instanceof IntObject && so2 instanceof IntObject ) {
			if ( dop.fn instanceof Divide || dop.fn instanceof Power ) {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				double rval = dop.fn.execute ( so1.getLongValue(), so2.getLongValue() );
				sores = (ScalarObject) new DoubleObject(rval);
			}
			else {
				// If both inputs are of type INT then output must be an INT if operation is not divide or power
				double tmpVal = dop.fn.execute ( so1.getLongValue(), so2.getLongValue() );
				//cast to long if no overflow, otherwise controlled exception
				if( tmpVal > Long.MAX_VALUE )
					throw new DMLRuntimeException("Integer operation created numerical result overflow ("+tmpVal+" > "+Long.MAX_VALUE+").");
				long rval = (long) tmpVal; 
				sores = (ScalarObject) new IntObject(rval);
			}
		}
		
		else {
			// If either of the input is of type DOUBLE then output is a DOUBLE
			double rval = dop.fn.execute ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new DoubleObject(rval); 
		}
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.getName(), sores);
	}
}
