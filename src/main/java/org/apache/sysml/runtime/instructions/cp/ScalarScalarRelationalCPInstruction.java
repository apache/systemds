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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class ScalarScalarRelationalCPInstruction extends RelationalBinaryCPInstruction
{
	
	public ScalarScalarRelationalCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2,
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException
	{
		ScalarObject so1 = ec.getScalarInput(input1.getName(), input1.getValueType(), input1.isLiteral());
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		ScalarObject sores = null;
		
		BinaryOperator dop = (BinaryOperator) _optr;
		
		if ( so1 instanceof IntObject && so2 instanceof IntObject ) {
			boolean rval = dop.fn.compare ( so1.getLongValue(), so2.getLongValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof DoubleObject && so2 instanceof DoubleObject ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof IntObject && so2 instanceof DoubleObject) {
			boolean rval = dop.fn.compare ( so1.getLongValue(), so2.getDoubleValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof DoubleObject && so2 instanceof IntObject ) {
			boolean rval = dop.fn.compare ( so1.getDoubleValue(), so2.getLongValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof BooleanObject && so2 instanceof BooleanObject ) {
			boolean rval = dop.fn.compare ( so1.getBooleanValue(), so2.getBooleanValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else if ( so1 instanceof StringObject && so2 instanceof StringObject ) {
			boolean rval = dop.fn.compare ( so1.getStringValue(), so2.getStringValue() );
			sores = (ScalarObject) new BooleanObject(rval); 
		}
		else throw new DMLRuntimeException("compare(): Invalid combination of value types.");
		
		ec.setScalarOutput(output.getName(), sores);
	}
}
