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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class ScalarScalarBuiltinCPInstruction extends BuiltinBinaryCPInstruction
{
	
	public ScalarScalarBuiltinCPInstruction(Operator op,
			  								CPOperand in1,
			  								CPOperand in2,
			  								CPOperand out,
			  								String opcode,
			  								String instr){
		super(op, in1, in2, out, 2, opcode, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		
		String opcode = getOpcode();
		ScalarObject sores = null;
		
		ScalarObject so1 = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
		ScalarObject so2 = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral() );
		
		if ( opcode.equalsIgnoreCase("print") ) {
			String buffer = "";
			if (input2.getValueType() != ValueType.STRING)
				throw new DMLRuntimeException("wrong value type in print");
			buffer = so2.getStringValue() + " ";
			
			if ( !DMLScript.suppressPrint2Stdout()) {
				switch (input1.getValueType()) {
					case INT:
						System.out.println(buffer + so1.getLongValue());
						break;
					case DOUBLE:
						System.out.println(buffer + so1.getDoubleValue());
						break;
					case BOOLEAN:
						System.out.println(buffer + so1.getBooleanValue());
						break;
					case STRING:
						System.out.println(buffer + so1.getStringValue());
						break;
					default:
						//do nothing
				}
			}
		}
		else {
			BinaryOperator dop = (BinaryOperator) _optr;
			
			if ( so1 instanceof IntObject 
					&& so2 instanceof IntObject 
					&& output.getValueType() == ValueType.INT) {
				long rval = (long) dop.fn.execute(so1.getLongValue(), so2.getLongValue());
				sores = (ScalarObject) new IntObject(rval);
			}
			else {
				double rval = dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue());
				sores = (ScalarObject) new DoubleObject(rval);
			}
		}
		
		// 3) Put the result value into ProgramBlock
		ec.setScalarOutput(output.getName(), sores);
	}
}
