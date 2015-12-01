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
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;


public class ScalarBuiltinCPInstruction extends BuiltinUnaryCPInstruction
{	
	
	public ScalarBuiltinCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr)
	{
		super(op, in, out, 1, opcode, instr);
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		String opcode = getOpcode();
		SimpleOperator dop = (SimpleOperator) _optr;
		ScalarObject sores = null;
		ScalarObject so = null;
		
		//get the scalar input 
		so = ec.getScalarInput( input1.getName(), input1.getValueType(), input1.isLiteral() );
			
		//core execution
		if ( opcode.equalsIgnoreCase("print") ) {
			String outString = so.getStringValue();
			
			// print to stdout only when suppress flag in DMLScript is not set.
			// The flag will be set, for example, when SystemML is invoked in fenced mode from Jaql.
			if (!DMLScript.suppressPrint2Stdout())
				System.out.println(outString);
			
			// String that is printed on stdout will be inserted into symbol table (dummy, not necessary!) 
			sores = new StringObject(outString);
		}
		else if ( opcode.equalsIgnoreCase("stop") ) {
			String msg = so.getStringValue();
			throw new DMLScriptException(msg);
		}
		else {
			//Inputs for all builtins other than PRINT are treated as DOUBLE.
			if ( so instanceof IntObject  && output.getValueType() == ValueType.INT )
			{
				long rval = (long) dop.fn.execute(so.getLongValue());
				sores = (ScalarObject) new IntObject(rval);
			}
			else 
			{
				double rval = dop.fn.execute(so.getDoubleValue());
				sores = (ScalarObject) new DoubleObject(rval);
			}
		}
		
		ec.setScalarOutput(output.getName(), sores);
	}

}
