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

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLScriptException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.matrix.operators.UnaryOperator;

public class UnaryScalarCPInstruction extends UnaryMatrixCPInstruction {

	protected UnaryScalarCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		super(op, in, out, opcode, instr);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		ScalarObject sores = null;
		ScalarObject so = null;
		
		//get the scalar input 
		so = ec.getScalarInput(input1);
		
		//core execution
		if ( opcode.equalsIgnoreCase("print") ) {
			String outString = so.getLanguageSpecificStringValue();
			
			// print to stdout only when suppress flag in DMLScript is not set.
			// The flag will be set, for example, when SystemDS is invoked in fenced mode from Jaql.
			if (!DMLScript.suppressPrint2Stdout())
				System.out.println(outString);
			
			// String that is printed on stdout will be inserted into symbol table (dummy, not necessary!) 
			sores = new StringObject(outString);
		}
		else if ( opcode.equalsIgnoreCase("stop") ) {
			throw new DMLScriptException(so.getStringValue());
		}
		else if ( opcode.equalsIgnoreCase("assert") ) {
			sores = new BooleanObject(so.getBooleanValue());
			if(!so.getBooleanValue()) {
				String fileName = this.getFilename() == null ? "" : this.getFilename() + " "; 
				throw new DMLScriptException("assertion failed at " + fileName  + this.getBeginLine() + ":" + this.getBeginColumn() + "-" + this.getEndLine() + ":" + this.getEndColumn());
			}
		}
		else {
			UnaryOperator dop = (UnaryOperator) _optr;
			if ( so instanceof IntObject && output.getValueType() == ValueType.INT64 )
				sores = new IntObject((long)dop.fn.execute(so.getLongValue()));
			else if( so instanceof BooleanObject && output.getValueType() == ValueType.BOOLEAN )
				sores = new BooleanObject(dop.fn.execute(so.getBooleanValue()));
			else
				sores = new DoubleObject(dop.fn.execute(so.getDoubleValue()));
		}
		
		ec.setScalarOutput(output.getName(), sores);
	}

}
