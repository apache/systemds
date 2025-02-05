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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.ValueComparisonFunction;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class BinaryScalarScalarCPInstruction extends BinaryCPInstruction {

	protected BinaryScalarScalarCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(CPType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		ScalarObject so1 = ec.getScalarInput(input1);
		ScalarObject so2 = ec.getScalarInput(input2);
		
		String opcode = getOpcode();
		BinaryOperator dop = (BinaryOperator) _optr;
		ScalarObject sores = null;
		
		//compare output value, incl implicit type promotion if necessary
		if( dop.fn instanceof ValueComparisonFunction ) {
			ValueComparisonFunction vcomp = (ValueComparisonFunction) dop.fn;
			if( so1 instanceof StringObject || so2 instanceof StringObject )
				sores = new  BooleanObject(vcomp.compare(so1.getStringValue(), so2.getStringValue()));
			else if( so1 instanceof DoubleObject || so2 instanceof DoubleObject )
				sores = new  BooleanObject(vcomp.compare(so1.getDoubleValue(), so2.getDoubleValue()));
			else if( so1 instanceof IntObject || so2 instanceof IntObject )
				sores = new  BooleanObject(vcomp.compare(so1.getLongValue(), so2.getLongValue()));
			else //all boolean
				sores = new  BooleanObject(vcomp.compare(so1.getBooleanValue(), so2.getBooleanValue()));
		}
		//compute output value, incl implicit type promotion if necessary
		else {
			if( so1 instanceof StringObject || so2 instanceof StringObject ) {
				if( !opcode.equals("+") ) //not string concatenation
					throw new DMLRuntimeException("Arithmetic '"+opcode+"' not supported over string inputs.");
				sores = new StringObject( dop.fn.execute(
					so1.getLanguageSpecificStringValue(), so2.getLanguageSpecificStringValue()) );
			}
			else if( so1 instanceof DoubleObject || so2 instanceof DoubleObject || output.getValueType()==ValueType.FP64 ) {
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
				if( opcode.equals(Opcodes.AND.toString()) || opcode.equals(Opcodes.OR.toString()) || opcode.equals(Opcodes.XOR.toString()) )
					sores = new BooleanObject( dop.fn.execute(so1.getBooleanValue(), so2.getBooleanValue()) );
				else
					sores = new DoubleObject( dop.fn.execute(so1.getDoubleValue(), so2.getDoubleValue()) );
			}
		}
		ec.setScalarOutput(output.getName(), sores);
	}
}
