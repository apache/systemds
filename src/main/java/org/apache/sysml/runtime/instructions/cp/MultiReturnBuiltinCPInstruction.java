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

import java.util.ArrayList;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.LibCommonsMath;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class MultiReturnBuiltinCPInstruction extends ComputationCPInstruction 
{
	
	int arity;
	protected ArrayList<CPOperand> _outputs;
	
	public MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs, String opcode, String istr )
	{
		super(op, input1, null, outputs.get(0), opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MultiReturnBuiltin;
		_outputs = outputs;
	}

	public int getArity() {
		return arity;
	}
	
	public CPOperand getOutput(int i)
	{
		return _outputs.get(i);
	}
	
	public static MultiReturnBuiltinCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<CPOperand>();
		// first part is always the opcode
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("qr") ) {
			// one input and two ouputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("lu") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		String opcode = getOpcode();
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		MatrixBlock[] out = null;
		
		if(LibCommonsMath.isSupportedMultiReturnOperation(opcode))
			out = LibCommonsMath.multiReturnOperations(mo, opcode);
		else 
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);

		
		for(int i=0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}
}
