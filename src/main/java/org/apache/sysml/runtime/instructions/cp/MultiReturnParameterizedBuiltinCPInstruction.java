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
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.transform.encode.Encoder;
import org.apache.sysml.runtime.transform.encode.EncoderFactory;


public class MultiReturnParameterizedBuiltinCPInstruction extends ComputationCPInstruction 
{
	protected ArrayList<CPOperand> _outputs;
	
	public MultiReturnParameterizedBuiltinCPInstruction(Operator op, CPOperand input1, CPOperand input2, ArrayList<CPOperand> outputs, String opcode, String istr ) {
		super(op, input1, input2, outputs.get(0), opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MultiReturnBuiltin;
		_outputs = outputs;
	}
	
	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MultiReturnParameterizedBuiltinCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<CPOperand>();
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("transformencode") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add ( new CPOperand(parts[3], ValueType.DOUBLE, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.STRING, DataType.FRAME) );
			return new MultiReturnParameterizedBuiltinCPInstruction(null, in1, in2, outputs, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{
		//obtain and pin input frame
		FrameBlock fin = ec.getFrameInput(input1.getName());
		String spec = ec.getScalarInput(input2.getName(), input2.getValueType(), input2.isLiteral()).getStringValue();
		
		//execute block transform encode
		Encoder encoder = EncoderFactory.createEncoder(spec, fin.getNumColumns());
		MatrixBlock data = encoder.encode(fin, new MatrixBlock(fin.getNumRows(), fin.getNumColumns(), false)); //build and apply
		FrameBlock meta = encoder.getMetaData(new FrameBlock(fin.getNumColumns(), ValueType.STRING));
		
		//release input and outputs
		ec.releaseFrameInput(input1.getName());
		ec.setMatrixOutput(getOutput(0).getName(), data);
		ec.setFrameOutput(getOutput(1).getName(), meta);
	}
}
