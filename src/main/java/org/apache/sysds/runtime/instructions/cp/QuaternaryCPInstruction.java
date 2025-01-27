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
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

public class QuaternaryCPInstruction extends ComputationCPInstruction {

	private final CPOperand input4;
	private final int _numThreads;

	private QuaternaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
			CPOperand out, int k, String opcode, String istr) {
		super(CPType.Quaternary, op, in1, in2, in3, out, opcode, istr);
		input4 = in4;
		_numThreads = k;
	}

	public static QuaternaryCPInstruction parseInstruction(String inst) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(inst);
		String opcode = parts[0];
		
		if( opcode.equalsIgnoreCase(Opcodes.WSLOSS.toString()) || opcode.equalsIgnoreCase(Opcodes.WDIVMM.toString()) || opcode.equalsIgnoreCase(Opcodes.WCEMM.toString()) )
		{
			InstructionUtils.checkNumFields ( parts, 7 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand in4 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			int k = Integer.parseInt(parts[7]);
			
			if( opcode.equalsIgnoreCase(Opcodes.WSLOSS.toString()) )
				return new QuaternaryCPInstruction(new QuaternaryOperator(WeightsType.valueOf(parts[6])), in1, in2, in3, in4, out, k, opcode, inst);	
			else if( opcode.equalsIgnoreCase(Opcodes.WDIVMM.toString()) )
				return new QuaternaryCPInstruction(new QuaternaryOperator(WDivMMType.valueOf(parts[6])), in1, in2, in3, in4, out, k, opcode, inst);				
			else if( opcode.equalsIgnoreCase(Opcodes.WCEMM.toString()) )
				return new QuaternaryCPInstruction(new QuaternaryOperator(WCeMMType.valueOf(parts[6])), in1, in2, in3, in4, out, k, opcode, inst);
		}
		else if( opcode.equalsIgnoreCase(Opcodes.WSIGMOID.toString()) )
		{
			InstructionUtils.checkNumFields ( parts, 6 );
			
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			int k = Integer.parseInt(parts[6]);
			
			if( opcode.equalsIgnoreCase(Opcodes.WSIGMOID.toString()) )
				return new QuaternaryCPInstruction(new QuaternaryOperator(WSigmoidType.valueOf(parts[5])), in1, in2, in3, null, out, k, opcode, inst);
		}
		else if( opcode.equalsIgnoreCase(Opcodes.WUMM.toString()) )
		{
			InstructionUtils.checkNumFields ( parts, 7 );
			
			String uopcode = parts[1];
			CPOperand in1 = new CPOperand(parts[2]);
			CPOperand in2 = new CPOperand(parts[3]);
			CPOperand in3 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
			int k = Integer.parseInt(parts[7]);
			
			return new QuaternaryCPInstruction(new QuaternaryOperator(WUMMType.valueOf(parts[6]),uopcode), in1, in2, in3, null, out, k, opcode, inst);
		}
		
		throw new DMLRuntimeException("Unexpected opcode in QuaternaryCPInstruction: " + inst);
	}

	public CPOperand getInput4() {
		return input4;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		
		MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
		MatrixBlock matBlock2 = ec.getMatrixInput(input2.getName());
		MatrixBlock matBlock3 = ec.getMatrixInput(input3.getName());
		MatrixBlock matBlock4 = null;
		if( qop.hasFourInputs() ) {
			if (input4.getDataType() == DataType.SCALAR)
				matBlock4 = new MatrixBlock(ec.getScalarInput(input4).getDoubleValue());
			else
				matBlock4 = ec.getMatrixInput(input4.getName());
		}
		
		//core execute
		MatrixBlock out = matBlock1.quaternaryOperations(qop, matBlock2, matBlock3, matBlock4, new MatrixBlock(), _numThreads);
		
		//release inputs and output
		ec.releaseMatrixInput(input1.getName(), input2.getName(), input3.getName());
		if( qop.wtype1 != null || qop.wtype4 != null ) { //wsloss/wcemm
			if( (qop.wtype1 != null && qop.wtype1.hasFourInputs()) ||
				(qop.wtype4 != null && qop.wtype4.hasFourInputs()) )
				if (input4.getDataType() == DataType.MATRIX) {
					ec.releaseMatrixInput(input4.getName());
				}
			ec.setVariable(output.getName(), new DoubleObject(out.get(0, 0)));
		}
		else { //wsigmoid / wdivmm / wumm
			if( qop.wtype3 != null && qop.wtype3.hasFourInputs() )
				if (input4.getDataType() == DataType.MATRIX) {
					ec.releaseMatrixInput(input4.getName());
				}
			ec.setMatrixOutput(output.getName(), out);
		}
	}
}
