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
package org.apache.sysds.runtime.instructions.gpu;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.utils.GPUStatistics;

import jcuda.Pointer;

public class MatrixReshapeGPUInstruction extends GPUInstruction {
	
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	private final CPOperand _opByRow;
	
	protected MatrixReshapeGPUInstruction(Operator op, String opcode, String istr, 
			CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out) {
		super(op, in1, null, out, opcode, istr);
		_opRows = in2;
		_opCols = in3;
		_opByRow = in4;
	}
	
	public static MatrixReshapeGPUInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// ToDo: Tensor GPU support

		InstructionUtils.checkNumFields( parts, 6 );
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);

		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an MatrixReshapeGPUInstruction: " + str);
		else
			return new MatrixReshapeGPUInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), opcode, str, in1, in2, in3, in4, out);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		int rows = (int)ec.getScalarInput(_opRows).getLongValue(); //save cast
		int cols = (int)ec.getScalarInput(_opCols).getLongValue(); //save cast
		BooleanObject byRow = (BooleanObject) ec.getScalarInput(_opByRow.getName(), ValueType.BITSET, _opByRow.isLiteral());
		
		GPUStatistics.incrementNoOfExecutedGPUInst();
		String instName = getExtendedOpcode();
		GPUContext gCtx = ec.getGPUContext(0); 
		MatrixObject mat = getMatrixInputForGPUInstruction(ec, _input1.getName());
		if(rows*cols != mat.getNumRows()*mat.getNumColumns()) {
			throw new DMLRuntimeException("Incorrect number of rows and cols in rshape instruction");
		}
		// We currently support only dense rshape
		Pointer inPtr = LibMatrixCUDA.getDensePointer(gCtx, mat, instName);
		MatrixObject out = LibMatrixCUDA.getDenseMatrixOutputForGPUInstruction(ec, instName, _output.getName(), rows,
			cols, false);
		Pointer outPtr = LibMatrixCUDA.getDensePointer(gCtx, out, instName);
		if(byRow.getBooleanValue()) {
			// byrow = TRUE is simple memcpy and metadata update
			LibMatrixCUDA.deviceCopy(instName, inPtr, outPtr, LibMatrixCUDA.toInt(mat.getNumRows()), LibMatrixCUDA.toInt(mat.getNumColumns()));
		}
		else  {
			// byrow = FALSE uses a custom kernel to perform rshape
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("colwise_reshape", 
				ExecutionConfig.getConfigForSimpleVectorOperations(LibMatrixCUDA.toInt(rows*cols)),
				inPtr, outPtr, LibMatrixCUDA.toInt(rows*cols), 
				LibMatrixCUDA.toInt(mat.getNumRows()), LibMatrixCUDA.toInt(mat.getNumColumns()),
				rows, cols);
		}
		ec.releaseMatrixInputForGPUInstruction(_input1.getName());
		ec.releaseMatrixOutputForGPUInstruction(_output.getName());
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(_output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, _input1, _opRows, _opCols, _opByRow)));
	}

}
