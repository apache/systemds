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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class MultiReturnComplexMatrixBuiltinCPInstruction extends ComputationCPInstruction {

	protected final ArrayList<CPOperand> _outputs;
	protected final int _numThreads;

	private MultiReturnComplexMatrixBuiltinCPInstruction(Operator op, CPOperand input1, CPOperand input2,
		ArrayList<CPOperand> outputs, String opcode, String istr, int threads) {
		super(CPType.MultiReturnBuiltin, op, input1, input2, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_numThreads = threads;
	}

	private MultiReturnComplexMatrixBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs,
		String opcode, String istr, int threads) {
		super(CPType.MultiReturnBuiltin, op, input1, null, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_numThreads = threads;
	}

	private MultiReturnComplexMatrixBuiltinCPInstruction(Operator op, CPOperand input1, CPOperand input2,
		CPOperand input3, CPOperand input4, ArrayList<CPOperand> outputs, String opcode, String istr, int threads) {
		super(CPType.MultiReturnBuiltin, op, input1, input2, input3, input4, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_numThreads = threads;
	}

	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public List<CPOperand> getOutputs() {
		return _outputs;
	}

	public String[] getOutputNames() {
		return _outputs.parallelStream().map(output -> output.getName()).toArray(String[]::new);
	}

	public static MultiReturnComplexMatrixBuiltinCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		// first part is always the opcode
		String opcode = parts[0];

		if(parts.length == 6 && opcode.equalsIgnoreCase(Opcodes.IFFT.toString())) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[5]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, in2, outputs, opcode, str, threads);
		}
		else if(parts.length == 5 && opcode.equalsIgnoreCase(Opcodes.IFFT.toString())) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[4]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
		}
		else if(parts.length == 6 && opcode.equalsIgnoreCase(Opcodes.IFFT_LINEARIZED.toString())) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[5]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, in2, outputs, opcode, str, threads);
		}
		else if(parts.length == 5 && opcode.equalsIgnoreCase(Opcodes.IFFT_LINEARIZED.toString())) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[4]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
		}
		else if(parts.length == 7 && opcode.equalsIgnoreCase(Opcodes.STFT.toString())) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand windowSize = new CPOperand(parts[2]);
			CPOperand overlap = new CPOperand(parts[3]);
			outputs.add(new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[5], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[6]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, null, windowSize, overlap, outputs, opcode,
				str, threads);
		}
		else if(parts.length == 8 && opcode.equalsIgnoreCase(Opcodes.STFT.toString())) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand windowSize = new CPOperand(parts[3]);
			CPOperand overlap = new CPOperand(parts[4]);
			outputs.add(new CPOperand(parts[5], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[6], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[7]);

			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, in2, windowSize, overlap, outputs, opcode,
				str, threads);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.RCM.toString()) ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[5]);
			return new MultiReturnComplexMatrixBuiltinCPInstruction(null, in1, in2, outputs, opcode, str, threads);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	public int getNumOutputs() {
		return _outputs.size();
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(getOpcode().equals(Opcodes.STFT.toString()) && input2 == null)
			processSTFTInstruction(ec);
		else if(getOpcode().equals(Opcodes.STFT.toString()))
			processSTFTTwoInstruction(ec);
		else if(input2 == null)
			processOneInputInstruction(ec);
		else
			processTwoInputInstruction(ec);
	}

	private void processOneInputInstruction(ExecutionContext ec) {
		if(!LibCommonsMath.isSupportedMultiReturnOperation(getOpcode()))
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + getOpcode());

		MatrixBlock in = ec.getMatrixInput(input1.getName());
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in, getOpcode(), _numThreads);

		ec.releaseMatrixInput(input1.getName());

		for(int i = 0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}

	private void processTwoInputInstruction(ExecutionContext ec) {
		if(!LibCommonsMath.isSupportedMultiReturnOperation(getOpcode()))
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + getOpcode());

		MatrixBlock in1 = ec.getMatrixInput(input1.getName());
		MatrixBlock in2 = ec.getMatrixInput(input2.getName());
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in1, in2, getOpcode());
		ec.releaseMatrixInput(input1.getName(), input2.getName());

		for(int i = 0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}

	private void processSTFTInstruction(ExecutionContext ec) {
		if(!LibCommonsMath.isSupportedMultiReturnOperation(getOpcode()))
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + getOpcode());

		MatrixBlock in1 = ec.getMatrixInput(input1.getName());
		// MatrixBlock in2 = ec.getMatrixInput(input2.getName());
		int windowSize = Integer.parseInt(input3.getName());
		int overlap = Integer.parseInt(input4.getName());
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in1, getOpcode(), windowSize, overlap);
		ec.releaseMatrixInput(input1.getName());

		for(int i = 0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}

	private void processSTFTTwoInstruction(ExecutionContext ec) {
		if(!LibCommonsMath.isSupportedMultiReturnOperation(getOpcode()))
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + getOpcode());

		MatrixBlock in1 = ec.getMatrixInput(input1.getName());
		MatrixBlock in2 = ec.getMatrixInput(input2.getName());
		int windowSize = Integer.parseInt(input3.getName());
		int overlap = Integer.parseInt(input4.getName());
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in1, in2, getOpcode(), windowSize, overlap);
		ec.releaseMatrixInput(input1.getName(), input2.getName());

		for(int i = 0; i < _outputs.size(); i++) {
			ec.setMatrixOutput(_outputs.get(i).getName(), out[i]);
		}
	}

	@Override
	public boolean hasSingleLineage() {
		return false;
	}

	@Override
	@SuppressWarnings("unchecked")
	public Pair<String, LineageItem>[] getLineageItems(ExecutionContext ec) {
		LineageItem[] inputLineage = LineageItemUtils.getLineage(ec, input1, input2, input3);
		final Pair<String, LineageItem>[] ret = new Pair[_outputs.size()];
		for(int i = 0; i < _outputs.size(); i++) {
			CPOperand out = _outputs.get(i);
			ret[i] = Pair.of(out.getName(), new LineageItem(getOpcode(), inputLineage));
		}
		return ret;
	}
}
