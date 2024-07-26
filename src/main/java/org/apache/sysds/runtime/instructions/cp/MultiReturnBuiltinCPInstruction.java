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
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
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

public class MultiReturnBuiltinCPInstruction extends ComputationCPInstruction {

	protected ArrayList<CPOperand> _outputs;
	protected int _numThreads;

	private MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs, String opcode,
			String istr, int threads) {
		super(CPType.MultiReturnBuiltin, op, input1, null, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_numThreads = threads;
	}
	
	private MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, CPOperand input2, ArrayList<CPOperand> outputs, String opcode,
			String istr, int threads) {
		super(CPType.MultiReturnBuiltin, op, input1, input2, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_numThreads = threads;
	}
	
	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public List<CPOperand> getOutputs(){
		return _outputs;
	}

	public String[] getOutputNames(){
		return _outputs.parallelStream().map(output -> output.getName()).toArray(String[]::new);
	}
	
	public static MultiReturnBuiltinCPInstruction parseInstruction(String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		// first part is always the opcode
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("qr") ) {
			// one input and two ouputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[4]);
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
		}
		else if ( opcode.equalsIgnoreCase("lu") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[5]);
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
			
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[4]);
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
		}
		else if( opcode.equalsIgnoreCase("fft")){
			if(parts.length == 5) {
				// one input and two outputs
				CPOperand in1 = new CPOperand(parts[1]);
				outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
				outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
				int threads = Integer.parseInt(parts[4]);
				return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
			}
			else if(parts.length == 4) {
				// one input and two outputs
				outputs.add(new CPOperand(parts[1], ValueType.FP64, DataType.MATRIX));
				outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
				int threads = Integer.parseInt(parts[3]);
				return new MultiReturnBuiltinCPInstruction(null, null, outputs, opcode, str, threads);
			}
			else 
				throw new NotImplementedException("Invalid number of arguments for FFT.");
		}
		else if(parts.length == 5 && opcode.equalsIgnoreCase("fft_linearized")) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[4]);

			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);

		}
		else if(parts.length == 3 && opcode.equalsIgnoreCase("fft_linearized")) {
			// one input and two outputs
			outputs.add(new CPOperand(parts[1], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX));
			int threads = Integer.parseInt(parts[3]);

			return new MultiReturnBuiltinCPInstruction(null, null, outputs, opcode, str, threads);

		} else if (opcode.equalsIgnoreCase("img_transform_matrix")) {
			// 5 inputs and two outputs
			CPOperand in1 = new CPOperand(parts[1]); //transformation matrix
			CPOperand in2 = new CPOperand(parts[2]); //original width
			CPOperand in3 = new CPOperand(parts[3]); //original height
			CPOperand in4 = new CPOperand(parts[4]); //output width
			CPOperand in5 = new CPOperand(parts[5]); //output height

			outputs.add(new CPOperand(parts[6], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[7], ValueType.BOOLEAN, DataType.SCALAR));
			throw new NotImplementedException("Has yet to be done. Check number of inputs" + Arrays.toString(parts) + "; Number of parts: " + parts.length);


		} else if ( opcode.equalsIgnoreCase("stft") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[4]);

			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
		}
		else if ( opcode.equalsIgnoreCase("svd") ) {
			CPOperand in1 = new CPOperand(parts[1]);

			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX) );
			int threads = Integer.parseInt(parts[5]);
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str, threads);
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
		if(!LibCommonsMath.isSupportedMultiReturnOperation(getOpcode()))
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + getOpcode());
		
		MatrixBlock in = ec.getMatrixInput(input1.getName());
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in, getOpcode(), _numThreads);
		ec.releaseMatrixInput(input1.getName());
		for(int i=0; i < _outputs.size(); i++) {
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
		final Pair<String,LineageItem>[] ret = new Pair[_outputs.size()];
		for(int i = 0; i < _outputs.size(); i++){
			CPOperand out = _outputs.get(i);
			ret[i] = Pair.of(out.getName(), new LineageItem(getOpcode(), inputLineage));
		}
		return ret; 
	}
}
