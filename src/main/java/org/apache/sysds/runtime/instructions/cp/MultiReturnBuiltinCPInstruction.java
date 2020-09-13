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

	private MultiReturnBuiltinCPInstruction(Operator op, CPOperand input1, ArrayList<CPOperand> outputs, String opcode,
			String istr) {
		super(CPType.MultiReturnBuiltin, op, input1, null, outputs.get(0), opcode, istr);
		_outputs = outputs;
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
	
	public static MultiReturnBuiltinCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		// first part is always the opcode
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("qr") ) {
			// one input and two ouputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("lu") ) {
			CPOperand in1 = new CPOperand(parts[1]);
			
			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else if ( opcode.equalsIgnoreCase("eigen") ) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);
			
		}
		else if ( opcode.equalsIgnoreCase("svd") ) {
			CPOperand in1 = new CPOperand(parts[1]);

			// one input and three outputs
			outputs.add ( new CPOperand(parts[2], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[3], ValueType.FP64, DataType.MATRIX) );
			outputs.add ( new CPOperand(parts[4], ValueType.FP64, DataType.MATRIX) );
			
			return new MultiReturnBuiltinCPInstruction(null, in1, outputs, opcode, str);

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
		MatrixBlock[] out = LibCommonsMath.multiReturnOperations(in, getOpcode());
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
	@SuppressWarnings({"rawtypes", "unchecked"})
	public Pair[] getLineageItems(ExecutionContext ec) {
		LineageItem[] inputLineage = LineageItemUtils.getLineage(ec, input1,input2,input3);
		ArrayList<Pair> items = new ArrayList<>();
		for (CPOperand out : _outputs)
			items.add(Pair.of(out.getName(), new LineageItem(getOpcode(), inputLineage)));
		return items.toArray(new Pair[items.size()]);
	}
}
