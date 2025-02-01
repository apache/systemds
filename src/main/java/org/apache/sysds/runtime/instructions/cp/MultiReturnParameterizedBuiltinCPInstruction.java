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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;

public class MultiReturnParameterizedBuiltinCPInstruction extends ComputationCPInstruction {
	protected final ArrayList<CPOperand> _outputs;
	protected final boolean _metaReturn;
	
	private MultiReturnParameterizedBuiltinCPInstruction(Operator op, CPOperand input1, CPOperand input2,
		boolean metaReturn, ArrayList<CPOperand> outputs, String opcode, String istr) {
		super(CPType.MultiReturnBuiltin, op, input1, input2, outputs.get(0), opcode, istr);
		_outputs = outputs;
		_metaReturn = metaReturn;
	}

	public CPOperand getOutput(int i) {
		return _outputs.get(i);
	}

	public List<CPOperand> getOutputs() {
		return _outputs;
	}

	public String[] getOutputNames() {
		return _outputs.stream().map(CPOperand::getName).toArray(String[]::new);
	}

	public static MultiReturnParameterizedBuiltinCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString())) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			int pos = 3;
			boolean metaReturn = true;
			if( parts.length == 7 ) //no need for meta data
				metaReturn = new CPOperand(parts[pos++]).getLiteral().getBooleanValue();
			outputs.add(new CPOperand(parts[pos], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[pos+1], ValueType.STRING, DataType.FRAME));
			return new MultiReturnParameterizedBuiltinCPInstruction(
				null, in1, in2, metaReturn, outputs, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}

	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// obtain and pin input frame
		FrameBlock fin = ec.getFrameInput(input1.getName());
		String spec = ec.getScalarInput(input2).getStringValue();
		String[] colnames = fin.getColumnNames();

		// execute block transform encode
		MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, colnames, fin.getNumColumns(), null);
		// TODO: Assign #threads in compiler and pass via the instruction string
		int k = OptimizerUtils.getTransformNumThreads();
		MatrixBlock data = encoder.encode(fin, OptimizerUtils.getTransformNumThreads()); // build and apply
		FrameBlock meta = !_metaReturn ? new FrameBlock() :
			encoder.getMetaData(new FrameBlock(fin.getNumColumns(), ValueType.STRING), k);
		meta.setColumnNames(colnames);

		// release input and outputs
		ec.releaseFrameInput(input1.getName());
		ec.setMatrixOutput(getOutput(0).getName(), data);
		ec.setFrameOutput(getOutput(1).getName(), meta);
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
		for(int i = 0; i < _outputs.size(); i++){
			CPOperand out = _outputs.get(i);
			ret[i] = Pair.of(out.getName(), new LineageItem(getOpcode(), inputLineage));
		}
		return ret; 
	}

	public boolean getMetaReturn() {
		return _metaReturn;
	}
}
