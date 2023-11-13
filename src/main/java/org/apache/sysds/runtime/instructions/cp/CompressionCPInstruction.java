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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.apache.sysds.runtime.compress.lib.CLALibBinCompress;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class CompressionCPInstruction extends ComputationCPInstruction {
	private static final Log LOG = LogFactory.getLog(CompressionCPInstruction.class.getName());

	private final int _singletonLookupID;

	/** This is only for binned compression with 2 outputs*/
	protected final List<CPOperand> _outputs;

	private CompressionCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr,
		int singletonLookupID) {
		super(CPType.Compression, op, in, null, null, out, opcode, istr);
		_outputs = null;
		this._singletonLookupID = singletonLookupID;
	}

	private CompressionCPInstruction(Operator op, CPOperand in1, CPOperand in2, List<CPOperand> out, String opcode, String istr,
		int singletonLookupID) {
		super(CPType.Compression, op, in1, in2, null, out.get(0), opcode, istr);
		_outputs = out;
		this._singletonLookupID = singletonLookupID;
	}

	public static CompressionCPInstruction parseInstruction(String str) {
		InstructionUtils.checkNumFields(str, 2, 3, 4);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		if(parts.length == 5) {
			/** Compression with bins that returns two outputs*/
			List<CPOperand> outputs = new ArrayList<>();
			outputs.add(new CPOperand(parts[3]));
			outputs.add(new CPOperand(parts[4]));
			return new CompressionCPInstruction(null, in1, out, outputs, opcode, str, 0);
		}
		else if(parts.length == 4) {
			int treeNodeID = Integer.parseInt(parts[3]);
			return new CompressionCPInstruction(null, in1, out, opcode, str, treeNodeID);
		}
		else
			return new CompressionCPInstruction(null, in1, out, opcode, str, 0);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(input2 == null)
			processSimpleCompressInstruction(ec);
		else
			processCompressByBinInstruction(ec);
	}

	private void processCompressByBinInstruction(ExecutionContext ec) {
		final MatrixBlock d = ec.getMatrixInput(input2.getName());

		final int k = OptimizerUtils.getConstrainedNumThreads(-1);

		Pair<MatrixBlock, FrameBlock> out;

		if(ec.isMatrixObject(input1.getName())) {
			final MatrixBlock X = ec.getMatrixInput(input1.getName());
			out = CLALibBinCompress.binCompress(X, d, k);
			ec.releaseMatrixInput(input1.getName());
		} else {
			final FrameBlock X = ec.getFrameInput(input1.getName());
			out = CLALibBinCompress.binCompress(X, d, k);
			ec.releaseFrameInput(input1.getName());
		}
		
		// Set output and release input
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(_outputs.get(0).getName(), out.getKey());
		ec.setFrameOutput(_outputs.get(1).getName(), out.getValue());
	}

	private void processSimpleCompressInstruction(ExecutionContext ec) {
		// final MatrixBlock in = ec.getMatrixInput(input1.getName());
		final SingletonLookupHashMap m = SingletonLookupHashMap.getMap();

		// Get and clear workload tree entry for this compression instruction.
		final WTreeRoot root = (_singletonLookupID != 0) ? (WTreeRoot) m.get(_singletonLookupID) : null;
		m.removeKey(_singletonLookupID);

		final int k = OptimizerUtils.getConstrainedNumThreads(-1);

		if(ec.isFrameObject(input1.getName()))
			processFrameBlockCompression(ec, ec.getFrameInput(input1.getName()), k, root);
		else if(ec.isMatrixObject(input1.getName()))
			processMatrixBlockCompression(ec, ec.getMatrixInput(input1.getName()), k, root);
		else{
			throw new NotImplementedException("Not supported other types of input for compression than frame and matrix");
		}
	}

	private void processMatrixBlockCompression(ExecutionContext ec, MatrixBlock in, int k, WTreeRoot root) {
		Pair<MatrixBlock, CompressionStatistics> compResult = CompressedMatrixBlockFactory.compress(in, k, root);
		if(LOG.isTraceEnabled())
			LOG.trace(compResult.getRight());
		MatrixBlock out = compResult.getLeft();
		// Set output and release input
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), out);
	}

	private void processFrameBlockCompression(ExecutionContext ec, FrameBlock in, int k, WTreeRoot root) {
		FrameBlock compResult = FrameLibCompress.compress(in, k, root);
		// Set output and release input
		ec.releaseFrameInput(input1.getName());
		ec.setFrameOutput(output.getName(), compResult);
	}
}
